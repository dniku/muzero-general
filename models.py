import torch
from torch import nn


SUPPORTS_CACHE = {}
def get_support(size, shape, device):
    key = size, shape, device
    if key not in SUPPORTS_CACHE:
        SUPPORTS_CACHE[key] = (
            torch.tensor([x for x in range(-size, size + 1)])
            .expand(shape)
            .float()
            .to(device=device)
        )
    return SUPPORTS_CACHE[key]


def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = get_support(support_size, probabilities.shape, probabilities.device)
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


def make_mlp(input_size, layer_sizes, output_size, output_activation=nn.Identity, activation=nn.ELU):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


def min_max_scale(tensor, eps=1e-5):
    min_tensor = tensor.min(1, keepdim=True)[0]
    max_tensor = tensor.max(1, keepdim=True)[0]
    scale_tensor = max_tensor - min_tensor
    scale_tensor[scale_tensor < eps] += eps
    tensor_normalized = (tensor - min_tensor) / scale_tensor
    return tensor_normalized


class MuZeroNetwork:
    def __new__(cls, config):
        return MuZeroFullyConnectedNetwork(
            config.observation_shape,
            config.stacked_observations,
            len(config.action_space),
            config.encoding_size,
            config.fc_reward_layers,
            config.fc_value_layers,
            config.fc_policy_layers,
            config.fc_representation_layers,
            config.fc_dynamics_layers,
            config.support_size,
        )


class MuZeroFullyConnectedNetwork(nn.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        observation_size = (
            observation_shape[0]
            * observation_shape[1]
            * observation_shape[2]
            * (stacked_observations + 1)
            + stacked_observations * observation_shape[1] * observation_shape[2]
        )
        self.representation_network = make_mlp(observation_size, fc_representation_layers, encoding_size)
        self.dynamics_encoded_state_network = make_mlp(encoding_size + self.action_space_size, fc_dynamics_layers, encoding_size)
        self.dynamics_reward_network = make_mlp(encoding_size, fc_reward_layers, self.full_support_size)
        self.prediction_policy_network = make_mlp(encoding_size, fc_policy_layers, self.action_space_size)
        self.prediction_value_network = make_mlp(encoding_size, fc_value_layers, self.full_support_size)

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation.view(observation.shape[0], -1))

        # Scale encoded state between [0, 1] (See appendix paper Training)
        encoded_state_normalized = min_max_scale(encoded_state)

        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = torch.zeros((action.shape[0], self.action_space_size)).to(action.device).float()
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        next_encoded_state_normalized = min_max_scale(next_encoded_state)

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        batch_size = observation.shape[0]
        # Since we replace regression with classification, reward is represented as a vector instead of 1 value.
        # The result of the following lines looks like log([0, ..., 0, 1, 0, ..., 0]), which means that we place
        # a likelihood of 1 onto the bin that corresponds to reward=0.
        reward = torch.log(
            torch.zeros(1, self.full_support_size)
            .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
            .repeat(batch_size, 1)
            .to(observation.device)
        )

        return value, reward, policy_logits, encoded_state

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)
