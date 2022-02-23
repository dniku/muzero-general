import copy

import numpy as np
import torch
import torch.nn.functional as F

import models


class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        self.model.train()

        self.training_step = initial_checkpoint["training_step"]

        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on GPU.\n")

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr_init,
            weight_decay=self.config.weight_decay,
        )

        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(copy.deepcopy(initial_checkpoint["optimizer_state"]))

    def update_weights_once(self, replay_buffer, shared_storage):
        next_batch = replay_buffer.get_batch()
        index_batch, batch = next_batch

        self.update_lr()
        (
            priorities,
            total_loss,
            value_loss,
            reward_loss,
            policy_loss,
        ) = self.update_weights(batch)

        if self.config.PER:
            # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
            replay_buffer.update_priorities(priorities, index_batch)

        # Save to the shared storage
        if self.training_step % self.config.checkpoint_interval == 0:
            shared_storage.set_info({
                "weights": copy.deepcopy(self.model.get_weights()),
                "optimizer_state": copy.deepcopy(
                    models.dict_to_cpu(self.optimizer.state_dict())
                ),
            })
            if self.config.save_model:
                shared_storage.save_checkpoint()
        shared_storage.set_info(
            {
                "training_step": self.training_step,
                "lr": self.optimizer.param_groups[0]["lr"],
                "total_loss": total_loss,
                "value_loss": value_loss,
                "reward_loss": reward_loss,
                "policy_loss": policy_loss,
            }
        )

    def update_weights(self, batch):
        """
        Perform one training step.
        """

        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            gradient_scale_batch,
        ) = batch

        # Keep values as scalars for calculating the priorities for the prioritized replay
        target_value_scalar = np.array(target_value, dtype=np.float32)
        priorities = np.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device
        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy(), dtype=torch.float32, device=device)
        observation_batch = torch.tensor(observation_batch, dtype=torch.float32, device=device)
        action_batch = torch.tensor(action_batch, dtype=torch.int64, device=device).unsqueeze(dim=-1)
        target_value = torch.tensor(target_value, dtype=torch.float32, device=device)
        target_reward = torch.tensor(target_reward, dtype=torch.float32, device=device)
        target_policy = torch.tensor(target_policy, dtype=torch.float32, device=device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch, dtype=torch.float32, device=device)
        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
        # target_value: batch, num_unroll_steps+1
        # target_reward: batch, num_unroll_steps+1
        # target_policy: batch, num_unroll_steps+1, len(action_space)
        # gradient_scale_batch: batch, num_unroll_steps+1

        target_value = models.scalar_to_support(target_value, self.config.support_size)
        target_reward = models.scalar_to_support(target_reward, self.config.support_size)
        # target_value: batch, num_unroll_steps+1, 2*support_size+1
        # target_reward: batch, num_unroll_steps+1, 2*support_size+1

        ## Generate predictions by unrolling model
        value, reward, policy_logits, hidden_state = self.model.initial_inference(observation_batch)
        predictions = [(value, reward, policy_logits)]
        for i in range(1, action_batch.shape[1]):
            value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                hidden_state, action_batch[:, i]
            )
            # Scale the gradient at the start of the dynamics function (See paper appendix Training)
            hidden_state.register_hook(lambda grad: grad * 0.5)
            predictions.append((value, reward, policy_logits))
        # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)

        ## Compute losses
        value_loss, reward_loss, policy_loss = (0, 0, 0)

        ### Handle first timestep in a special way
        value, reward, policy_logits = predictions[0]
        # Ignore reward loss for the first timestep step b/c reward prediction is forced to 0 and loss is NaN
        current_value_loss, _, current_policy_loss = self.loss_function(
            value.squeeze(axis=-1),
            reward.squeeze(axis=-1),
            policy_logits,
            target_value[:, 0],
            target_reward[:, 0],
            target_policy[:, 0],
        )
        value_loss += current_value_loss
        policy_loss += current_policy_loss
        # Compute priorities for the prioritized replay (See paper appendix Training)
        pred_value_scalar = (
            models.support_to_scalar(value, self.config.support_size)
            .detach()
            .cpu()
            .numpy()
            .squeeze(axis=-1)
        )
        priorities[:, 0] = self.compute_priorities(pred_value_scalar, target_value_scalar[:, 0])

        ### Handle remaining timesteps
        for i in range(1, len(predictions)):
            value, reward, policy_logits = predictions[i]
            (
                current_value_loss,
                current_reward_loss,
                current_policy_loss,
            ) = self.loss_function(
                value.squeeze(axis=-1),
                reward.squeeze(axis=-1),
                policy_logits,
                target_value[:, i],
                target_reward[:, i],
                target_policy[:, i],
            )

            # Scale gradient by the number of unroll steps (See paper appendix Training)
            # Note: i=i is due to a peculiarity in how lambdas capture parameters in closures
            current_value_loss.register_hook(lambda grad, i=i: grad / gradient_scale_batch[:, i])
            current_reward_loss.register_hook(lambda grad, i=i: grad / gradient_scale_batch[:, i])
            current_policy_loss.register_hook(lambda grad, i=i: grad / gradient_scale_batch[:, i])

            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss

            # Compute priorities for the prioritized replay (See paper appendix Training)
            pred_value_scalar = (
                models.support_to_scalar(value, self.config.support_size)
                .detach()
                .cpu()
                .numpy()
                .squeeze(axis=-1)
            )
            priorities[:, i] = self.compute_priorities(pred_value_scalar, target_value_scalar[:, i])

        # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
        loss = value_loss * self.config.value_loss_weight + reward_loss + policy_loss
        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            loss *= weight_batch
        # Mean over batch dimension (pseudocode do a sum)
        loss = loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return (
            priorities,
            # For log purpose
            loss.item(),
            value_loss.mean().item(),
            reward_loss.mean().item(),
            policy_loss.mean().item(),
        )

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (self.training_step / self.config.lr_decay_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @staticmethod
    def loss_function(
        value,
        reward,
        policy_logits,
        target_value,
        target_reward,
        target_policy,
    ):
        # Cross-entropy seems to have a better convergence than MSE
        value_loss = F.cross_entropy(value, target_value, reduction='none')
        reward_loss = F.cross_entropy(reward, target_reward, reduction='none')
        policy_loss = F.cross_entropy(policy_logits, target_policy, reduction='none')

        return value_loss, reward_loss, policy_loss

    def compute_priorities(self, predicted, target):
        assert isinstance(predicted, np.ndarray)
        assert isinstance(target, np.ndarray)
        assert predicted.shape == target.shape

        return np.abs(predicted - target) ** self.config.PER_alpha
