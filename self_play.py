from typing import Optional

import math

import numpy as np
import torch

import models


class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_checkpoint, game_cls, config, seed):
        self.config = config
        self.game = game_cls(seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if config.selfplay_on_gpu else "cpu"))
        self.model.eval()

    def self_play_once_train_mode(self, shared_storage, replay_buffer):
        self.model.set_weights(shared_storage.get_info("weights"))

        game_history = self.play_game(
            self.config.visit_softmax_temperature_fn(trained_steps=shared_storage.get_info("training_step")),
            self.config.temperature_threshold,
            render=False,
        )

        replay_buffer.save_game(game_history, shared_storage)

    def self_play_once_test_mode(self, shared_storage):
        self.model.set_weights(shared_storage.get_info("weights"))

        # Take the best action (no exploration) in test mode
        game_history = self.play_game(
            0,
            self.config.temperature_threshold,
            render=False,
        )

        info = shared_storage.get_info(["num_tested_games", "num_tested_steps"])

        # Save to the shared storage
        shared_storage.set_info(
            {
                "episode_length": len(game_history.action_history) - 1,
                "total_reward": sum(game_history.reward_history),
                "mean_value": np.mean(
                    [value for value in game_history.root_values if value]
                ),
                "num_tested_games": info["num_tested_games"] + 1,
                "num_tested_steps": info["num_tested_steps"] + len(game_history.root_values),
            }
        )

    def play_game(
            self,
            temperature: float,
            temperature_threshold: Optional[float],
            render: bool,
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)

        done = False

        if render:
            self.game.render()

        with torch.no_grad():
            while not done and len(game_history.action_history) <= self.config.max_moves:
                assert (
                    len(np.array(observation).shape) == 3
                ), f"Observation should be 3 dimensional instead of {len(np.array(observation).shape)} dimensional. Got observation of shape: {np.array(observation).shape}"
                assert (
                    np.array(observation).shape == self.config.observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {np.array(observation).shape}."

                last_observation = game_history.get_observation(-1)

                # Choose the action
                root, mcts_info = MCTS(self.config).run(
                    model=self.model,
                    observation=last_observation,
                    legal_actions=self.game.legal_actions(),
                    add_exploration_noise=True,
                )

                if temperature_threshold is None or len(game_history.action_history) < temperature_threshold:
                    action_temperature = temperature
                else:
                    # Optionally select action deterministically after a certain number of steps
                    action_temperature = 0
                action = self.select_action(root, action_temperature)

                if render:
                    print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                    print(f"Root value for player: {root.value():.2f}")

                observation, reward, done = self.game.step(action)

                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

                game_history.store_search_statistics(root, self.config.action_space)

                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)

        return game_history

    def close_game(self):
        self.game.close()

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = np.array([child.visit_count for child in node.children.values()], dtype=np.int32)
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action


# Game independent
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(self, model, observation, legal_actions, add_exploration_noise: bool):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        assert (
            legal_actions
        ), f"Legal actions should not be an empty array. Got {legal_actions}."
        assert set(legal_actions).issubset(
            set(self.config.action_space)
        ), "Legal actions should be a subset of the action space."

        root = Node(0)
        device = next(model.parameters()).device
        observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        (
            root_predicted_value,
            reward,
            policy_logits,
            hidden_state,
        ) = model.initial_inference(observation)
        root_predicted_value = models.support_to_scalar(root_predicted_value, self.config.support_size).item()
        reward = models.support_to_scalar(reward, self.config.support_size).item()

        min_max_stats = MinMaxStats()

        # Expanding root to enable adding exploration noise
        root.expand(legal_actions, reward, policy_logits, hidden_state)

        # Update root value and visit count
        self.backpropagate([root], root_predicted_value, min_max_stats)

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[action]], device=parent.hidden_state.device),
            )
            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            node.expand(self.config.action_space, reward, policy_logits, hidden_state)

            self.backpropagate(search_path, value, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        actions_with_ucb = [
            (action, self.ucb_score(node, child, min_max_stats))
            for action, child in node.children.items()
        ]
        max_ucb = max(ucb for action, ucb in actions_with_ucb)
        action = np.random.choice([
            action
            for action, ucb in actions_with_ucb
            if ucb == max_ucb
        ])
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base)
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(child.reward + self.config.discount * child.value())
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.reward + self.config.discount * node.value())

            value = node.reward + self.config.discount * value


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.reward = reward
        self.hidden_state = hidden_state

        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class GameHistory:
    """
    Store only useful information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])

        self.root_values.append(root.value())

    def get_observation(self, index):
        return self.observation_history[index].copy()


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
