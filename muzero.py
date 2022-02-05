import os
import pickle
import sys
from glob import glob

import numpy
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm

import diagnose_model
import models
import self_play
import trainer
import games.cartpole as game_module
from replay_buffer import ReplayBuffer, Reanalyse
from shared_storage import SharedStorage


class MuZero:
    """
    Main class to manage MuZero.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        config (dict, MuZeroConfig, optional): Override the default config of the game.

        split_resources_in (int, optional): Split the GPU usage when using concurent muzero instances.

    Example:
        >>> muzero = MuZero("cartpole")
        >>> muzero.train()
        >>> muzero.test(render=True)
    """

    def __init__(self):
        # Load the game and the config from the module with the game name
        self.game_cls = game_module.Game
        self.config = game_module.MuZeroConfig()

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Checkpoint and replay buffer used to initialize workers
        model = models.MuZeroNetwork(self.config)
        self.replay_buffer = {}
        self.checkpoint = {
            "weights": model.get_weights(),
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "num_tested_games": 0,
            "num_tested_steps": 0,
        }

        self.summary = str(model).replace("\n", " \n\n")

    def train(self, log_in_tensorboard=True):
        """
        Launch the training.

        Args:
            log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.
        """
        if log_in_tensorboard or self.config.save_model:
            os.makedirs(self.config.results_path, exist_ok=True)

        # Initialize workers
        self_play_worker = self_play.SelfPlay(
            self.checkpoint, self.game_cls, self.config, self.config.seed)
        test_worker = self_play.SelfPlay(
            self.checkpoint, self.game_cls, self.config, self.config.seed + self.config.num_workers)

        training_worker = trainer.Trainer(self.checkpoint, self.config)
        shared_storage = SharedStorage(self.checkpoint, self.config)
        replay_buffer = ReplayBuffer(self.checkpoint, self.replay_buffer, self.config)

        if self.config.use_last_model_value:
            reanalyse_worker = Reanalyse(self.checkpoint, self.config)
        else:
            reanalyse_worker = None

        with SummaryWriter(self.config.results_path) as writer:
            hp_table = [f"| {key} | {value} |" for key, value in self.config.__dict__.items()]
            writer.add_text("Hyperparameters", "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table))
            writer.add_text("Model summary", self.summary)

            try:
                with tqdm.trange(self.config.training_steps) as progress_bar:
                    for training_step in progress_bar:
                        while shared_storage.get_info("num_played_steps") * self.config.ratio <= training_step:
                            self_play_worker.self_play_once_train_mode(shared_storage, replay_buffer)

                        training_worker.update_weights_once(replay_buffer, shared_storage)
                        if reanalyse_worker is not None:
                            reanalyse_worker.reanalyse_once(replay_buffer, shared_storage)

                        # while shared_storage.get_info("num_tested_steps") * self.config.test_ratio <= training_step:
                        if (training_step + 1) % 500 == 0:
                            test_worker.self_play_once_test_mode(shared_storage)

                        self.log_once(shared_storage, writer, progress_bar, training_step)
            finally:
                self_play_worker.close_game()
                test_worker.close_game()

                self.checkpoint = shared_storage.get_checkpoint()
                self.replay_buffer = replay_buffer.get_buffer()

                if self.config.save_model:
                    # Persist replay buffer to disk
                    print("\n\nPersisting replay buffer games to disk...")
                    with open(os.path.join(self.config.results_path, "replay_buffer.pkl"), "wb") as fp:
                        pickle.dump(
                            {
                                "buffer": self.replay_buffer,
                                "num_played_games": self.checkpoint["num_played_games"],
                                "num_played_steps": self.checkpoint["num_played_steps"],
                                "num_reanalysed_games": self.checkpoint["num_reanalysed_games"],
                            },
                            fp,
                        )

    def log_once(self, shared_storage, writer, progress_bar, counter):
        keys = [
            "total_reward",
            "muzero_reward",
            "opponent_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "num_played_games",
            "num_played_steps",
            "num_tested_games",
            "num_tested_steps",
            "num_reanalysed_games",
        ]
        info = shared_storage.get_info(keys)
        writer.add_scalar("1.Total_reward/1.Total_reward", info["total_reward"], counter)
        writer.add_scalar("1.Total_reward/2.Mean_value", info["mean_value"], counter)
        writer.add_scalar("1.Total_reward/3.Episode_length", info["episode_length"], counter)
        writer.add_scalar("1.Total_reward/4.MuZero_reward", info["muzero_reward"], counter)
        writer.add_scalar("1.Total_reward/5.Opponent_reward", info["opponent_reward"], counter)

        writer.add_scalar("2.Workers/1.Self_played_games", info["num_played_games"], counter)
        writer.add_scalar("2.Workers/2.Training_steps", info["training_step"], counter)
        writer.add_scalar("2.Workers/3.Self_played_steps", info["num_played_steps"], counter)
        writer.add_scalar("2.Workers/4.Reanalysed_games", info["num_reanalysed_games"], counter)
        writer.add_scalar(
            "2.Workers/5.Training_steps_per_self_played_step_ratio",
            info["training_step"] / max(1, info["num_played_steps"]),
            counter,
        )
        writer.add_scalar("2.Workers/6.Learning_rate", info["lr"], counter)

        writer.add_scalar("3.Loss/1.Total_weighted_loss", info["total_loss"], counter)
        writer.add_scalar("3.Loss/Value_loss", info["value_loss"], counter)
        writer.add_scalar("3.Loss/Reward_loss", info["reward_loss"], counter)
        writer.add_scalar("3.Loss/Policy_loss", info["policy_loss"], counter)

        progress_bar.set_postfix_str(
            '. '.join([
                f'Last test reward: {info["total_reward"]:.2f}',
                f'Training step: {info["training_step"]}/{self.config.training_steps}',
                f'Played games|steps: {info["num_played_games"]}|{info["num_played_steps"]}',
                f'Tested games|steps: {info["num_tested_games"]}|{info["num_tested_steps"]}',
                f'Reanalysed games: {info["num_reanalysed_games"]}',
                f'Loss: {info["total_loss"]:.2f}',
            ])
        )

    def test(self, render=True, opponent=None, muzero_player=None, num_tests=1):
        """
        Test the model in a dedicated thread.

        Args:
            render (bool): To display or not the environment. Defaults to True.

            opponent (str): "self" for self-play, "human" for playing against MuZero and "random"
            for a random agent, None will use the opponent in the config. Defaults to None.

            muzero_player (int): Player number of MuZero in case of multiplayer
            games, None let MuZero play all players turn by turn, None will use muzero_player in
            the config. Defaults to None.

            num_tests (int): Number of games to average. Defaults to 1.
        """
        opponent = opponent if opponent else self.config.opponent
        muzero_player = muzero_player if muzero_player else self.config.muzero_player
        self_play_worker = self_play.SelfPlay(self.checkpoint, self.game_cls, self.config, numpy.random.randint(10000))
        results = []
        for i in range(num_tests):
            print(f"Testing {i+1}/{num_tests}")
            results.append(
                self_play_worker.play_game(
                    0, 0, render, opponent, muzero_player,
                )
            )
        self_play_worker.close_game()

        if len(self.config.players) == 1:
            result = numpy.mean([sum(history.reward_history) for history in results])
        else:
            result = numpy.mean([
                sum(
                    reward for i, reward in enumerate(history.reward_history)
                    if history.to_play_history[i - 1] == muzero_player
                )
                for history in results
            ])
        return result

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        """
        Load a model and/or a saved replay buffer.

        Args:
            checkpoint_path (str): Path to model.checkpoint or model.weights.

            replay_buffer_path (str): Path to replay_buffer.pkl
        """
        # Load checkpoint
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                self.checkpoint = torch.load(checkpoint_path)
                print(f"\nUsing checkpoint from {checkpoint_path}")
            else:
                print(f"\nThere is no model saved in {checkpoint_path}.")

        # Load replay buffer
        if replay_buffer_path:
            if os.path.exists(replay_buffer_path):
                with open(replay_buffer_path, "rb") as f:
                    replay_buffer_infos = pickle.load(f)
                self.replay_buffer = replay_buffer_infos["buffer"]
                self.checkpoint["num_played_steps"] = replay_buffer_infos[
                    "num_played_steps"
                ]
                self.checkpoint["num_played_games"] = replay_buffer_infos[
                    "num_played_games"
                ]
                self.checkpoint["num_reanalysed_games"] = replay_buffer_infos[
                    "num_reanalysed_games"
                ]

                print(f"\nInitializing replay buffer with {replay_buffer_path}")
            else:
                print(
                    f"Warning: Replay buffer path '{replay_buffer_path}' doesn't exist.  Using empty buffer."
                )
                self.checkpoint["training_step"] = 0
                self.checkpoint["num_played_steps"] = 0
                self.checkpoint["num_played_games"] = 0
                self.checkpoint["num_reanalysed_games"] = 0

    def diagnose_model(self, horizon):
        """
        Play a game only with the learned model then play the same trajectory in the real
        environment and display information.

        Args:
            horizon (int): Number of timesteps for which we collect information.
        """
        game = self.game_cls(self.config.seed)
        obs = game.reset()
        dm = diagnose_model.DiagnoseModel(self.checkpoint, self.config)
        dm.compare_virtual_with_real_trajectories(obs, game, horizon)
        input("Press enter to close all plots")
        dm.close_all()


def load_model_menu(muzero, game_name):
    # Configure running options
    options = ["Specify paths manually"] + sorted(glob(f"results/{game_name}/*/"))
    options.reverse()
    print()
    for i in range(len(options)):
        print(f"{i}. {options[i]}")

    choice = input("Enter a number to choose a model to load: ")
    valid_inputs = [str(i) for i in range(len(options))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")
    choice = int(choice)

    if choice == (len(options) - 1):
        # manual path option
        checkpoint_path = input(
            "Enter a path to the model.checkpoint, or ENTER if none: "
        )
        while checkpoint_path and not os.path.isfile(checkpoint_path):
            checkpoint_path = input("Invalid checkpoint path. Try again: ")
        replay_buffer_path = input(
            "Enter a path to the replay_buffer.pkl, or ENTER if none: "
        )
        while replay_buffer_path and not os.path.isfile(replay_buffer_path):
            replay_buffer_path = input("Invalid replay buffer path. Try again: ")
    else:
        checkpoint_path = f"{options[choice]}model.checkpoint"
        replay_buffer_path = f"{options[choice]}replay_buffer.pkl"

    muzero.load_model(
        checkpoint_path=checkpoint_path, replay_buffer_path=replay_buffer_path,
    )


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Train directly with "python muzero.py cartpole"
        muzero = MuZero()
        muzero.train()
    else:
        print("\nWelcome to MuZero! Here's a list of games:")
        # Let user pick a game
        games = [
            filename[:-3]
            for filename in sorted(
                os.listdir(os.path.dirname(os.path.realpath(__file__)) + "/games")
            )
            if filename.endswith(".py") and filename != "abstract_game.py"
        ]
        for i in range(len(games)):
            print(f"{i}. {games[i]}")
        choice = input("Enter a number to choose the game: ")
        valid_inputs = [str(i) for i in range(len(games))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")

        # Initialize MuZero
        choice = int(choice)
        game_name = games[choice]
        muzero = MuZero(game_name)

        while True:
            # Configure running options
            options = [
                "Train",
                "Load pretrained model",
                "Diagnose model",
                "Render some self play games",
                "Play against MuZero",
                "Test the game manually",
                "Exit",
            ]
            print()
            for i in range(len(options)):
                print(f"{i}. {options[i]}")

            choice = input("Enter a number to choose an action: ")
            valid_inputs = [str(i) for i in range(len(options))]
            while choice not in valid_inputs:
                choice = input("Invalid input, enter a number listed above: ")
            choice = int(choice)
            if choice == 0:
                muzero.train()
            elif choice == 1:
                load_model_menu(muzero, game_name)
            elif choice == 2:
                muzero.diagnose_model(30)
            elif choice == 3:
                muzero.test(render=True, opponent="self", muzero_player=None)
            elif choice == 4:
                muzero.test(render=True, opponent="human", muzero_player=0)
            elif choice == 5:
                env = muzero.Game()
                env.reset()
                env.render()

                done = False
                while not done:
                    action = env.human_to_action()
                    observation, reward, done = env.step(action)
                    print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
                    env.render()
            else:
                break
            print("\nDone")
