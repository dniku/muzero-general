import os
import pickle
from pathlib import Path
import random

import numpy
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm

import models
import self_play
import trainer
import games.cartpole as game_module
from replay_buffer import ReplayBuffer, Reanalyse
from shared_storage import SharedStorage


def fix_random_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def log_progress(shared_storage, writer, progress_bar, counter, total_training_steps):
    keys = [
        'total_reward',
        'muzero_reward',
        'opponent_reward',
        'episode_length',
        'mean_value',
        'training_step',
        'lr',
        'total_loss',
        'value_loss',
        'reward_loss',
        'policy_loss',
        'num_played_games',
        'num_played_steps',
        'num_tested_games',
        'num_tested_steps',
        'num_reanalysed_games',
    ]
    info = shared_storage.get_info(keys)
    writer.add_scalar('1.Total_reward/1.Total_reward', info['total_reward'], counter)
    writer.add_scalar('1.Total_reward/2.Mean_value', info['mean_value'], counter)
    writer.add_scalar('1.Total_reward/3.Episode_length', info['episode_length'], counter)
    writer.add_scalar('1.Total_reward/4.MuZero_reward', info['muzero_reward'], counter)
    writer.add_scalar('1.Total_reward/5.Opponent_reward', info['opponent_reward'], counter)

    writer.add_scalar('2.Workers/1.Self_played_games', info['num_played_games'], counter)
    writer.add_scalar('2.Workers/2.Training_steps', info['training_step'], counter)
    writer.add_scalar('2.Workers/3.Self_played_steps', info['num_played_steps'], counter)
    writer.add_scalar('2.Workers/4.Reanalysed_games', info['num_reanalysed_games'], counter)
    writer.add_scalar(
        '2.Workers/5.Training_steps_per_self_played_step_ratio',
        info['training_step'] / max(1, info['num_played_steps']),
        counter,
    )
    writer.add_scalar('2.Workers/6.Learning_rate', info['lr'], counter)

    writer.add_scalar('3.Loss/1.Total_weighted_loss', info['total_loss'], counter)
    writer.add_scalar('3.Loss/Value_loss', info['value_loss'], counter)
    writer.add_scalar('3.Loss/Reward_loss', info['reward_loss'], counter)
    writer.add_scalar('3.Loss/Policy_loss', info['policy_loss'], counter)

    progress_bar.set_postfix_str(
        '. '.join([
            f'Last test reward: {info["total_reward"]:.2f}',
            f'Training step: {info["training_step"]}/{total_training_steps}',
            f'Played games|steps: {info["num_played_games"]}|{info["num_played_steps"]}',
            f'Tested games|steps: {info["num_tested_games"]}|{info["num_tested_steps"]}',
            f'Reanalysed games: {info["num_reanalysed_games"]}',
            f'Loss: {info["total_loss"]:.2f}',
        ])
    )


if __name__ == '__main__':
    # Load the game and the config from the module with the game name
    game_cls = game_module.Game
    config = game_module.MuZeroConfig()

    fix_random_seed(config.seed)

    # Checkpoint and replay buffer used to initialize workers
    model = models.MuZeroNetwork(config)
    replay_buffer = {}
    checkpoint = {
        'weights': model.get_weights(),
        'optimizer_state': None,
        'total_reward': 0,
        'muzero_reward': 0,
        'opponent_reward': 0,
        'episode_length': 0,
        'mean_value': 0,
        'training_step': 0,
        'lr': 0,
        'total_loss': 0,
        'value_loss': 0,
        'reward_loss': 0,
        'policy_loss': 0,
        'num_played_games': 0,
        'num_played_steps': 0,
        'num_reanalysed_games': 0,
        'num_tested_games': 0,
        'num_tested_steps': 0,
    }
    model_summary = str(model).replace('\n', ' \n\n')
    del model  # free up memory

    Path(config.results_path).mkdir(parents=True, exist_ok=True)

    # Initialize workers
    self_play_worker = self_play.SelfPlay(checkpoint, game_cls, config, config.seed)
    test_worker = self_play.SelfPlay(checkpoint, game_cls, config, config.seed + 1)

    training_worker = trainer.Trainer(checkpoint, config)
    shared_storage = SharedStorage(checkpoint, config)
    replay_buffer = ReplayBuffer(checkpoint, replay_buffer, config)

    if config.use_last_model_value:
        reanalyse_worker = Reanalyse(checkpoint, config)
    else:
        reanalyse_worker = None

    with SummaryWriter(config.results_path) as writer:
        hp_table = [f'| {key} | {value} |' for key, value in config.__dict__.items()]
        writer.add_text('Hyperparameters', '| Parameter | Value |\n|-------|-------|\n' + '\n'.join(hp_table))
        writer.add_text('Model summary', model_summary)

        try:
            with tqdm.trange(config.training_steps) as progress_bar:
                for training_step in progress_bar:
                    while shared_storage.get_info('num_played_steps') * config.ratio <= training_step:
                        self_play_worker.self_play_once_train_mode(shared_storage, replay_buffer)

                    training_worker.update_weights_once(replay_buffer, shared_storage)
                    if reanalyse_worker is not None:
                        reanalyse_worker.reanalyse_once(replay_buffer, shared_storage)

                    if (training_step + 1) % config.test_period == 0:
                        test_worker.self_play_once_test_mode(shared_storage)

                    log_progress(shared_storage, writer, progress_bar, training_step, config.training_steps)
        finally:
            self_play_worker.close_game()
            test_worker.close_game()

            checkpoint = shared_storage.get_checkpoint()
            replay_buffer_contents = replay_buffer.get_buffer()

            if config.save_model:
                # Persist replay buffer to disk
                print('\n\nPersisting replay buffer games to disk...')
                with open(os.path.join(config.results_path, 'replay_buffer.pkl'), 'wb') as fp:
                    pickle.dump(
                        {
                            'buffer': replay_buffer_contents,
                            'num_played_games': checkpoint['num_played_games'],
                            'num_played_steps': checkpoint['num_played_steps'],
                            'num_reanalysed_games': checkpoint['num_reanalysed_games'],
                        },
                        fp,
                    )
