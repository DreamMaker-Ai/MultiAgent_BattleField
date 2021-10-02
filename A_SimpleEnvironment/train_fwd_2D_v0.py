"""
Homogeneous multi agent setting - for custom FWD model with Conv1D

if CONTINUAL, specify checkpoint
"""
import argparse
import gym
import datetime
import os
import random
import tempfile
import numpy as np
import pickle

import ray
from ray import tune
from ray.tune.logger import Logger, UnifiedLogger, pretty_print
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.examples.models.shared_weights_model import TF2SharedWeightsModel
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.ppo import ppo
from ray.rllib.models import ModelCatalog
from battle_field_strategy_2D_v0 import BattleFieldStrategy
from settings.initial_settings import *
from settings.reset_conditions import reset_conditions
#from modules.models import MyConv2DModel_v0B_Small_CBAM_1DConv_Share
from modules.models import DenseNetModelLargeShare
from tensorflow.keras.utils import plot_model
from modules.savers import save_conditions

# tf1, tf, tfv = try_import_tf()

import tensorflow as tf


def custom_log_creator(custom_path, custom_str):
    timestr = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def main():
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    ModelCatalog.register_custom_model('my_model', DenseNetModelLargeShare)

    config = {"env": BattleFieldStrategy,
              "num_workers": NUM_WORKERS,
              "num_gpus": NUM_GPUS,
              "num_cpus_per_worker": NUM_CPUS_PER_WORKER,
              "num_sgd_iter": NUM_SGD_ITER,
              "lr": LEARNING_RATE,
              "gamma": 0.99,  # default=0.99
              "model": {"custom_model": "my_model"}
              # "framework": framework
              }  # use tensorflow 2

    conditions_dir = os.path.join('./' + PROJECT + '/conditions/')
    if not os.path.exists(conditions_dir):
        os.makedirs(conditions_dir)
    save_conditions(conditions_dir)

    # PPOTrainer()は、try_import_tfを使うと、なぜかTensorflowのeager modeのエラーになる。
    trainer = ppo.PPOTrainer(config=config,
                             logger_creator=custom_log_creator(
                                 os.path.expanduser("./" + PROJECT + "/logs"), TRIAL))

    if CONTINUAL:
        # Continual learning: Need to specify the checkpoint
        model_path = PROJECT + '/checkpoints/' + 'fwd_20500/checkpoint_001901/checkpoint-1901'
        trainer.restore(checkpoint_path=model_path)

    models_dir = os.path.join('./' + PROJECT + '/models/')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    text_name = models_dir + TRIAL + '.txt'
    with open(text_name, "w") as fp:
        trainer.get_policy().model.base_model.summary(print_fn=lambda x: fp.write(x + "\r\n"))
    png_name = models_dir + TRIAL + '.png'
    plot_model(trainer.get_policy().model.base_model, to_file=png_name, show_shapes=True)

    # Instanciate the evaluation env
    eval_env = BattleFieldStrategy({})

    # Define checkpoint dir
    check_point_dir = os.path.join('./' + PROJECT + '/checkpoints/', TRIAL)
    if not os.path.exists(check_point_dir):
        os.makedirs(check_point_dir)

    # Training & evaluation
    results_dir = os.path.join('./' + PROJECT + '/results/')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file = results_dir + TRIAL + '.pkl'
    success_history = []
    fail_history = []
    draw_history = []
    steps_history = []
    alive_red_history = []
    alive_blue_history = []
    red_force_history = []
    blue_force_history = []
    episode_length_history = []
    for steps in range(10001):
        # Training
        results = trainer.train()

        # Evaluation
        if steps % EVAL_FREQ == 0:
            eval_result = []
            alive_red = []
            alive_blue = []
            red_force = []
            blue_force = []
            episode_length = []
            print(f'\n----------------- Evaluation at steps:{steps} starting ! -----------------')
            #print(pretty_print(results))
            check_point = trainer.save(checkpoint_dir=check_point_dir)

            for i in range(NUM_EVAL):
                # print(f'\nEvaluation {i}:')
                obs = eval_env.reset()
                done = False

                while not done:
                    action_dict = {}
                    for j in range(eval_env.num_red):
                        if eval_env.red.alive[j]:
                            action_dict['red_' + str(j)] = trainer.compute_action(obs['red_' + str(j)])

                    obs, rewards, dones, infos = eval_env.step(action_dict)
                    done = dones["__all__"]
                    #print(f'rewards:{rewards}')
                    eval_env.render(mode='rgb')

                if np.all(np.logical_not(eval_env.blue.alive)):
                    eval_result.append('success')
                elif np.all(np.logical_not(eval_env.red.alive)):
                    eval_result.append('fail')
                else:
                    eval_result.append('draw')

                alive_red.append(np.sum(eval_env.red.alive))
                alive_blue.append(np.sum(eval_env.blue.alive))

                red_force.append(np.sum(eval_env.red.force))
                blue_force.append(np.sum(eval_env.blue.force))

                episode_length.append(eval_env.steps)

            s = [m for m in range(len(eval_result)) if eval_result[m] == 'success']
            success = len(s)
            f = [m for m in range(len(eval_result)) if eval_result[m] == 'fail']
            fail = len(f)
            d = [m for m in range(len(eval_result)) if eval_result[m] == 'draw']
            draw = len(d)
            print(f'evaluation: success = {success},   fail = {fail},   draw = {draw}')

            print(f'final alive red = {np.mean(alive_red)},   '
                  f'final alive blue = {np.mean(alive_blue)}')
            print(f'final red force = {np.mean(red_force)},   '
                  f'final blue force = {np.mean(blue_force)}')
            print(f'episode length ={np.mean(episode_length)}')

            steps_history.append(steps)
            success_history.append(success)
            fail_history.append(fail)
            draw_history.append(draw)
            alive_red_history.append(np.mean(alive_red))
            alive_blue_history.append(np.mean(alive_blue))
            red_force_history.append(np.mean(red_force))
            blue_force_history.append(np.mean(blue_force))
            episode_length_history.append(np.mean(episode_length))

            f = open(results_file, 'wb')
            pickle.dump(steps_history, f)
            pickle.dump(success_history, f)
            pickle.dump(fail_history, f)
            pickle.dump(draw_history, f)
            pickle.dump(alive_red_history, f)
            pickle.dump(alive_blue_history, f)
            pickle.dump(red_force_history, f)
            pickle.dump(blue_force_history, f)
            pickle.dump(episode_length_history, f)
            f.close()

    ray.shutdown()


if __name__ == '__main__':
    main()
