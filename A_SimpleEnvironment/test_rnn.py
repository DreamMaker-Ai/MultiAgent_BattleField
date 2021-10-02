"""
For RNN test
"""
import argparse
import gym
import datetime
import os
import random
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
from battle_field_strategy_v3 import BattleFieldStrategy
from settings.test_settings import *
from modules.models import MyRNNConv1DModel_v3
from tensorflow.keras.utils import plot_model

# tf1, tf, tfv = try_import_tf()

import tensorflow as tf

model_path = PROJECT + '/checkpoints/' + TRIAL + '/checkpoint_000501/checkpoint-501'

results_path = PROJECT + '/results/' + TRIAL + '/'
if not os.path.exists(results_path):
    os.makedirs(results_path)


def get_engage_image_1(env):
    red_matrix = np.zeros(env.grid_size)
    blue_matrix = np.zeros(env.grid_size)

    for j in range(env.num_red):
        if env.red.alive[j]:
            red_matrix[env.red.pos[j]] += env.red.force[j]

    for j in range(env.num_blue):
        if env.blue.alive[j]:
            blue_matrix[env.blue.pos[j]] += env.blue.force[j]

    engage_matrix = red_matrix - blue_matrix
    engage_matrix = np.expand_dims(engage_matrix, axis=1)
    img = plt.imshow(engage_matrix, vmin=-env.max_force, vmax=env.max_force, animated=True, cmap='bwr')
    plt.tick_params(labelbottom=False, bottom=False)
    return img


def get_engage_image_2(env):
    red_matrix = np.zeros((env.grid_size, env.num_red))
    blue_matrix = np.zeros((env.grid_size, env.num_blue))

    for j in range(env.num_red):
        if env.red.alive[j]:
            red_matrix[env.red.pos[j], j] += env.red.force[j]

    for j in range(env.num_blue):
        if env.blue.alive[j]:
            blue_matrix[env.blue.pos[j], j] -= env.blue.force[j]

    total_matrix = np.hstack([red_matrix, blue_matrix])
    img = plt.imshow(total_matrix, vmin=-env.max_force, vmax=env.max_force, animated=True, cmap='bwr')
    plt.tick_params(labelbottom=False, bottom=False)
    return img


def get_engage_image_3(env):
    red_matrix = np.zeros((env.grid_size, env.num_red))
    blue_matrix = np.zeros((env.grid_size, env.num_blue))

    for j in range(env.num_red):
        if env.red.alive[j]:
            red_matrix[env.red.pos[j], j] += env.red.force[j]

    for j in range(env.num_blue):
        if env.blue.alive[j]:
            blue_matrix[env.blue.pos[j], j] -= env.blue.force[j]

    red_team = np.sum(red_matrix, axis=1, keepdims=True)
    blue_team = np.sum(blue_matrix, axis=1, keepdims=True)

    spaces = np.zeros((env.grid_size, 1))

    total_matrix = np.hstack([red_team + blue_team, spaces, red_matrix, blue_matrix])
    img = plt.imshow(total_matrix, vmin=-env.max_force, vmax=env.max_force, animated=True, cmap='bwr')
    plt.tick_params(labelbottom=False, bottom=False)
    return img


def main():
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    ModelCatalog.register_custom_model('my_model', MyRNNConv1DModel_v3)

    config = {"env": BattleFieldStrategy,
              "model": {"custom_model": "my_model",
                        "max_seq_len": 20}
              # "framework": framework
              }  # use tensorflow 2

    # PPOTrainer()は、try_import_tfを使うと、なぜかTensorflowのeager modeのエラーになる。
    trainer = ppo.PPOTrainer(config=config)
    trainer.restore(checkpoint_path=model_path)
    trainer.get_policy().model.rnn_model.summary()
    # restore_model_name = results_path + 'rnn_model.png'
    # plot_model(trainer.get_policy().model.rnn_model, to_file=restore_model_name, show_shapes=True)

    # Instanciate the evaluation env
    eval_env = BattleFieldStrategy({})

    # Evaluation
    eval_result = []
    alive_red = []
    alive_blue = []
    red_force = []
    blue_force = []
    steps = []
    success_steps = []
    fail_steps = []
    draw_steps = []

    for i in range(NUM_TEST):
        fig = plt.figure()
        ims = []
        print(f'\nEvaluation {i}:')
        obs = eval_env.reset()
        if MAKE_ANIMATION:
            im = get_engage_image_3(eval_env)
            ims.append([im])
        done = False
        state = trainer.get_policy().get_initial_state()

        while not done:
            action_dict = {}
            for j in range(eval_env.num_red):
                if eval_env.red.alive[j]:
                    act, state, _ = trainer.compute_action(obs['red_' + str(j)], state)
                    action_dict['red_' + str(j)] = act

            obs, rewards, dones, infos = eval_env.step(action_dict)

            if MAKE_ANIMATION:
                im = get_engage_image_3(eval_env)
                ims.append([im])

            done = dones["__all__"]
            # print(f'step: {eval_env.steps}, done:{done}, rewards:{rewards}, '
            #      f'action:{action_dict}, obs:{obs}')

            eval_env.render()

        steps.append(eval_env.steps)
        if np.all(np.logical_not(eval_env.blue.alive)):
            eval_result.append('success')
            success_steps.append(eval_env.steps)
        elif np.all(np.logical_not(eval_env.red.alive)):
            eval_result.append('fail')
            fail_steps.append(eval_env.steps)
        else:
            eval_result.append('draw')
            draw_steps.append(eval_env.steps)

        alive_red.append(np.sum(eval_env.red.alive))
        alive_blue.append(np.sum(eval_env.blue.alive))

        red_force.append(np.sum(eval_env.red.force))
        blue_force.append(np.sum(eval_env.blue.force))

        if MAKE_ANIMATION:
            anim = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=5000)
            anim_name = results_path + 'anim_' + str(i)
            anim.save(anim_name + '.gif', writer='imagemagick')
            anim.save(anim_name + '.mp4', writer='ffmpeg')

    print(f'step length: mean={np.mean(steps)}, std={np.std(steps)}')
    if len(success_steps) > 0:
        print(f'   success step length: mean={np.mean(success_steps)}, std={np.std(success_steps)}')
    if len(fail_steps) > 0:
        print(f'   fail step length: mean={np.mean(fail_steps)}, std={np.std(fail_steps)}')
    if len(draw_steps) > 0:
        print(f'   draw step length: mean={np.mean(draw_steps)}, std={np.std(draw_steps)}')

    s = [m for m in range(len(eval_result)) if eval_result[m] == 'success']
    success = len(s)
    f = [m for m in range(len(eval_result)) if eval_result[m] == 'fail']
    fail = len(f)
    d = [m for m in range(len(eval_result)) if eval_result[m] == 'draw']
    draw = len(d)
    print(f'evaluation: success = {success},   fail = {fail},   draw = {draw}')

    print(f'final alive red: mean={np.mean(alive_red)}, std={np.std(alive_red)}   '
          f'final alive blue: mean={np.mean(alive_blue)}, std={np.std(alive_blue)}')
    print(f'final red force: mean={np.mean(red_force)}, std={np.std(red_force)}   '
          f'final blue force: mean={np.mean(blue_force)}, std={np.std(blue_force)}')

    ray.shutdown()


if __name__ == '__main__':
    # framework = "tf2"
    # as_test = True
    main()
