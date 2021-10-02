"""
For FWD test
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
from battle_field_strategy_2D_v0 import BattleFieldStrategy
from settings.initial_settings import *
from modules.models import MyConv2DModel_v0B_Small_CBAM_1DConv_Share  # Need to specify proper model
from tensorflow.keras.utils import plot_model
import json

# tf1, tf, tfv = try_import_tf()

import tensorflow as tf

model_path = PROJECT + '/checkpoints/' + TRIAL + '/checkpoint_010001/checkpoint-10001'

results_path = PROJECT + '/results/' + TRIAL + '/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

test_results_file = results_path + 'test_' + str(TEST_ID) + '.json'

def get_engage_image_1(env):
    red_matrix = np.zeros((env.grid_size, env.grid_size))
    blue_matrix = np.zeros((env.grid_size, env.grid_size))

    for j in range(env.num_red):
        if env.red.alive[j]:
            red_matrix[env.red.pos[j][0], env.red.pos[j][1]] += env.red.force[j]

    for j in range(env.num_blue):
        if env.blue.alive[j]:
            blue_matrix[env.blue.pos[j][0], env.blue.pos[j][1]] += env.blue.force[j]

    engage_matrix = red_matrix - blue_matrix
    engage_matrix = np.expand_dims(engage_matrix, axis=-1)
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

    ModelCatalog.register_custom_model('my_model', MyConv2DModel_v0B_Small_CBAM_1DConv_Share)

    config = {"env": BattleFieldStrategy,
              "model": {"custom_model": "my_model"}
              # "framework": framework
              }  # use tensorflow 2

    # PPOTrainer()は、try_import_tfを使うと、なぜかTensorflowのeager modeのエラーになる。
    trainer = ppo.PPOTrainer(config=config)
    trainer.restore(checkpoint_path=model_path)
    # trainer.get_policy().model.base_model.summary()
    # restore_model_name = results_path + 'base_model.png'
    # plot_model(trainer.get_policy().model.base_model, to_file=restore_model_name, show_shapes=True)

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
        if MAKE_ANIMATION:
            fig = plt.figure()
            ims = []
        print(f'\nEvaluation {i}:')
        obs = eval_env.reset()
        if MAKE_ANIMATION:
            im = get_engage_image_1(eval_env)
            ims.append([im])
        done = False

        while not done:
            action_dict = {}
            for j in range(eval_env.num_red):
                if eval_env.red.alive[j]:
                    action_dict['red_' + str(j)] = trainer.compute_action(obs['red_' + str(j)])

            obs, rewards, dones, infos = eval_env.step(action_dict)

            if MAKE_ANIMATION:
                im = get_engage_image_1(eval_env)
                ims.append([im])

            done = dones["__all__"]
            # print(f'step: {eval_env.steps}, done:{done}, rewards:{rewards}, '
            #      f'action:{action_dict}, obs:{obs}')

            eval_env.render(mode=MODE)

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
            anim = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=3000)
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

    setting = dict()
    setting['trial_id'] = TRIAL
    setting['model_path'] = model_path
    setting['test_id'] = TEST_ID
    setting['number of tests'] = NUM_TEST

    setting['red'] = {}
    setting['red']['num_red_max'] = NUM_RED_MAX
    setting['red']['num_red_min'] = NUM_RED_MIN
    setting['red']['total force'] = RED_TOTAL_FORCE
    setting['red']['max force'] = RED_MAX_FORCE
    setting['red']['min force'] = RED_MIN_FORCE
    setting['red']['max efficiency'] = RED_MAX_EFFICIENCY
    setting['red']['min efficiency'] = RED_MIN_EFFICIENCY

    setting['blue'] = {}
    setting['blue']['num_blue_max'] = NUM_BLUE_MAX
    setting['blue']['num_blue_min'] = NUM_BLUE_MIN
    setting['blue']['total force'] = BLUE_TOTAL_FORCE
    setting['blue']['max force'] = BLUE_MAX_FORCE
    setting['blue']['min force'] = BLUE_MIN_FORCE
    setting['blue']['max efficiency'] = BLUE_MAX_EFFICIENCY
    setting['blue']['min efficiency'] = BLUE_MIN_EFFICIENCY
    
    results = dict()
    results['step_length'] = {}
    results['step_length']['mean'] = np.mean(steps)
    results['step_length']['std'] = np.std(steps)
    
    if len(success_steps) > 0:
        results['success step length'] = {}
        results['success step length']['mean'] = np.mean(success_steps)
        results['success step length']['std'] = np.std(success_steps)
    if len(fail_steps) > 0:
        results['fail step length'] = {}
        results['fail step length']['mean'] = np.mean(fail_steps)
        results['fail step length']['std'] = np.std(fail_steps)
    if len(draw_steps) > 0:
        results['draw step length'] = {}
        results['draw step length']['mean'] = np.mean(draw_steps)
        results['draw step length']['std'] = np.std(draw_steps)

    results['success'] = success
    results['fail'] = fail
    results['draw'] = draw

    results['final alive red'] = {}
    results['final alive red']['mean'] = np.mean(alive_red)
    results['final alive red']['std'] = np.std(alive_red)

    results['final alive blue'] = {}
    results['final alive blue']['mean'] = np.mean(alive_blue)
    results['final alive blue']['std'] = np.std(alive_blue)

    results['final red force'] = {}
    results['final red force']['mean'] = np.mean(red_force)
    results['final red force']['std'] = np.std(red_force)

    results['final blue force'] = {}
    results['final blue force']['mean'] = np.mean(blue_force)
    results['final blue force']['std'] = np.std(blue_force)

    total_dict = dict()
    total_dict['settings'] = setting
    total_dict['results'] = results
    
    with open(test_results_file, mode='wt', encoding='utf8') as file:
        json.dump(total_dict, file, ensure_ascii=False, indent=2)

    ray.shutdown()


if __name__ == '__main__':
    # framework = "tf2"
    # as_test = True
    main()
