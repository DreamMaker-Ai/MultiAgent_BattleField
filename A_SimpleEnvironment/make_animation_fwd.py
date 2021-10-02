"""
For making animation
    * Need specify layer number
    交戦マップ - 残存戦力
    交戦マップ - 1x1 Convolution output feature map
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
from modules.models import DenseNetModelLarge  # Need to specify proper model
from modules.animation_utils import feature_map_model, get_feature_map_image, make_feature_map_anim
from modules.animation_utils import get_engage_image, make_engagement_anim
from modules.animation_utils import get_force_image, make_forces_anim
from make_engage_array_movie import m_space_hcombine
from tensorflow.keras.utils import plot_model
import json
import copy

# tf1, tf, tfv = try_import_tf()

import tensorflow as tf

model_path = PROJECT + '/checkpoints/' + TRIAL + '/checkpoint_000901/checkpoint-901'


def main():
    if not MAKE_ANIMATION:
        print('To make animation, specify MAKE_ANIMATION=True in initial_settings.py')

    ray.init(ignore_reinit_error=True, log_to_driver=False)

    ModelCatalog.register_custom_model('my_model', DenseNetModelLarge)

    config = {"env": BattleFieldStrategy,
              "model": {"custom_model": "my_model"}
              # "framework": framework
              }  # use tensorflow 2

    # PPOTrainer()は、try_import_tfを使うと、なぜかTensorflowのeager modeのエラーになる。
    trainer = ppo.PPOTrainer(config=config)
    trainer.restore(checkpoint_path=model_path)

    # 特徴量を抽出するレイヤを定義
    ixs = [1]    # [1,3,4,7,8]等, get_layer_numbers.py で調べられる
    feature_model = feature_map_model(trainer, ixs)

    # 環境をインスタンス化
    eval_env = BattleFieldStrategy({})

    # テスト評価用
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
        # 格納ディレクトリの用意
        results_path = PROJECT + '/results/' + TRIAL + '/animation/test_' + str(TEST_ID) + '/'
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        test_results_file = results_path + 'test_' + str(TEST_ID) + '.json'
        results_path = results_path + 'episode_' + str(i) + '/'
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # 初期化
        if MAKE_ANIMATION:
            ims = []
            maps = []
            r_forces = []
            b_forces = []
            pos_xy = []
        print(f'\nEvaluation {i}:')
        obs = eval_env.reset()

        # 初期マップ生成
        if MAKE_ANIMATION:
            pos_xy.append(copy.copy(eval_env.red.pos[0]))
            feature_map = get_feature_map_image(feature_model, obs)
            maps.append(feature_map)

            im = get_engage_image(eval_env)
            ims.append(im)

            r_force, b_force = get_force_image(eval_env)
            r_forces.append(r_force)
            b_forces.append(b_force)

        done = False
        red_0_initial_pos = copy.copy(eval_env.red.pos[0])
        red_initial_forces = copy.copy(eval_env.red.force)
        blue_initial_forces = copy.copy(eval_env.blue.force)
        red_efficiencies = copy.copy(eval_env.red.efficiency)
        blue_efficiencies = copy.copy(eval_env.blue.efficiency)
        print(f'initial position of red_0 = {red_0_initial_pos}')
        print(f'red initial forces = {red_initial_forces}')
        print(f'blue initial forces = {blue_initial_forces}\n')

        # red_0 agentが生きている限り、マップをリストに追加
        while not done:
            action_dict = {}
            for j in range(eval_env.num_red):
                if eval_env.red.alive[j]:
                    action_dict['red_' + str(j)] = trainer.compute_action(obs['red_' + str(j)])

            obs, rewards, dones, infos = eval_env.step(action_dict)

            if MAKE_ANIMATION and (eval_env.red.alive[0] > 0):
                """ Make animation while red_0 agent is alive """
                pos_xy.append(copy.copy(eval_env.red.pos[0]))
                feature_map = get_feature_map_image(feature_model, obs)
                maps.append(feature_map)

                im = get_engage_image(eval_env)
                ims.append(im)

                r_force, b_force = get_force_image(eval_env)
                r_forces.append(r_force)
                b_forces.append(b_force)

            done = dones["__all__"]
            # print(f'step: {eval_env.steps}, done:{done}, rewards:{rewards}, '
            #      f'action:{action_dict}, obs:{obs}')

            eval_env.render(mode=MODE)

        # エピソード結果の整理
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
            # マップのリストからアニメーション作成
            engagement_anim_name = make_engagement_anim(ims, results_path, i, pos_xy)
            feature_map_anim_name = make_feature_map_anim(maps, results_path, i, ixs)
            force_anim_name = make_forces_anim(r_forces, b_forces, results_path, i)

            """
                Combine two animations
                    交戦マップ - 残存戦力
                    交戦マップ - 1x1 Convolution output feature map
            """
            scale_factor = 1  # FPSにかけるスケールファクタ
            movie1 = [engagement_anim_name, True]  # True for color
            movie2 = [feature_map_anim_name, True]
            movie3 = [force_anim_name, True]

            outpath1 = results_path + 'Final_animation_2_engage_feature.mp4'
            outpath2 = results_path + 'Final_animation_1_engage_force.mp4'

            m_space_hcombine(movie1, movie2, outpath1, scale_factor)
            m_space_hcombine(movie1, movie3, outpath2, scale_factor)

    # テスト全体の結果を整理し、jsonファイルとして書き出し
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
    setting['red']['red_0_position'] = pos_xy[0].tolist()
    setting['red']['initial_forces'] = red_initial_forces.tolist()
    setting['red']['efficiencies'] = red_efficiencies.tolist()

    setting['blue'] = {}
    setting['blue']['num_blue_max'] = NUM_BLUE_MAX
    setting['blue']['num_blue_min'] = NUM_BLUE_MIN
    setting['blue']['total force'] = BLUE_TOTAL_FORCE
    setting['blue']['max force'] = BLUE_MAX_FORCE
    setting['blue']['min force'] = BLUE_MIN_FORCE
    setting['blue']['max efficiency'] = BLUE_MAX_EFFICIENCY
    setting['blue']['min efficiency'] = BLUE_MIN_EFFICIENCY
    setting['blue']['initial_forces'] = blue_initial_forces.tolist()
    setting['blue']['efficiencies'] = blue_efficiencies.tolist()

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
