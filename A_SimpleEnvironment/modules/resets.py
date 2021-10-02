import numpy as np
import math
from modules.agents import RED, BLUE


def reset_red(env, init_pos_range):
    env.red = RED()

    # reset position
    candidate_pos_1 = np.arange(0, init_pos_range, dtype=np.int16)
    candidate_pos_2 = np.arange(env.grid_size - init_pos_range, env.grid_size, dtype=np.int16)
    candidate_pos = np.concatenate([candidate_pos_1, candidate_pos_2])
    pos_x = np.random.choice(candidate_pos, env.num_red)

    candidate_pos_3 = np.arange(0, env.grid_size, dtype=np.int16)
    pos_y = np.random.choice(candidate_pos_3, env.num_red)

    env.red.pos = np.vstack([pos_x, pos_y]).T

    min_red_force = env.red_min_force
    init_force_ratio = np.random.rand(env.num_red)
    init_force_ratio = init_force_ratio / np.sum(init_force_ratio)
    env.red.force = \
        min_red_force + init_force_ratio * (env.red_total_force - env.num_red * min_red_force)
    if np.round(np.sum(env.red.force)) != np.round(env.red_total_force):
        raise Exception(f'red force error: {env.red.force}, vs {env.red_total_force}')

    # reset efficiency
    red_efficiency = (env.red_max_efficiency + env.red_min_efficiency) / 2
    env.red.efficiency = np.ones(env.num_red) * red_efficiency

    # reset alive
    env.red.alive = np.array([True] * env.num_red)


def reset_blue(env, init_pos_range):
    env.blue = BLUE()

    # reset position
    mid_pos = math.floor((env.grid_size - init_pos_range) / 2)
    candidate_pos = np.arange(mid_pos, mid_pos + init_pos_range, dtype=np.int16)
    pos_x = np.random.choice(candidate_pos, env.num_blue)

    candidate_pos_3 = np.arange(0, env.grid_size, dtype=np.int16)
    pos_y = np.random.choice(candidate_pos_3, env.num_blue)

    env.blue.pos = np.vstack([pos_x, pos_y]).T

    min_blue_force = env.blue_min_force
    init_force_ratio = np.random.rand(env.num_blue)
    init_force_ratio = init_force_ratio / np.sum(init_force_ratio)
    env.blue.force = \
        min_blue_force + init_force_ratio * (env.blue_total_force - env.num_blue * min_blue_force)
    if np.round(np.sum(env.blue.force)) != np.round(env.blue_total_force):
        raise Exception(f'blue force error: {env.blue.force}, vs {env.blue_total_force}')

    # reset efficiency
    blue_efficiency = (env.blue_max_efficiency + env.blue_min_efficiency) / 2
    env.blue.efficiency = np.ones(env.num_blue) * blue_efficiency

    # reset alive
    env.blue.alive = np.array([True] * env.num_blue)
