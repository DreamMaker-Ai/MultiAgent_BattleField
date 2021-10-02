import json
import os
from settings.initial_settings import *


def save_conditions(dir):
    file_name = os.path.join(dir, TRIAL + '.json')
    training = dict()
    test = dict()
    env_setting = dict()
    setting = dict()

    training["name"] = PROJECT
    training["algorithm"] = ALGORITHM
    training["trial"] = TRIAL
    training["eval_freq"] = EVAL_FREQ
    training["num_eval"] = NUM_EVAL

    training["num_workers"] = NUM_WORKERS
    training["num_gpus"] = NUM_GPUS
    training["num_cpus_per_worker"] = NUM_CPUS_PER_WORKER
    training["num_sgd_iter"] = NUM_SGD_ITER
    training["learning_rate"] = LEARNING_RATE

    test["num_test"] = NUM_TEST

    env_setting["max_steps"] = MAX_STEPS
    env_setting["grid_size"] = GRID_SIZE
    env_setting["channel"] = CHANNEL
    env_setting["alive_criteria"] = ALIVE_CRITERIA
    env_setting["dt"] = DT
    env_setting["init_pos_range_ratio"] = INIT_POS_RANGE_RATIO

    env_setting["num_red_max"] = NUM_RED_MAX
    env_setting["num_red_min"] = NUM_RED_MIN
    env_setting["num_blue_max"] = NUM_BLUE_MAX
    env_setting["num_blue_min"] = NUM_BLUE_MIN
    env_setting["red_total_force"] = RED_TOTAL_FORCE
    env_setting["blue_total_rofce"] = BLUE_TOTAL_FORCE
    env_setting["red_max_force"] = RED_MAX_FORCE
    env_setting["blue_max_force"] = BLUE_MAX_FORCE
    env_setting["red_min_force"] = RED_MIN_FORCE
    env_setting["blue_min_force"] = BLUE_MIN_FORCE
    env_setting["red_max_efficiency"] = RED_MAX_EFFICIENCY
    env_setting["blue_max_efficiency"] = BLUE_MAX_EFFICIENCY
    env_setting["red_min_efficiency"] = RED_MIN_EFFICIENCY
    env_setting["blue_min_efficiency"] = BLUE_MIN_EFFICIENCY

    setting['training'] = training
    setting['test'] = test
    setting['env_setting'] = env_setting

    with open(file_name, mode='wt', encoding='utf-8') as file:
        json.dump(setting, file, ensure_ascii=False, indent=2)
