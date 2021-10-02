"""
Multi agents are learning 2D battle field strategy.
Only red agents are learnable, and blue agents are stationary.
Learnable blue agents will be in the future work.
"""
import gym
from gym import spaces
import numpy as np
import math
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from settings.initial_settings import *  # Import settings  # For training
# from settings.test_settings import *  # Import settings  # For testing
from settings.reset_conditions import reset_conditions
from modules.resets import reset_red, reset_blue
from modules.observations import get_observation
from modules.rewards import get_reward


class BattleFieldStrategy(MultiAgentEnv):
    def __init__(self, config={}):
        super(BattleFieldStrategy, self).__init__()
        self.max_steps = MAX_STEPS
        self.grid_size = GRID_SIZE
        self.channel = CHANNEL
        self.alive_criteria = ALIVE_CRITERIA
        self.dt = DT
        self.steps = None

        self.num_red_max = NUM_RED_MAX
        self.num_blue_max = NUM_BLUE_MAX
        self.num_red_min = NUM_RED_MIN
        self.num_blue_min = NUM_BLUE_MIN
        self.num_red = None
        self.num_blue = None

        self.red_total_force = RED_TOTAL_FORCE
        self.blue_total_force = BLUE_TOTAL_FORCE
        self.red_max_force = RED_MAX_FORCE
        self.blue_max_force = BLUE_MAX_FORCE
        self.red_min_force = RED_MIN_FORCE
        self.blue_min_force = BLUE_MIN_FORCE
        self.max_force = max([self.red_max_force, self.blue_max_force])
        self.min_force = max([self.red_min_force, self.blue_min_force])

        self.red_max_efficiency = RED_MAX_EFFICIENCY
        self.red_min_efficiency = RED_MIN_EFFICIENCY
        self.blue_max_efficiency = BLUE_MAX_EFFICIENCY
        self.blue_min_efficiency = BLUE_MIN_EFFICIENCY

        self.action_space = spaces.Discrete(5)
        self.observation_space = \
            spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, self.channel))

        obs = self.reset()

    def reset(self):
        self.steps = 0

        self.num_red, self.num_blue = \
            reset_conditions(self.num_red_max, self.num_red_min, self.num_blue_max, self.num_blue_min)

        init_pos_range = math.floor(self.grid_size * INIT_POS_RANGE_RATIO)
        reset_red(self, init_pos_range)
        reset_blue(self, init_pos_range)
        obs = get_observation(self)

        return obs

    def step(self, action_dict):
        obs, rewards, dones, infos = {}, {}, {}, {}

        # state transition 1: move of red
        for i in range(self.num_red):
            if self.red.alive[i]:
                action = action_dict['red_' + str(i)]
                if action == 1:
                    self.red.pos[i][0] -= 1
                elif action == 2:
                    self.red.pos[i][0] += 1
                elif action == 3:
                    self.red.pos[i][1] -= 1
                elif action == 4:
                    self.red.pos[i][1] += 1

                self.red.pos[i] = np.clip(self.red.pos[i], 0, self.grid_size - 1)

        # state transition 2: move of blue
        pass

        # state transition 3 & local step reward: engage (Lanchester's combat model)
        # red_team_reward = np.zeros(self.num_red)  # No negative step reward
        # red_team_reward = - np.ones(self.num_red) * self.dt * 1  # negative step reward
        red_team_reward = - np.ones(self.num_red) * 0.1  # negative step reward

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                red_team_id = [m for m in range(self.num_red)
                               if (np.all(self.red.pos[m] == (i, j)) and self.red.alive[m])]
                blue_team_id = [m for m in range(self.num_blue)
                                if (np.all(self.blue.pos[m] == (i, j)) and self.blue.alive[m])]

                if (len(red_team_id) > 0) and (len(blue_team_id) > 0):
                    red_team_force = self.red.force[red_team_id]
                    red_team_efficiency = self.red.efficiency[red_team_id]

                    blue_team_force = self.blue.force[blue_team_id]
                    blue_team_efficiency = self.blue.efficiency[blue_team_id]

                    red_force = np.sum(red_team_force)
                    blue_force = np.sum(blue_team_force)
                    red_force_efficiency = np.sum(red_team_force * red_team_efficiency)
                    blue_force_efficiency = np.sum(blue_team_force * blue_team_efficiency)

                    dr = red_team_force / red_force * blue_force_efficiency * self.dt
                    db = blue_team_force / blue_force * red_force_efficiency * self.dt

                    previous_red_team_force = red_team_force.copy()
                    previous_blue_team_force = blue_team_force.copy()

                    red_team_force -= dr
                    blue_team_force -= db

                    red_team_force = np.clip(red_team_force, 0, None)
                    blue_team_force = np.clip(blue_team_force, 0, None)

                    red_reward = get_reward(self, red_team_force, previous_red_team_force,
                                            blue_team_force, previous_blue_team_force,
                                            red_team_efficiency, blue_team_efficiency)

                    # update force and reward
                    for m, idx in enumerate(red_team_id):
                        self.red.force[idx] = red_team_force[m]
                        red_team_reward[idx] += red_reward[m]

                    for m, idx in enumerate(blue_team_id):
                        self.blue.force[idx] = blue_team_force[m]

        obs = get_observation(self)

        alive_criteria = self.alive_criteria
        for i in range(self.num_red):
            if self.red.alive[i]:
                rewards['red_' + str(i)] = red_team_reward[i]
                if self.red.force[i] <= alive_criteria:
                    dones['red_' + str(i)] = True
                    self.red.alive[i] = False
                else:
                    dones['red_' + str(i)] = False

        for i in range(self.num_blue):
            if (self.blue.force[i] <= alive_criteria) and (self.blue.alive[i]):
                self.blue.alive[i] = False

        # is done all?
        dones['__all__'] = False

        criteria_1 = np.all(np.logical_not(self.red.alive))
        criteria_2 = np.all(np.logical_not(self.blue.alive))
        if criteria_1 or criteria_2:
            dones['__all__'] = True

        # max step check
        if self.steps >= self.max_steps:
            dones["__all__"] = True

        # info
        for i in range(self.num_red):
            if self.red.alive[i]:
                infos['red_' + str(i)] = {}

        self.steps += 1

        return obs, rewards, dones, infos

    def render(self, mode='human', close=False):
        if mode == 'human':
            pass
            """
            print(f'steps: {self.steps}')
            for i in range(self.num_red):
                if self.red.alive[i]:
                    print('.' * self.red.pos[i], end='')
                    print('R', end='')
                    print('.' * (self.grid_size - 1 - self.red.pos[i]))

            for i in range(self.num_blue):
                if self.blue.alive[i]:
                    print('.' * self.blue.pos[i], end='')
                    print('B', end='')
                    print('.' * (self.grid_size - 1 - self.blue.pos[i]))

            print('\n')
            """
        else:
            pass


if __name__ == '__main__':
    # Simple checker of the code
    config = {}
    env = BattleFieldStrategy(config)

    env.reset()

    action_dict = {}
    action_dict['red_0'] = env.action_space.sample()
    action_dict['red_1'] = env.action_space.sample()

    observations, rewards, dones, infos = env.step(action_dict)
    # print(observations)

    env.render()
