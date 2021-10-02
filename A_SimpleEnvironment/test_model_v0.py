import gym
import numpy as np
from battle_field_strategy_2D_v0 import BattleFieldStrategy
from modules.models import DenseNetModelLargeShare


def main():
    # 環境の生成
    env = BattleFieldStrategy()
    observations = env.reset()

    my_model = DenseNetModelLargeShare(env.observation_space,
                                       env.action_space,
                                       env.action_space.n,
                                       {}, 'my_model')

    while True:
        action_dict = {}
        for i in range(env.num_red):
            action_dict['red_' + str(i)] = env.action_space.sample()

        observations, rewards, dones, infos = env.step(action_dict)
        # print(observations)
        print(f'env.steps: {env.steps}')
        np.set_printoptions(precision=1)
        print(f'red_force: {env.red.force}')
        np.set_printoptions(precision=1)
        print(f'blue_force: {env.blue.force}')
        print(f'dones: {dones}')
        np.set_printoptions(precision=3)
        print(f'observations:{observations}')
        np.set_printoptions(precision=3)
        print(f'rewards: {rewards}')
        print(f'infos: {infos}')

        env.render(mode='human')

        # エピソードの終了処理
        if dones["__all__"]:
            print(f'all done at {env.steps}')
            break


if __name__ == '__main__':
    for _ in range(1):
        main()
