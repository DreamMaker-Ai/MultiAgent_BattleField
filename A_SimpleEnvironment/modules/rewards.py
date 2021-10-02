import numpy as np


def get_reward(env, red_team_force, previous_red_team_force, blue_team_force, previous_blue_team_force,
               red_team_efficiency, blue_team_efficiency):
    # red_reward_1 = get_reward_1(env, red_team_force, previous_red_team_force,
    #                           blue_team_force, previous_blue_team_force)

    # red_reward_2 = get_reward_2(env, red_team_force, previous_red_team_force,
    #                            blue_team_force, previous_blue_team_force)

    # red_reward_3 = get_reward_3(env, red_team_force, previous_red_team_force,
    #                            blue_team_force, previous_blue_team_force)

    # red_reward_4 = get_reward_4(env, red_team_force, previous_red_team_force,
    #                            blue_team_force, previous_blue_team_force)

    # red_reward_5 = get_reward_5(env, red_team_force, previous_red_team_force,
    #                            blue_team_force, previous_blue_team_force)

    # red_reward_6 = get_reward_6(env, red_team_force, previous_red_team_force,
    #                            blue_team_force, previous_blue_team_force,
    #                            red_team_efficiency, blue_team_efficiency)  # Good

    # red_reward_7 = get_reward_7(env, red_team_force, previous_red_team_force,
    #                            blue_team_force, previous_blue_team_force,
    #                            red_team_efficiency, blue_team_efficiency)

    # red_reward_7A = get_reward_7A(env, red_team_force, previous_red_team_force,
    #                              blue_team_force, previous_blue_team_force,
    #                              red_team_efficiency, blue_team_efficiency)

    # red_reward_8 = get_reward_8(env, red_team_force, previous_red_team_force,
    #                            blue_team_force, previous_blue_team_force,
    #                            red_team_efficiency, blue_team_efficiency)

    # red_reward_8A = get_reward_8A(env, red_team_force, previous_red_team_force,
    #                              blue_team_force, previous_blue_team_force,
    #                              red_team_efficiency, blue_team_efficiency)

    #red_reward_8B = get_reward_8B(env, red_team_force, previous_red_team_force,
    #                              blue_team_force, previous_blue_team_force,
    #                              red_team_efficiency, blue_team_efficiency)

    red_reward_8C = get_reward_8C(env, red_team_force, previous_red_team_force,
                                  blue_team_force, previous_blue_team_force,
                                  red_team_efficiency, blue_team_efficiency)

    # red_reward_9 = get_reward_9(env, red_team_force, previous_red_team_force,
    #                            blue_team_force, previous_blue_team_force,
    #                            red_team_efficiency, blue_team_efficiency)

    # red_reward_10 = get_reward_10(env, red_team_force, previous_red_team_force,
    #                              blue_team_force, previous_blue_team_force,
    #                              red_team_efficiency, blue_team_efficiency)  # Better

    # red_reward_100 = get_reward_8A(env, red_team_force, previous_red_team_force,
    #                               blue_team_force, previous_blue_team_force,
    #                               red_team_efficiency, blue_team_efficiency)

    return red_reward_8C


def get_reward_1(env, red_team_force, previous_red_team_force, blue_team_force, previous_blue_team_force):
    # 1. local step reward proportional to effect (Not stable)
    # 学習が安定しない。
    # rewards ∝ R/R0 x (1-B/B0)
    alpha = 1 / env.dt * .1
    red_reward = alpha * red_team_force / previous_red_team_force \
                 * (1 - np.sum(blue_team_force) / np.sum(previous_blue_team_force))
    return red_reward


def get_reward_2(env, red_team_force, previous_red_team_force, blue_team_force, previous_blue_team_force):
    # 2. local step reward proportional to clipped effect
    #    coef = 0.1: makes longer episode mean length
    # １を安定化させるために、red_team_forceが小さくなった時に、報酬の上限をクリップ
    # rewards ∝ clip(R/R0 x (1-B/B0))
    coef = 0.5
    alpha = 1 / env.dt * .1
    red_reward = alpha * red_team_force / previous_red_team_force \
                 * (1 - np.sum(blue_team_force) / np.sum(previous_blue_team_force))
    red_reward = np.clip(red_reward, 0, alpha * coef)
    return red_reward


def get_reward_3(env, red_team_force, previous_red_team_force, blue_team_force, previous_blue_team_force):
    # 3. local step reward proportional to force size (Not good!)
    # 集結したチーム力に比例した報酬
    # rewards ∝ R
    red_reward = previous_red_team_force / env.dt * 5
    return red_reward


def get_reward_4(env, red_team_force, previous_red_team_force, blue_team_force, previous_blue_team_force):
    # 4. local step reward just for engagement (Not good!)
    # 集結したことに対し、集結力とは無関係に報酬
    # rewards ∝ 1
    red_reward = np.ones_like(red_team_force) * env.dt
    return red_reward


def get_reward_5(env, red_team_force, previous_red_team_force, blue_team_force, previous_blue_team_force):
    # 5. local step reward proportional to attrition
    # 敵と自分の消耗率を考慮した報酬
    # 理屈としては、これが良い気がするが、戦いを避ける傾向がある。トレーニング不足か？
    # rewards ∝ (|ΔB|-|ΔR|)xR
    positive_reward = np.sum(previous_blue_team_force - blue_team_force)
    negative_reward = np.sum(previous_red_team_force - red_team_force)
    red_reward = (positive_reward - negative_reward) * previous_red_team_force / env.red_max_force / env.dt
    return red_reward


def get_reward_6(env, red_team_force, previous_red_team_force, blue_team_force, previous_blue_team_force,
                 red_team_efficiency, blue_team_efficiency):
    # 6. local step reward proportional to consolidation
    # 3 の改良。戦力を集中させるが、負けると判っていても報酬が出るので戦い続ける。
    # rewards ∝ |B|xR
    blue_size = np.sum(previous_blue_team_force * blue_team_efficiency)
    red_size = previous_red_team_force * red_team_efficiency
    red_reward = (blue_size / env.blue_max_force) * (red_size / env.red_max_force) / env.dt * 5
    return red_reward


def get_reward_7(env, red_team_force, previous_red_team_force, blue_team_force, previous_blue_team_force,
                 red_team_efficiency, blue_team_efficiency):
    # 7. local step reward proportional to consolidation
    # 6 の改良。戦力サイズが大きい（つまり勝てる）時だけ、彼我戦力に比例した報酬を与える。
    # rewards ∝ |B|x|R|xR if |R| >= |B|
    blue_size = np.sum(previous_blue_team_force * blue_team_efficiency)
    red_size = np.sum(previous_red_team_force * red_team_efficiency)

    if red_size >= blue_size:
        red_reward = (blue_size / env.blue_max_force) * (red_size / env.red_max_force) \
                     * (previous_red_team_force * red_team_efficiency) / env.red_max_force / env.dt * 5
        red_reward = np.maximum(red_reward, 0.1 * 0.1 / env.dt * 5)
    else:
        red_reward = np.zeros_like(red_team_force)

    return red_reward


def get_reward_7A(env, red_team_force, previous_red_team_force, blue_team_force, previous_blue_team_force,
                  red_team_efficiency, blue_team_efficiency):
    # 7 の改良。penaltyを与える。
    # rewards ∝ |B|xR if |R| >= |B|
    blue_size = np.sum(previous_blue_team_force * blue_team_efficiency)
    red_size = np.sum(previous_red_team_force * red_team_efficiency)

    if red_size >= blue_size:
        red_reward = (blue_size / env.blue_max_force) * (red_size / env.red_max_force) \
                     * (previous_red_team_force * red_team_efficiency) / env.red_max_force / env.dt * 5
        red_reward = np.maximum(red_reward, 0.1 * 0.1 / env.dt * 5)
    else:
        red_reward = - np.ones_like(red_team_force) * 0.1 * 0.1 / env.dt * 5

    return red_reward


def get_reward_8(env, red_team_force, previous_red_team_force, blue_team_force, previous_blue_team_force,
                 red_team_efficiency, blue_team_efficiency):
    # 8. local step reward proportional to consolidation
    # 6 の改良。
    # rewards ∝ |B|x|R|xR
    blue_size = np.sum(previous_blue_team_force * blue_team_efficiency)
    red_size = np.sum(previous_red_team_force * red_team_efficiency)
    red_reward = (blue_size / env.blue_max_force) * (red_size / env.red_max_force) \
                 * (previous_red_team_force * red_team_efficiency) / env.red_max_force / env.dt * 5
    # red_reward = (blue_size / env.blue_max_force) * (red_size / env.red_max_force) \
    #             * previous_red_team_force * red_team_efficiency
    red_reward = np.maximum(red_reward, 0.1 * 0.1 / env.dt * 5)
    return red_reward


def get_reward_8A(env, red_team_force, previous_red_team_force, blue_team_force, previous_blue_team_force,
                  red_team_efficiency, blue_team_efficiency):
    # 8. local step reward proportional to consolidation
    # 6 の改良。
    # rewards ∝ |B|x|R|xR
    blue_size = np.sum(previous_blue_team_force * blue_team_efficiency)
    red_size = np.sum(previous_red_team_force * red_team_efficiency)

    # red_reward = (blue_size / env.blue_max_force) * (red_size / env.red_max_force) \
    #             * (previous_red_team_force * red_team_efficiency) / env.red_max_force / env.dt * 5
    red_reward = (blue_size / env.blue_max_force) * (red_size / env.red_max_force) \
                 * (previous_red_team_force * red_team_efficiency) / env.red_max_force * 100

    # red_reward = np.maximum(red_reward, 0.1 / env.dt * 0.5)
    red_reward = np.maximum(red_reward, 0.5)
    return red_reward


def get_reward_8B(env, red_team_force, previous_red_team_force, blue_team_force, previous_blue_team_force,
                  red_team_efficiency, blue_team_efficiency):
    # 8Aの報酬の３倍
    blue_size = np.sum(previous_blue_team_force * blue_team_efficiency)
    red_size = np.sum(previous_red_team_force * red_team_efficiency)

    # red_reward = (blue_size / env.blue_max_force) * (red_size / env.red_max_force) \
    #             * (previous_red_team_force * red_team_efficiency) / env.red_max_force / env.dt * 5
    red_reward = (blue_size / env.blue_max_force) * (red_size / env.red_max_force) \
                 * (previous_red_team_force * red_team_efficiency) / env.red_max_force * 100 * 3

    # red_reward = np.maximum(red_reward, 0.1 / env.dt * 0.5)
    red_reward = np.maximum(red_reward, 0.5)
    return red_reward


def get_reward_8C(env, red_team_force, previous_red_team_force, blue_team_force, previous_blue_team_force,
                  red_team_efficiency, blue_team_efficiency):
    # 8Bの足切り 0.5 → 1.1
    blue_size = np.sum(previous_blue_team_force * blue_team_efficiency)
    red_size = np.sum(previous_red_team_force * red_team_efficiency)

    # red_reward = (blue_size / env.blue_max_force) * (red_size / env.red_max_force) \
    #             * (previous_red_team_force * red_team_efficiency) / env.red_max_force / env.dt * 5
    red_reward = (blue_size / env.blue_max_force) * (red_size / env.red_max_force) \
                 * (previous_red_team_force * red_team_efficiency) / env.red_max_force * 100 * 3

    # red_reward = np.maximum(red_reward, 0.1 / env.dt * 0.5)
    red_reward = np.maximum(red_reward, 1.1)
    return red_reward


def get_reward_9(env, red_team_force, previous_red_team_force, blue_team_force, previous_blue_team_force,
                 red_team_efficiency, blue_team_efficiency):
    # 6. local step reward proportional to consolidation
    # 6 の改良。Rに比例させずに、同じ報酬を配分。
    # rewards ∝ |B|x|R| for each member of R
    blue_size = np.sum(previous_blue_team_force * blue_team_efficiency)
    red_size = np.sum(previous_red_team_force * red_team_efficiency)
    reward = (blue_size / env.blue_max_force) * (red_size / env.red_max_force) / env.dt * 5
    red_reward = reward * np.ones_like(red_team_force)
    red_reward = np.maximum(red_reward, 0.1 * 0.1 / env.dt * 5)
    return red_reward


def get_reward_10(env, red_team_force, previous_red_team_force, blue_team_force, previous_blue_team_force,
                  red_team_efficiency, blue_team_efficiency):
    # 6. local step reward proportional to consolidation
    # 3 の改良。最小の報酬を red, blue initial total force の10%で足切り（サイスが小さくなった時に、
    # 彷徨い始めないようにする措置）
    # rewards ∝ max(|B|xR, 0.1)
    blue_size = np.sum(previous_blue_team_force * blue_team_efficiency)
    red_size = previous_red_team_force * red_team_efficiency
    red_reward = (blue_size / env.blue_max_force) * (red_size / env.red_max_force) / env.dt * 5
    red_reward = np.maximum(red_reward, 0.1 * 0.1 / env.dt * 5)
    return red_reward


def get_reward_100(env, red_team_force, previous_red_team_force, blue_team_force, previous_blue_team_force,
                   red_team_efficiency, blue_team_efficiency):
    # rewards ∝ sign(|R|-|B|)
    blue_size = np.sum(blue_team_force * blue_team_efficiency)
    red_size = np.sum(red_team_force * red_team_efficiency)
    red_reward = np.sign(red_size - blue_size) * np.ones_like(red_team_force)
    return red_reward
