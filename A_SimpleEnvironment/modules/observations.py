import numpy as np


def get_observation(env):
    # observations = get_observation_1(env)
    observations = get_observation_2(env)

    return observations


"""
def get_observation_1(env):
    observation = {}

    for i in range(env.num_red):
        if env.red.alive[i]:
            obs = np.array([env.red.pos[i] / (env.grid_size - 1), env.red.force[i] / env.max_force])

            teammate_id = [j for j in range(env.num_red) if j != i]
            teammate_pos = env.red.pos[teammate_id] / (env.grid_size - 1)
            teammate_force = env.red.force[teammate_id] / env.max_force
            teammate = np.vstack([teammate_pos, teammate_force]).T.flatten()
            obs = np.append(obs, teammate)

            enemy = np.vstack([env.blue.pos / (env.grid_size - 1), env.blue.force / env.max_force])
            obs = np.append(obs, enemy)

            observation['red_' + str(i)] = obs.astype(np.float32)
    return observation
"""


def get_observation_2(env):
    observation = {}

    for i in range(env.num_red):
        if env.red.alive[i]:
            my_matrix = np.zeros((env.grid_size, env.grid_size))
            teammate_matrix = np.zeros((env.grid_size, env.grid_size))
            adversarial_matrix = np.zeros((env.grid_size, env.grid_size))

            # my position & force map
            my_matrix[env.red.pos[i][0], env.red.pos[i][1]] += env.red.force[i]

            # teammate position & force map
            teammate_id = [j for j in range(env.num_red) if j != i]
            # teammate_pos = env.red.pos[teammate_id]
            # teammate_force = env.red.force[teammate_id]
            for j in teammate_id:
                teammate_matrix[env.red.pos[j][0], env.red.pos[j][1]] += env.red.force[j]

            # adversarial position & force map
            for j in range(env.num_blue):
                adversarial_matrix[env.blue.pos[j][0], env.blue.pos[j][1]] += env.blue.force[j]

            # stack the maps
            my_matrix = np.expand_dims(my_matrix, axis=2)
            teammate_matrix = np.expand_dims(teammate_matrix, axis=2)
            adversarial_matrix = np.expand_dims(adversarial_matrix, axis=2)

            # normalize the maps
            my_matrix = my_matrix / env.max_force
            teammate_matrix = teammate_matrix / env.max_force
            adversarial_matrix = adversarial_matrix / env.max_force

            obs = np.concatenate([my_matrix, teammate_matrix, adversarial_matrix], axis=-1)

            observation['red_' + str(i)] = obs.astype(np.float32)
    return observation


