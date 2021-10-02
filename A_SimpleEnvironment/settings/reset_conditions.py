import numpy as np

def reset_conditions(num_red_max, num_red_min, num_blue_max, num_blue_min):
    # Reset number of agents
    num_red_selection = np.arange(num_red_min, num_red_max + 1)
    num_blue_selection = np.arange(num_blue_min, num_blue_max + 1)
    num_red = np.random.choice(num_red_selection, 1)[0]
    num_blue = np.random.choice(num_blue_selection, 1)[0]
    return num_red, num_blue