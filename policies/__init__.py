import numpy as np


def tabular_uniform_random_policy(space_size: int, action_size: int):
    return np.ones((space_size, action_size)) / action_size