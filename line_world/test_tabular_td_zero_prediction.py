import numpy as np

from algorithms import tabular_td_zero_prediction
from line_world import S, A, is_terminal, reset, step
from policies import tabular_uniform_random_policy

if __name__ == "__main__":
    Pi = tabular_uniform_random_policy(S.shape[0], A.shape[0])
    V = tabular_td_zero_prediction(Pi, is_terminal, reset, step)
    print(V)

    Pi = np.zeros((S.shape[0], A.shape[0]))
    Pi[:, 1] = 1.0
    V = tabular_td_zero_prediction(Pi, is_terminal, reset, step)
    print(V)
