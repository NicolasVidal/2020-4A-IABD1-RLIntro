import numpy as np

from algorithms import iterative_policy_evaluation
from grid_world import S, A, P, T
from policies import tabular_uniform_random_policy

if __name__ == "__main__":
    Pi = tabular_uniform_random_policy(S.shape[0], A.shape[0])
    V = iterative_policy_evaluation(S, A, P, T, Pi)
    print(V)
