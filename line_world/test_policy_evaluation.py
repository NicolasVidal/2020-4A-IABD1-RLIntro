import numpy as np

from algorithms import iterative_policy_evaluation
from line_world import S, A, P, T
from policies import tabular_uniform_random_policy

if __name__ == "__main__":
    import time

    start_time = time.time()
    Pi = tabular_uniform_random_policy(S.shape[0], A.shape[0])
    V = iterative_policy_evaluation(S, A, P, T, Pi)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(V)

    # Pi = np.zeros((S.shape[0], A.shape[0]))
    # Pi[:, 1] = 1.0
    # V = iterative_policy_evaluation(S, A, P, T, Pi)
    # print(V)
    #
    # Pi = np.zeros((S.shape[0], A.shape[0]))
    # Pi[:, 0] = 1.0
    # V = iterative_policy_evaluation(S, A, P, T, Pi)
    # print(V)
