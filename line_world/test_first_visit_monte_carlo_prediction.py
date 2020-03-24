from algorithms import first_visit_monte_carlo_prediction
from line_world import S, A, is_terminal, reset, step
import numpy as np
from policies import tabular_uniform_random_policy

if __name__ == "__main__":
    import time

    # start_time = time.time()
    # Pi = tabular_uniform_random_policy(S.shape[0], A.shape[0])
    # V = first_visit_monte_carlo_prediction(Pi, is_terminal, reset, step)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print(V)

    Pi = np.zeros((S.shape[0], A.shape[0]))
    Pi[:, 1] = 1.0
    V = first_visit_monte_carlo_prediction(Pi, is_terminal, reset, step)
    print(V)
    #
    # Pi = np.zeros((S.shape[0], A.shape[0]))
    # Pi[:, 0] = 1.0
    # V = iterative_policy_evaluation(S, A, P, T, Pi)
    # print(V)
