from algorithms import first_visit_monte_carlo_prediction
from grid_world import S, A, is_terminal, reset, step
from policies import tabular_uniform_random_policy

if __name__ == "__main__":
    import time

    start_time = time.time()
    Pi = tabular_uniform_random_policy(S.shape[0], A.shape[0])
    V = first_visit_monte_carlo_prediction(Pi, is_terminal, reset, step, max_steps_per_episode=10, episodes_count=10000)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(V)
