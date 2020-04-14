from algorithms import monte_carlo_with_exploring_starts_control
from grid_world import S, A, is_terminal, step

if __name__ == "__main__":
    Q, Pi = monte_carlo_with_exploring_starts_control(len(S), len(A), is_terminal, step,
                                                      episodes_count=10000, max_steps_per_episode=100)
    print(Q)
    print(Pi)
