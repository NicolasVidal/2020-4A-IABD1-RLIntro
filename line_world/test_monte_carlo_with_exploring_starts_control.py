from algorithms import monte_carlo_with_exploring_starts_control
from line_world import S, A, is_terminal, step

if __name__ == "__main__":
    Q, Pi = monte_carlo_with_exploring_starts_control(len(S), len(A), is_terminal, step)
    print(Q)
    print(Pi)
