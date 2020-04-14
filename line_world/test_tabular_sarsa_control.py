from algorithms import tabular_sarsa_control
from line_world import S, A, is_terminal, step, reset

if __name__ == "__main__":
    Q, Pi = tabular_sarsa_control(len(S), len(A),
                                  reset,
                                  is_terminal, step)
    print(Q)
    print(Pi)
