from algorithms import tabular_q_learning_control
from grid_world import S, A, is_terminal, step, reset

if __name__ == "__main__":
    Q, Pi = tabular_q_learning_control(len(S), len(A),
                                       reset,
                                       is_terminal, step,
                                       epsilon=0.75,
                                       max_steps_per_episode=100)
    print(Q)
    print(Pi)
