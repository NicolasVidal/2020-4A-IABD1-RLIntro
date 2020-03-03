from algorithms import value_iteration
from grid_world import S, A, P, T

if __name__ == "__main__":
    V, Pi = value_iteration(S, A, P, T)
    print(V)
    print(Pi)
