import numpy as np


def iterative_policy_evaluation(
        S: np.ndarray,
        A: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        Pi: np.ndarray,
        gamma: float = 0.99,
        theta: float = 0.000001
) -> np.ndarray:
    assert 0 <= gamma <= 1
    assert theta > 0

    V = np.random.random((S.shape[0],))
    V[T] = 0.0
    while True:
        delta = 0
        for s in S:
            v_temp = V[s]
            tmp_sum = 0
            for a in A:
                for s_p in S:
                    tmp_sum += Pi[s, a] * P[s, a, s_p, 0] * (
                            P[s, a, s_p, 1] + gamma * V[s_p]
                    )
            V[s] = tmp_sum
            delta = np.maximum(delta, np.abs(tmp_sum - v_temp))
        if delta < theta:
            break
    return V
