import numpy as np

S = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
A = np.array([0, 1])  # 0: Left, 1: Right
P = np.zeros((S.shape[0], A.shape[0], S.shape[0], 2))

for s in range(1, 8):
    P[s, 0, s - 1, 0] = 1.0
    P[s, 1, s + 1, 0] = 1.0

P[1, 0, 0, 1] = -1.0
P[7, 1, 8, 1] = 1.0

T = np.array([0, 8])
