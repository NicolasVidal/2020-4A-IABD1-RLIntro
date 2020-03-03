import numpy as np

S = np.arange(16)
A = np.arange(4)  # 0: Left, 1: Right, 2: Up, 3: Down
P = np.zeros((S.shape[0], A.shape[0], S.shape[0], 2))

P[S[S >= 4], 2, S[S < 12], 0] = 1
P[S[S < 12], 3, S[S >= 4], 0] = 1
P[S[(S + 1) % 4 != 0], 1, S[S % 4 != 0], 0] = 1
P[S[S % 4 != 0], 0, S[(S + 1) % 4 != 0], 0] = 1

P[S[S % 4 == 0], 0, S[S % 4 == 0], 0] = 1
P[S[(S + 1) % 4 == 0], 1, S[(S + 1) % 4 == 0], 0] = 1
P[S[S < 4], 2, S[S < 4], 0] = 1
P[S[S >= 12], 3, S[S >= 12], 0] = 1

P[:, :, 3, 1] = -3
P[:, :, 15, 1] = 1

T = np.array([3, 15])

P[T, :, :, 0] = 0
