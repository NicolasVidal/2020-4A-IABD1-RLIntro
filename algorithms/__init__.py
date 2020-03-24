from typing import Callable

import numpy as np

from policies import tabular_uniform_random_policy
from utils import step_until_the_end_of_the_episode_and_return_history


def iterative_policy_evaluation(
        S: np.ndarray,
        A: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        Pi: np.ndarray,
        gamma: float = 0.99,
        theta: float = 0.000001,
        V: np.ndarray = None
) -> np.ndarray:
    assert 0 <= gamma <= 1
    assert theta > 0

    if V is None:
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


def policy_iteration(
        S: np.ndarray,
        A: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        gamma: float = 0.99,
        theta: float = 0.000001
) -> (np.ndarray, np.ndarray):
    Pi = tabular_uniform_random_policy(S.shape[0], A.shape[0])
    V = np.random.random((S.shape[0],))
    V[T] = 0.0
    while True:
        V = iterative_policy_evaluation(S, A, P, T, Pi, gamma, theta, V)
        policy_stable = True
        for s in S:
            old_action = np.argmax(Pi[s])
            best_action = 0
            best_action_score = -9999999999999
            for a in A:
                tmp_sum = 0
                for s_p in S:
                    tmp_sum += P[s, a, s_p, 0] * (
                            P[s, a, s_p, 1] + gamma * V[s_p]
                    )
                if tmp_sum > best_action_score:
                    best_action = a
                    best_action_score = tmp_sum
            Pi[s] = 0.0
            Pi[s, best_action] = 1.0
            if best_action != old_action:
                policy_stable = False
        if policy_stable:
            break
    return V, Pi


def value_iteration(
        S: np.ndarray,
        A: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        gamma: float = 0.99,
        theta: float = 0.000001
) -> (np.ndarray, np.ndarray):
    assert 0 <= gamma <= 1
    assert theta > 0

    V = np.random.random((S.shape[0],))
    V[T] = 0.0
    while True:
        delta = 0
        for s in S:
            v_temp = V[s]
            best_score = -9999999999
            for a in A:
                tmp_sum = 0
                for s_p in S:
                    tmp_sum += P[s, a, s_p, 0] * (
                            P[s, a, s_p, 1] + gamma * V[s_p]
                    )
                if best_score < tmp_sum:
                    best_score = tmp_sum
            V[s] = best_score
            delta = np.maximum(delta, np.abs(V[s] - v_temp))
        if delta < theta:
            break

    Pi = np.zeros((S.shape[0], A.shape[0]))
    for s in S:
        best_action = 0
        best_action_score = -9999999999999
        for a in A:
            tmp_sum = 0
            for s_p in S:
                tmp_sum += P[s, a, s_p, 0] * (
                        P[s, a, s_p, 1] + gamma * V[s_p]
                )
            if tmp_sum > best_action_score:
                best_action = a
                best_action_score = tmp_sum
        Pi[s] = 0.0
        Pi[s, best_action] = 1.0
    return V, Pi


def first_visit_monte_carlo_prediction(
        pi: np.ndarray,
        is_terminal_func: Callable,
        reset_func: Callable,
        step_func: Callable,
        episodes_count: int = 100000,
        max_steps_per_episode: int = 100,
        gamma: float = 0.99,
        exploring_start: bool = False
) -> np.ndarray:
    states = np.arange(pi.shape[0])
    V = np.random.random(pi.shape[0])
    for s in states:
        if is_terminal_func(s):
            V[s] = 0
    returns = np.zeros(V.shape[0])
    returns_count = np.zeros(V.shape[0])
    for episode_id in range(episodes_count):
        s0 = np.random.choice(states) if exploring_start else reset_func()
        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_return_history(s0, pi, is_terminal_func,
                                                                                         step_func,
                                                                                         max_steps_per_episode)
        G = 0
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            if st in s_list[0:t]:
                continue
            returns[st] += G
            returns_count[st] += 1
            V[st] = returns[st] / returns_count[st]
    return V


def monte_carlo_with_exploring_starts_control(
        states_count: int,
        actions_count: int,
        is_terminal_func: Callable,
        step_func: Callable,
        episodes_count: int = 10000,
        max_steps_per_episode: int = 10,
        gamma: float = 0.99,
) -> (np.ndarray, np.ndarray):
    states = np.arange(states_count)
    actions = np.arange(actions_count)
    pi = tabular_uniform_random_policy(states_count, actions_count)
    q = np.random.random((states_count, actions_count))
    for s in states:
        if is_terminal_func(s):
            q[s, :] = 0.0
            pi[s, :] = 0.0

    returns = np.zeros((states_count, actions_count))
    returns_count = np.zeros((states_count, actions_count))
    for episode_id in range(episodes_count):
        s0 = np.random.choice(states)

        if is_terminal_func(s0):
            continue

        a0 = np.random.choice(actions)
        s1, r1, t1 = step_func(s0, a0)

        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_return_history(s1, pi, is_terminal_func,
                                                                                         step_func,
                                                                                         max_steps_per_episode)
        s_list = [s0] + s_list
        a_list = [a0] + a_list
        r_list = [r1] + r_list

        G = 0
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            at = a_list[t]

            if (st, at) in zip(s_list[0:t], a_list[0:t]):
                continue
            returns[st, at] += G
            returns_count[st, at] += 1
            q[st, at] = returns[st, at] / returns_count[st, at]
            pi[st, :] = 0.0
            pi[st, np.argmax(q[st, :])] = 1.0
    return q, pi
