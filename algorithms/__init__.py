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


def on_policy_first_visit_monte_carlo_control(
        states_count: int,
        actions_count: int,
        reset_func: Callable,
        is_terminal_func: Callable,
        step_func: Callable,
        episodes_count: int = 10000,
        max_steps_per_episode: int = 10,
        epsilon: float = 0.2,
        gamma: float = 0.99,
) -> (np.ndarray, np.ndarray):
    states = np.arange(states_count)
    pi = tabular_uniform_random_policy(states_count, actions_count)
    q = np.random.random((states_count, actions_count))
    for s in states:
        if is_terminal_func(s):
            q[s, :] = 0.0
            pi[s, :] = 0.0

    returns = np.zeros((states_count, actions_count))
    returns_count = np.zeros((states_count, actions_count))
    for episode_id in range(episodes_count):
        s0 = reset_func()

        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_return_history(s0, pi, is_terminal_func,
                                                                                         step_func,
                                                                                         max_steps_per_episode)

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
            pi[st, :] = epsilon / actions_count
            pi[st, np.argmax(q[st, :])] = 1.0 - epsilon + epsilon / actions_count
    return q, pi


def off_policy_monte_carlo_control(
        states_count: int,
        actions_count: int,
        reset_func: Callable,
        is_terminal_func: Callable,
        step_func: Callable,
        episodes_count: int = 10000,
        max_steps_per_episode: int = 10,
        epsilon: float = 0.2,
        gamma: float = 0.99,
) -> (np.ndarray, np.ndarray):
    states = np.arange(states_count)
    b = tabular_uniform_random_policy(states_count, actions_count)
    pi = np.zeros((states_count, actions_count))
    C = np.zeros((states_count, actions_count))
    q = np.random.random((states_count, actions_count))
    for s in states:
        if is_terminal_func(s):
            q[s, :] = 0.0
            pi[s, :] = 0.0
        pi[s, :] = 0.0
        pi[s, np.argmax(q[s, :])] = 1.0

    for episode_id in range(episodes_count):
        s0 = reset_func()

        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_return_history(s0, b, is_terminal_func,
                                                                                         step_func,
                                                                                         max_steps_per_episode)

        G = 0
        W = 1
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            at = a_list[t]

            C[st, at] += W

            q[st, at] += W / C[st, at] * (G - q[st, at])
            pi[st, :] = 0.0
            pi[st, np.argmax(q[st, :])] = 1.0

            if at != np.argmax(q[st, :]):
                break

            W = W / b[st, at]

    return q, pi


def tabular_td_zero_prediction(
        pi: np.ndarray,
        is_terminal_func: Callable,
        reset_func: Callable,
        step_func: Callable,
        episodes_count: int = 100000,
        max_steps_per_episode: int = 100,
        gamma: float = 0.99,
        alpha: float = 0.01
) -> np.ndarray:
    states = np.arange(pi.shape[0])
    actions = np.arange(pi.shape[1])
    V = np.random.random(pi.shape[0])
    for s in states:
        if is_terminal_func(s):
            V[s] = 0

    for episode_id in range(episodes_count):
        s = reset_func()

        step = 0
        while not is_terminal_func(s) and step < max_steps_per_episode:
            a = np.random.choice(actions, p=pi[s])
            (s_p, r, t) = step_func(s, a)
            V[s] += alpha * (r + gamma * V[s_p] - V[s])
            s = s_p
            step += 1

    return V


def tabular_sarsa_control(
        states_count: int,
        actions_count: int,
        reset_func: Callable,
        is_terminal_func: Callable,
        step_func: Callable,
        episodes_count: int = 50000,
        max_steps_per_episode: int = 10,
        epsilon: float = 0.2,
        alpha: float = 0.1,
        gamma: float = 0.99,
) -> (np.ndarray, np.ndarray):
    states = np.arange(states_count)
    actions = np.arange(actions_count)
    q = np.random.random((states_count, actions_count))
    for s in states:
        if is_terminal_func(s):
            q[s, :] = 0.0

    for episode_id in range(episodes_count):
        s = reset_func()
        rdm = np.random.random()
        a = np.random.choice(actions) if rdm < epsilon else np.argmax(q[s, :])
        step = 0
        while not is_terminal_func(s) and step < max_steps_per_episode:
            (s_p, r, t) = step_func(s, a)
            rdm = np.random.random()
            a_p = np.random.choice(actions) if rdm < epsilon else np.argmax(q[s_p, :])
            q[s, a] += alpha * (r + gamma * q[s_p, a_p] - q[s, a])
            s = s_p
            a = a_p
            step += 1

    pi = np.zeros_like(q)
    for s in states:
        pi[s, :] = epsilon / actions_count
        pi[s, np.argmax(q[s, :])] = 1.0 - epsilon + epsilon / actions_count

    return q, pi


def tabular_q_learning_control(
        states_count: int,
        actions_count: int,
        reset_func: Callable,
        is_terminal_func: Callable,
        step_func: Callable,
        episodes_count: int = 50000,
        max_steps_per_episode: int = 10,
        epsilon: float = 0.2,
        alpha: float = 0.1,
        gamma: float = 0.99,
) -> (np.ndarray, np.ndarray):
    states = np.arange(states_count)
    actions = np.arange(actions_count)
    q = np.random.random((states_count, actions_count))
    for s in states:
        if is_terminal_func(s):
            q[s, :] = 0.0

    for episode_id in range(episodes_count):
        s = reset_func()
        step = 0
        while not is_terminal_func(s) and step < max_steps_per_episode:
            rdm = np.random.random()
            a = np.random.choice(actions) if rdm < epsilon else np.argmax(q[s, :])
            (s_p, r, t) = step_func(s, a)
            q[s, a] += alpha * (r + gamma * np.max(q[s_p, :]) - q[s, a])
            s = s_p
            step += 1

    pi = np.zeros_like(q)
    for s in states:
        pi[s, :] = 0.0
        pi[s, np.argmax(q[s, :])] = 1.0

    return q, pi
