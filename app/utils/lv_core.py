from __future__ import annotations

from typing import Tuple


def base_lv_ode(
    s: float,
    w: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> Tuple[float, float]:
    ds_dt = alpha * s - beta * s * w
    dw_dt = -gamma * w + delta * s * w
    return ds_dt, dw_dt


def lv_star_ode(
    s: float,
    w: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    K: float,
    A: float,
) -> Tuple[float, float]:
    if s >= K:
        # At capacity we can shrink but not grow
        ds_dt = min(0.0, alpha * s - beta * s * w)
    else:
        ds_dt = alpha * s - beta * s * w

    dw_dt = -gamma * w + delta * s * w
    return ds_dt, dw_dt


def base_lv_ode_ivp(
    t: float,
    x: Tuple[float, float],
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> Tuple[float, float]:
    s, w = x
    return base_lv_ode(s, w, alpha, beta, gamma, delta)


def lv_star_ode_ivp(
    t: float,
    x: Tuple[float, float],
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    K: float,
    A: float,
) -> Tuple[float, float]:
    s, w = x
    return lv_star_ode(s, w, alpha, beta, gamma, delta, K, A)


def allee_sheep_event(
    t: float,
    x: Tuple[float, float],
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    K: float,
    A: float,
) -> float:
    return x[0] - A


def allee_wolf_event(
    t: float,
    x: Tuple[float, float],
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    K: float,
    A: float,
) -> float:
    return x[1] - A


def allee_terminal_event(
    t: float,
    x: Tuple[float, float],
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    K: float,
    A: float,
) -> float:
    return max(x[0] - A, x[1] - A)


def reset_events() -> None:
    # Configure event properties to reflect intended behavior
    allee_sheep_event.terminal = False
    allee_wolf_event.terminal = False
    allee_terminal_event.terminal = True


