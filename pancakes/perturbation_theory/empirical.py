from typing import Callable
import numpy as np


def radial_velocity(
        r: np.array,
        tpcf: Callable,
        integrated_tpcf: Callable,
        f: float,
        bias: float,
        nu: float,
        correlation_type: str
):
    """
    Radial pairwise velocity from the empirical model
    introduced in https://arxiv.org/abs/2101.09854
    Args:
        r: radial pairwise separation.
        tpcf: real-space tpcf
        integrated_tpcf: integrated real-space tpcf
        f: logarithmic growth rate
        bias: linear galaxy bias
        nu: free parameter
        correlation_type: 'autocorrelation' or
        'cross-correlation'

    Returns:
        Radial pairwise velocity as a function of radial
        separation.
    """
    beta = f / bias

    if correlation_type == 'autocorrelation':
        int2_xi_r = integrated_tpcf(r) / (1 + tpcf(r))
        v_r = -2/3 * beta * r * int2_xi_r * \
            (1 + nu * int2_xi_r)
    else:
        v_r = -1/3 * beta * r * integrated_tpcf(r) / \
            (1 + nu * tpcf(r))

    return v_r
