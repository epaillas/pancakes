from typing import Callable
import numpy as np


def radial_velocity(
        r: np.array,
        integrated_tpcf: Callable,
        f: float,
        bias: float
):
    """
    Radial pairwise velocity from linear perturbation
    theory.
    Args:
        r: radial pairwise separation..
        integrated_tpcf: real-space integrated tpcf.
        f: logarithmic growth rate.
        bias: linear galaxy bias.

    Returns:
        Radial pairwise velocity as a function of radial
        separation.
    """
    
    beta = f / bias
    
    return -1/3 * beta * r * integrated_tpcf(r)
