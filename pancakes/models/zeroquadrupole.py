import numpy as np
import yaml
import os
import sys
from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
from scipy.stats import norm
from ..perturbation_theory.empirical import radial_velocity
from ..utilities.utilities import multipole, read_2darray
from ..utilities.cosmology import Cosmology
import matplotlib.pyplot as plt


class ZeroQuadrupole:
    """
    Class to perform RSD & AP fits to the Density Split cross-correlation
    functons without any template. 
    """
    def __init__(
        self,
        param_file: str
    ):
        """
        Initialize class.
        Args:
            param_file: YAML file containing configuration parameters.
        """
        with open(param_file) as file:
            self.params = yaml.full_load(file)

        quadrupoles_fn = self.params['quadrupoles_fn']
        self.quadrupoles = np.load(quadrupoles_fn, allow_pickle=True) #array of shape (beta, epsilon, rbins)

        smins = [int(i) for i in str(self.params['smin']).split(',')]
        smaxs = [int(i) for i in str(self.params['smax']).split(',')]

        self.beta_grid = np.load(self.params['beta_grid_fn'])
        self.beta_prior = [self.beta_grid.min(), self.beta_grid.max()]

        self.epsilon_grid = np.load(self.params['epsilon_grid_fn'])
        self.epsilon_prior = [self.epsilon_grid.min(), self.epsilon_grid.max()]

        # read covariance matrix
        if os.path.isfile(self.params['covmat_fn']):
            self.cov = np.load(self.params['covmat_fn'])
            nbins = len(self.cov)
            nmocks = self.params['nmocks_covmat']
            if self.params['use_hartlap']:
                hartlap = (1 - (nbins + 1) / (nmocks - 1))
                self.icov = hartlap * np.linalg.inv(self.cov)
            else:
                self.icov = np.linalg.inv(self.cov)
        else:
            raise FileNotFoundError('Covariance matrix not found.')

        self.chi2_for_theta = self.chi2_interpolator()


    def chi2_interpolator(self):
        chi2_grid = np.zeros([len(self.beta_grid), len(self.epsilon_grid)])
        for ibeta, beta in enumerate(self.beta_grid):
            for ieps, epsilon in enumerate(self.epsilon_grid):
                data = self.quadrupoles[ibeta, ieps]
                model = np.zeros_like(data)

                chi2 = np.dot(np.dot((model - data),
                self.icov), model - data)
                  
                chi2_grid[ibeta, ieps] = chi2

        chi2_interpolator = RectBivariateSpline(
            self.beta_grid, self.epsilon_grid,
            chi2_grid, kx=1, ky=1
        )

        return chi2_interpolator


    def log_likelihood(self, theta):
        """
        Log-likelihood for the RSD & AP fit.
        """
        beta, epsilon = theta

        chi2 = self.chi2_for_theta(beta, epsilon)

        loglike = -0.5 * chi2

        return loglike

    def log_prior(self, theta):
        """
        Priors for the RSD & AP parameters.
        """
        beta, epsilon = theta

        if self.beta_prior[0] < beta < self.beta_prior[1]\
        and self.epsilon_prior[0] < epsilon < self.epsilon_prior[1]:
            return 0.0
        return -np.inf
