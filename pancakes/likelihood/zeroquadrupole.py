from multiprocessing import Pool
import multiprocessing
import emcee
import numpy as np
from pancakes.models.zeroquadrupole import ZeroQuadrupole


class SamplePosterior:
    """
    Samples the posterior distribution
    of the parameters of the density split
    cross-correlation function model.
    """

    def __init__(
        self,
        param_file: str
    ):

        self.model = ZeroQuadrupole(param_file)

        # configuration for MCMC
        self.ncores = self.model.params['mcmc']['ncores'] 
        self.nwalkers = self.ncores
        self.niters = 10_000
        self.stop_factor = 100
        self.burnin = 200
        self.backend = self.model.params['mcmc']['backend']

        # parameters to fit
        beta = 0.36
        f = 0.78
        b = 2.1

        self.start_params = np.array(
            [f, b]
        )

        # how parameters scale with respect to one another
        self.param_scales = [1, 1]

        # initial seed for MCMC
        self.ndim = len(self.start_params)

        self.initial_params = [
            self.start_params +
            1e-2 * np.random.rand(self.ndim) * self.param_scales
            for i in range(self.nwalkers)
        ]

    def run_mcmc(self):
        """
        Run MCMC algorithm to sample
        the posterior distribution.
        """

        backend = emcee.backends.HDFBackend(self.backend)
        backend.reset(self.nwalkers, self.ndim)

        print(multiprocessing.cpu_count())

        with Pool(processes=self.ncores) as pool:

            sampler = emcee.EnsembleSampler(
                self.nwalkers, self.ndim,
                self.log_probability,
                backend=backend,
                pool=pool
            )

            # burn-in
            state = sampler.run_mcmc(
                self.initial_params, self.burnin, progress=True
            )
            sampler.reset()
            print('Burn-in finished. Initializinig main run.')

            index = 0
            autocorr = np.empty(self.niters)
            old_tau = np.inf

            for sample in sampler.sample(
                state, iterations=self.niters, progress=True
            ):
                if sampler.iteration % 200:
                    continue

                # Compute the autocorrelation time so far
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1

                # Check convergence
                converged = np.all(tau * self.stop_factor < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    print('Convergence in all parameters achieved.')
                    break
                old_tau = tau

        print('Main run finished.')

    def log_probability(self, theta):
        """
        Log-posterior probability of the
        model parameters.
        """
        lp = self.model.log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf
        return lp + self.model.log_likelihood(theta)
