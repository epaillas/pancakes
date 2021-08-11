from multiprocessing import Pool
import emcee
import numpy as np
from pancakes.models.densitysplit import DensitySplitCCF


class SamplePosterior:
    """
    Samples the posterior distribution
    of the parameters of the density split
    cross-correlation function model.
    """

    def __init__(
        self,
        xi_smu_filename: str,
        xi_r_filename: str,
        sv_rmu_filename: str,
        covmat_filename: str,
        backend: str,
        smin: float,
        smax: float,
        effective_z: float,
        multipoles: str = '0+2+4',
        ncores: int = 1,
        omega_m: float = 0.301,
        sigma_8: float = 0.828,
        velocity_coupling: str = 'empirical',
        bias_mocks: float = None,
        use_hartlap: bool = True,
        rescale_rspace: bool = False,
        use_reconstruction: bool = False,
        betagrid_filename: str = None
    ):

        # figure out number of bins
        ndenbins = len(xi_r_filename.split(','))

        # configuration for MCMC
        self.ncores = ncores
        self.nwalkers = ncores
        self.niters = 10_000
        self.stop_factor = 100
        self.burnin = 200
        self.backend = backend

        # parameters to fit
        fs8 = 0.4727
        bs8 = 1.2880
        q_perp = 1.0
        q_para = 1.0
        sigma_v = 360
        nu_i = [0.5] * ndenbins

        self.start_params = np.array(
            [fs8, bs8, sigma_v,
             q_perp, q_para] + nu_i
        )

        # how parameters scale with respect to one another
        self.param_scales = [1, 100, 1, 1] + \
            [1] * ndenbins

        # initial seed for MCMC
        self.ndim = len(self.start_params)

        self.initial_params = [
            self.start_params +
            1e-2 * np.random.rand(self.ndim) * self.param_scales
            for i in range(self.nwalkers)
        ]

        self.model = DensitySplitCCF(
            xi_smu_filename=xi_smu_filename,
            xi_r_filename=xi_r_filename,
            sv_rmu_filename=sv_rmu_filename,
            covmat_filename=covmat_filename,
            betagrid_filename=betagrid_filename,
            smin=smin,
            smax=smax,
            multipoles=multipoles,
            velocity_coupling=velocity_coupling,
            rescale_rspace=rescale_rspace,
            use_hartlap=use_hartlap,
            omega_m=omega_m,
            sigma_8=sigma_8,
            effective_z=effective_z,
            bias_mocks=bias_mocks,
            use_reconstruction=use_reconstruction
        )

    def run_mcmc(self):
        """
        Run MCMC algorithm to sample
        the posterior distribution.
        """
        with Pool(processes=self.ncores) as pool:

            sampler = emcee.EnsembleSampler(
                self.nwalkers, self.ndim,
                self.log_probability,
                backend=self.backend,
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
