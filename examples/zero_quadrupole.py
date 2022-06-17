from pancakes.likelihood.zeroquadrupole import SamplePosterior


if __name__ == '__main__':

    param_file = ('zero_quadrupole.yaml')
    model = SamplePosterior(param_file=param_file)
    model.run_mcmc()

