import sys
from os import path
import numpy as np
import subprocess
import yaml
from scipy.io import FortranFile
from .cosmology import Cosmology
from .utilities import sky_to_cartesian


class DensitySplit:
    """
    Class to perform the density split algorithm on
    a cosmological survey, as in 
    https://arxiv.org/abs/1810.02864.
    """

    def __init__(self, parameter_file: str):
        """
        Initialize class.

        Args:
            parameter_file: YAML file containing configuration parameters.
        """

        with open(parameter_file) as file:
            self.params = yaml.full_load(file)
    
    def random_points(self):
        """
        Generates the initial random points for the
        density split algorithm. These can either be
        subsampled from an input file, or be randomly
        selected from a uniform distribution within
        a rectangular boundary.
        
        Args:
            parameter_file: YAML file containing configuration parameters.
        """
        np.random.seed(0)  # set random seed for reproducibility

        sampling_filename = self.params['sampling_filename']
        if self.params['seeds_method'] == 'subsampling':
            if not path.isfile(sampling_filename):
                raise FileNotFoundError(f'{sampling_filename} not found.')

            # only text file supported for now
            sampling_data = np.genfromtxt(self.params['sampling_filename'])
            idx = np.random.choice(
                len(sampling_data), size=self.params['nseeds'],
                replace=False
            )
            seeds = sampling_data[idx]

            # convert to comoving coordinates if necessary
            if self.params['convert_seeds']:
                if self.params['omega_m'] is None:
                    raise ValueError('If convert_coordinates is True, '
                                     'omega_m needs to be specified.')
            cosmo = Cosmology(omega_m=self.params['omega_m'])
            seeds = sky_to_cartesian(seeds, cosmo)

        elif self.params['seeds_method'] == 'uniform':
            x = np.random.uniform(
                self.params['seeds_xmin'],
                self.params['seeds_xmax'],
                self.params['nseeds']
            )
            y = np.random.uniform(
                self.params['seeds_ymin'],
                self.params['seeds_ymax'],
                self.params['nseeds']
            )
            z = np.random.uniform(
                self.params['seeds_zmin'],
                self.params['seeds_zmax'],
                self.params['nseeds']
            )

            seeds = np.c_[x, y, z]
        else:
            sys.exit('Sampling method not recognized')

        # save to file
        handle = self.params['handle']
        seeds_filename = f'{handle}_seeds.dat'
        np.savetxt(seeds_filename, seeds)


def filtered_density(
    data_filename1, data_filename2, random_filename2,
    output_filename, dim1_min, dim1_max,
    filter_type, filter_size, ngrid, 
    gridmin, gridmax, random_filename1=None,
    nthreads=1, estimator='DP', output_format='unformatted',
    input_format='unformatted', use_weights=True
):

    # check if files exist
    for filename in [data_filename1,  data_filename2,
                     random_filename2]:
        if not path.isfile(filename):
            raise FileNotFoundError(f'{filename} not found.')

    if estimator == 'LS' and random_filename1 is None:
        raise RuntimeError('Lady-Szalay estimator requires a random catalogue'
                           'for dataset 1.')

    if random_filename1 is None:
        random_filename1 = random_filename2

    if use_weights:
        use_weights = 1
    else: 
        use_weights = 0

    if dim1_max is None:
        if filter_type == 'tophat':
            dim1_max = filter_size
        elif filter_type == 'gaussian':
            dim1_max = 5 * filter_size

    binpath = path.join(path.dirname(__file__),
    'bin', '{}_filter.exe'.format(filter_type))

    cmd = [
        binpath, data_filename1, data_filename2,
        random_filename1, random_filename2, output_filename,
        str(dim1_min), str(dim1_max), str(filter_size),
        str(ngrid), str(gridmin), str(gridmax),
        estimator, str(nthreads), str(use_weights),
        input_format
    ]

    subprocess.call(cmd)

    # open filter file
    f = FortranFile(output_filename, 'r')
    smoothed_delta = f.read_ints()[0]
    smoothed_delta = f.read_reals(dtype=np.float64)
    f.close()

    if output_format != 'unformatted':
        if output_format == 'npy':
            subprocess.call(['rm', output_filename])
        elif output_format == 'ascii':
            np.savetxt(output_filename, smoothed_delta)
        else:
            print('Output format not recognized. Using unformatted F90 file.')
        np.save(output_filename, smoothed_delta)
  
    return smoothed_delta


def split_centres(
  centres_filename, filter_filename, quantiles,
  handle=None, output_format='unformatted'
):

  # read centres
  # first check if this is a numpy file
  if '.npy' in centres_filename:
    centres = np.load(centres_filename)
  else:
    # if not, check if it is a text file
    try:
      centres = np.genfromtxt(centres_filename)
    except:
      # else, check if it is an unformatted file
      try:
        fin = FortranFile(centres_filename, 'r')
        nrows = fin.read_ints()[0]
        ncols = fin.read_ints()[0]
        centres = fin.read_reals(dtype=np.float64).reshape(nrows, ncols)
      except:
        sys.exit('Format of centres file not recognized.')

  # read smoothed densities
  # first check if this is a numpy file
  if '.npy' in filter_filename:
    smoothed_delta = np.load(filter_filename)
    ncentres = len(smoothed_delta)
  else:
    # if not, check if it is a text file
    try:
      smoothed_delta = np.genfromtxt(filter_filename)
      ncentres = len(smoothed_delta)
    except:
      # else, check if it is an unformatted file
      try:
        f = FortranFile(filter_filename, 'r')
        ncentres = f.read_ints()[0]
        smoothed_delta = f.read_reals(dtype=np.float64)
        f.close()
      except:
        sys.exit('Format of filter file not recognized.')
  idx = np.argsort(smoothed_delta)

  # sort centres using smoothed densities
  sorted_centres = centres[idx]

  # generate quantiles
  binned_centres = {}
  for i in range(1, quantiles + 1):
      binned_centres['DS{}'.format(i)] = sorted_centres[int((i-1)*ncentres/quantiles):int(i*ncentres/quantiles)]
      cout = binned_centres['DS{}'.format(i)]

      if handle != None:
            output_filename = handle + '_DS{}'.format(i)
      else:
          output_filename = centres_filename.split('.unf')[0] + '_DS{}'.format(i)
      
      if output_format == 'unformatted':
        output_filename += '.unf'
        f = FortranFile(output_filename, 'w')
        f.write_record(np.shape(cout)[0])
        f.write_record(np.shape(cout)[1])
        f.write_record(cout)
        f.close()
      
      elif output_format == 'ascii':
        output_filename += '.dat'
        np.savetxt(output_filename, cout)

      elif output_format == 'npy':
        np.save(output_filename, cout)
      
      else:
        sys.exit('Output format not recognized.')

  return binned_centres
