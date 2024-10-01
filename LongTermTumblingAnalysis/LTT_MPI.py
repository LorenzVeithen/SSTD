from solarSailPropagationFunction import runPropagationAnalysis
from mpi4py import MPI
import numpy as np
import itertools
import sys

overwrite_previous_bool = False

# argv[0] is the executable
optical_model_mode = int(sys.argv[1])  # 0: ACS3 optical model, 1: double-sided  ideal model, 2: single-sided ideal model
sma_ecc_inc_combination_mode = int(sys.argv[2])    # see below for the combinations

optical_mode_str = ["ACS3_optical_model", "double_ideal_optical_model", "single_ideal_optical_model"][optical_model_mode]


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
n_processes = size

# generate all combinations of rotational velocity components from a selected subset to analyse
# for later
# [-100, -85, -70, -55, -40, -30, -20, -10, 0, 100, 85, 70, 55, 40, 30, 20, 10]
omega_x_list = np.array([-100, -85, -70, -55, -40, -30, -20, -10, 0, 100, 85, 70, 55, 40, 30, 20, 10])
omega_y_list = np.array([-100, -85, -70, -55, -40, -30, -20, -10, 0, 100, 85, 70, 55, 40, 30, 20, 10])
omega_z_list = np.array([0])

all_combinations = list(itertools.product(omega_x_list, omega_y_list, omega_z_list))

print(f"hello from rank {rank}")
if (rank==0):
    runPropagationAnalysis(all_combinations,
                           optical_mode_str,
                           sma_ecc_inc_combination_mode,
                           rank,
                           size,
                           overwrite_previous=False,
                           include_shadow_bool=False,
                           run_mode='keplerian_LTT',
                           output_frequency_in_seconds_=1)

runPropagationAnalysis(all_combinations,
                          optical_mode_str,
                          sma_ecc_inc_combination_mode,
                          rank,
                          size,
                          overwrite_previous=False,
                          include_shadow_bool=False,
                          run_mode='LTT',
                          output_frequency_in_seconds_=100)

