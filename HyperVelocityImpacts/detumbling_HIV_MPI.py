from solarSailPropagationFunction import runPropagationAnalysis
from mpi4py import MPI
import numpy as np
import sys
import random

random.seed(42)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
n_processes = size

# argv[0] is the executable
optical_model_mode = int(sys.argv[1])  # 0: ACS3 optical model, 1: double-sided  ideal model, 2: single-sided ideal model
sma_ecc_inc_combination_mode = 0   # see below for the combinations
include_shadow_b = 0     #0: False (no shadow), 1: True (with shadow)
optical_mode_str = ["ACS3_optical_model", "double_ideal_optical_model", "single_ideal_optical_model"][optical_model_mode]


omega_list = np.arange(0, 1000, 10)  # rotations per hour

all_combinations = []
for omega in omega_list:
    all_combinations.append((omega, omega, omega))
    #all_combinations.append((omega, omega, 0))

random.shuffle(all_combinations)
print(f"hello from rank {rank}")
runPropagationAnalysis(all_combinations,
                          optical_mode_str,
                          sma_ecc_inc_combination_mode,
                          rank,
                          size,
                          overwrite_previous=False,
                          include_shadow_bool=bool(include_shadow_b),
                          run_mode='vane_detumbling_few_orbits',
                          output_frequency_in_seconds_=10,
                          initial_orientation_str='sun_pointing')
