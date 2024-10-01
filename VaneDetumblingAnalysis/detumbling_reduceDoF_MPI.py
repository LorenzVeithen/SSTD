from solarSailPropagationFunction import runPropagationAnalysis
from mpi4py import MPI
import numpy as np
import itertools
import sys
import random
from MiscFunctions import divide_list

random.seed(42)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
n_processes = size

number_of_portions = 1      # > 1
portion = int(sys.argv[1])  # 0, 1, 2, 3,... depending on number of portions

reduced_dof_list = ["full_1_dof_x"]     # full_2D
mode_combinations = list(itertools.product([0, 1, 2], [0], [0], reduced_dof_list))
mode_combs_chunks = divide_list(mode_combinations, number_of_portions)
current_chunk = mode_combs_chunks[portion]

all_combinations = [(5, 0, 0),
                    (0, 5, 0),
                    (0, 0, 5),
                    (5, 5, 0),
                    (0, 5, 5),
                    (5, 0, 5),
                    (5, 5, 5)
                    ]

chunks_list = divide_list(current_chunk, n_processes)
selected_mode_combinations = chunks_list[rank]

for mode_comb in selected_mode_combinations:
    optical_model_mode = mode_comb[0]
    sma_ecc_inc_combination_mode = mode_comb[1]
    include_shadow_b = mode_comb[2]
    dof_mode_str = mode_comb[3]

    optical_mode_str = ["ACS3_optical_model", "double_ideal_optical_model", "single_ideal_optical_model"][
        optical_model_mode]
    print(f'rank: {rank}: dof_mode = {dof_mode_str}, {optical_mode_str}')
    # run keplerian
    runPropagationAnalysis(all_combinations,
                           optical_mode_str,
                           sma_ecc_inc_combination_mode,
                           0,
                           1,
                           overwrite_previous=False,
                           include_shadow_bool=bool(include_shadow_b),
                           run_mode='keplerian_vane_detumbling_reduced_dof',
                           output_frequency_in_seconds_=2,
                           initial_orientation_str="sun_pointing",
                           dof_mode=dof_mode_str
                           )

    # run the actual propagation
    runPropagationAnalysis(all_combinations,
                              optical_mode_str,
                              sma_ecc_inc_combination_mode,
                              0,
                              1,
                              overwrite_previous=False,
                              include_shadow_bool=bool(include_shadow_b),
                              run_mode='vane_detumbling_reduced_dof',
                              output_frequency_in_seconds_=2,
                              initial_orientation_str="sun_pointing",
                              dof_mode=dof_mode_str
                              )
