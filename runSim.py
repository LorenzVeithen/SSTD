from solarSailPropagationFunction import runSimulation
import numpy as np
from tudatpy.astro.time_conversion import DateTime
from generalConstants import Project_directory


runSimulation([(5, 5, 5)],  # rotations per hour in each axis of the body-fixed frame
                  save_dir=Project_directory + '/0_GeneratedData/test_run_detumbling',
                  run_mode='LTT',
                  simulation_start_epoch=DateTime(2024, 6, 1, 0).epoch(),
                  simulation_end_epoch=DateTime(2024, 6, 30, 0).epoch(),
                  output_frequency_in_seconds_=100,
                  overwrite_previous=True)

