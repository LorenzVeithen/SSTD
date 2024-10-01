from solarSailPropagationFunction import runSimulation
import numpy as np
from tudatpy.astro.time_conversion import DateTime
from generalConstants import Project_directory


runSimulation([(5, 5, 5)],
                  Project_directory + '/0_GeneratedData/test_run_detumbling',
                  run_mode='vane_detumbling',
                  simulation_start_epoch=DateTime(2024, 6, 1, 0).epoch(),
                  simulation_end_epoch=DateTime(2024, 6, 30, 0).epoch(),
                  initial_sun_angles_degrees=[90, 0],
                  wings_optical_model=np.array([None]),  # numpy array
                  vanes_optical_model=np.array([None]),  # numpy array
                  output_frequency_in_seconds_=100,
                  overwrite_previous=True)

