import os
import sys
import numpy as np

current_working_directory = os.getcwd()
home_name = current_working_directory.split('/')[2]

Project_directory = str(current_working_directory)
tudat_path = r"/Users/lorenz_veithen/tudat-bundle/build/tudatpy"    # TODO: have user put the TUDAT directory

sys.path.insert(0, tudat_path)
from tudatpy.interface import spice

AMS_directory = Project_directory + "/0_GeneratedData/AMS_Data"
detumbling_data_directory = Project_directory + f'/0_GeneratedData/DetumblingAnalysis'
tumbling_data_directory = Project_directory + f'/0_GeneratedData/LTT_Data'

c_sol = 299792458   # [m/s]
W = 1400    # [W/m^2] - around Earth, roughly
Sun_luminosity = 382.8 * 10**24
R_E = 6371e3    # [m]
sigmoid_start_tolerance = (1/2000)/100

# Constant with no impact on the simulation really, therefore kept here
default_ellipse_bounding_box_margin = 2

# optical models used
ACS3_opt_model_coeffs_set = np.array([0.1, 0.57, 0.74, 0.23, 0.16, 0.2, 2/3, 2/3, 0.03, 0.6])
double_ideal_opt_model_coeffs_set = np.array([0., 0., 1., 1., 0.0, 0.0, 2 / 3, 2 / 3, 1.0, 1.0])
single_ideal_opt_model_coeffs_set = np.array([0., 0., 1., 0., 0.0, 0.0, 2 / 3, 2 / 3, 1.0, 1.0])

# Load spice kernels once for ever
spice.load_standard_kernels()
