import numpy as np
from scipy.spatial.transform import Rotation as R
from generalConstants import R_E
from generalConstants import (single_ideal_opt_model_coeffs_set, double_ideal_opt_model_coeffs_set,
                              ACS3_opt_model_coeffs_set)

analysis_save_data_dir = ''


# initial orbit
a_0 = R_E + 1000e3           # [m] initial spacecraft semi-major axis
e_0 = 4.03294322e-03         # [-] initial spacecraft eccentricity
i_0 = np.deg2rad(98.0131)    # [deg] initial spacecraft inclination
w_0 = np.deg2rad(120.0)      # [deg] initial spacecraft argument of pericentre
raan_0 = np.deg2rad(27.0)    # [deg] initial spacecraft RAAN
theta_0 = np.deg2rad(275.0)  # [deg] initial spacecraft true anomaly

# Sail characteristics - using ACS3 as baseline for initial testing
sail_mass = 16                              # kg
sail_mass_without_wings = 15.66             # kg
sail_I = np.zeros((3, 3))                   # kg m^2
sail_I[0, 0] = 10.5                         # kg m^2
sail_I[1, 1] = 10.5                         # kg m^2
sail_I[2, 2] = 21                           # kg m^2
sail_nominal_CoM = np.array([0., 0., 0.])   # m
sail_material_areal_density = 0.00425       # kg/m^2
vane_mechanical_rotation_limits = ([-np.pi, -np.pi], [np.pi, np.pi])    # rad

# Sail shape
boom_length = 7.                # m
boom_attachment_point = 0.64    # m

# Wings characteristics
wing_optical_coefficients = ACS3_opt_model_coeffs_set    # [-]

# Vane characteristics
vane_angle = np.deg2rad(30.)    # deg
vane_side_length = 0.5          # m
vane_optical_model_str = "ACS3_optical_model"
vanes_rotational_dof = np.array([[True, True], [True, True], [True, True], [True, True]])

# Detumbling algorithm constants
algorithm_constants = {}
algorithm_constants["tol_vane_angle_determination_start_golden_section"] = 1e-3
algorithm_constants["tol_vane_angle_determination_golden_section"] = 1e-3
algorithm_constants["tol_vane_angle_determination"] = 1e-3

algorithm_constants["tol_torque_allocation_problem_constraint"] = 1e-7
algorithm_constants["tol_torque_allocation_problem_objective"] = 1e-4
algorithm_constants["tol_torque_allocation_problem_x"] = 1e-3

algorithm_constants["max_rotational_velocity_orientation_change_update_vane_angles_degrees"] = 8  # [deg]
algorithm_constants["max_sunlight_vector_body_frame_orientation_change_update_vane_angles_degrees"] = 7  # [deg]
algorithm_constants["max_relative_change_in_rotational_velocity_magnitude"] = 0.1  # [-]

# Error threholds above which a robust global algorithm is called
algorithm_constants["max_vane_torque_orientation_error"] = 15.  # [deg]
algorithm_constants["max_vane_torque_relative_magnitude_error"] = 0.25  # [-]

# purely design, not really tuning parameters
algorithm_constants["sigmoid_scaling_parameter"] = 3        # [-] but is related to the rate of change of the vane angles
algorithm_constants["sigmoid_time_shift_parameter"] = 4     # [s]
algorithm_constants["vane_controller_shut_down_rotational_velocity_tolerance"] = 0.01
algorithm_constants["torque_allocation_problem_target_weight"] = 2/3


if (vane_optical_model_str == "double_ideal_optical_model"):
    vane_optical_coefficients = single_ideal_opt_model_coeffs_set
elif (vane_optical_model_str == "single_ideal_optical_model"):
    vane_optical_coefficients = double_ideal_opt_model_coeffs_set
elif (vane_optical_model_str == "ACS3_optical_model"):
    vane_optical_coefficients = ACS3_opt_model_coeffs_set
else:
    raise Exception("Unsupported vane optical model")


# Wings properties
boom1 = np.array([[0, 0, 0], [0, boom_length, 0]])
boom2 = np.array([[0, 0, 0], [boom_length, 0, 0]])
boom3 = np.array([[0, 0, 0], [0, -boom_length, 0]])
boom4 = np.array([[0, 0, 0], [-boom_length, 0, 0]])
boom_list = [boom1, boom2, boom3, boom4]

panel1 = np.array([[boom_attachment_point, 0., 0.],
                   [boom_length, 0., 0.],
                   [0., boom_length, 0.],
                   [0., boom_attachment_point, 0.]])

panel2 = np.array([[0., -boom_attachment_point, 0.],
                    [0., -boom_length, 0.],
                    [boom_length, 0., 0.],
                    [boom_attachment_point, 0., 0.]])

panel3 = np.array([[-boom_attachment_point, 0., 0.],
                   [-boom_length, 0., 0.],
                   [0., -boom_length, 0.],
                   [0., -boom_attachment_point, 0.]])

panel4 = np.array([[0., boom_attachment_point, 0.],
                    [0., boom_length, 0.],
                    [-boom_length, 0., 0.],
                    [-boom_attachment_point, 0., 0.]])

wings_coordinates_list = [panel1, panel2, panel3, panel4]
wings_optical_properties = [wing_optical_coefficients] * len(wings_coordinates_list)

wings_rotation_matrices_list = [R.from_euler('z', -45., degrees=True).as_matrix(),
                                R.from_euler('z', -135., degrees=True).as_matrix(),
                                R.from_euler('z', -225., degrees=True).as_matrix(),
                                R.from_euler('z', -315., degrees=True).as_matrix()]


vanes_rotation_matrices_list = [R.from_euler('z', 90., degrees=True).as_matrix(),
                                R.from_euler('z', 0., degrees=True).as_matrix(),
                                R.from_euler('z', 270., degrees=True).as_matrix(),
                                R.from_euler('z', 180., degrees=True).as_matrix(),
                                ]

vanes_origin_list = [np.array([0., boom_length, 0.]),
                     np.array([boom_length, 0., 0.]),
                     np.array([0, -boom_length, 0.]),
                     np.array([-boom_length, 0., 0.]),
                     ]

vanes_coordinates_list = []
for i in range(len(vanes_origin_list)):
    current_vane_coords_body_frame_coords = vanes_origin_list[i]
    current_vane_rotation_matrix_body_to_vane = vanes_rotation_matrices_list[i]
    current_vane_rotation_matrix_vane_to_body = current_vane_rotation_matrix_body_to_vane

    second_point_body_frame = (np.dot(current_vane_rotation_matrix_vane_to_body,
                                     np.array([np.sin(vane_angle) * vane_side_length, -np.cos(vane_angle) * vane_side_length, 0]))
                               + current_vane_coords_body_frame_coords)

    third_point_body_frame = (np.dot(current_vane_rotation_matrix_vane_to_body,
                                     np.array([np.sin(vane_angle) * vane_side_length, np.cos(vane_angle) * vane_side_length, 0]))
                               + current_vane_coords_body_frame_coords)

    current_vane_coords_body_frame_coords = np.vstack((current_vane_coords_body_frame_coords, second_point_body_frame, third_point_body_frame))
    vanes_coordinates_list.append(current_vane_coords_body_frame_coords)

vanes_optical_properties = [vane_optical_coefficients] * len(vanes_origin_list)