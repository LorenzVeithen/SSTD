import sys
from generalConstants import tudat_path
sys.path.insert(0, tudat_path)

# Load standard modules

from generalConstants import R_E, ACS3_opt_model_coeffs_set, double_ideal_opt_model_coeffs_set, single_ideal_opt_model_coeffs_set, sigmoid_start_tolerance
from generalConstants import detumbling_data_directory, tumbling_data_directory
import numpy as np

from attitudeControllersClass import sail_attitude_control_systems
from sailCraftClass import sail_craft
from dynamicsSim import sailCoupledDynamicsProblem
from scipy.spatial.transform import Rotation as R
from MiscFunctions import divide_list

from tudatpy.astro.element_conversion import rotation_matrix_to_quaternion_entries
from tudatpy.astro import element_conversion
from tudatpy.astro.time_conversion import DateTime
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.data import save2txt
from tudatpy.interface import spice
from tudatpy.kernel.interface import spice_interface
import time
import os

def performCrossProduct(v1, v2):
    """
    Function performing the cross product between two numpy arrays of the same shape, v1 and v2.
    This function is necessary to circumvent the 'code unreachable' Pycharm bug.
    """
    return np.cross(v1, v2)


def runSimulation(all_combinations,
                  save_dir,
                  run_mode='vane_detumbling',
                  simulation_start_epoch=DateTime(2024, 6, 1, 0).epoch(),
                  simulation_end_epoch=DateTime(2024, 6, 30, 0).epoch(),
                  initial_orbital_elements=np.array([None]),
                  initial_sun_angles_degrees=[0, 0],
                  wings_optical_model=np.array([None]), # numpy array
                  vanes_optical_model=np.array([None]), # numpy array
                  include_shadow_bool=False,
                  vane_speed_rad_s=2*np.pi,
                  vanes_dof=np.array([None]),
                  output_frequency_in_seconds_=10,
                  create_sub_dirs=True,
                  overwrite_previous=False):
    # Make it such that when the simulation is run, a file is created and an ancillary info file is also generated with all the options

    # Run mode specific information
    if ('vane_detumbling' in run_mode):
        import constants as sail_model
        acs_mode = 'vanes'
    elif ('LTT' in run_mode):
        import constants as sail_model
        acs_mode = 'None'
        sail_model.vanes_coordinates_list = []
        sail_model.vanes_origin_list = []
        sail_model.vanes_rotational_dof = np.array([])
        sail_model.vanes_rotation_matrices_list = []
        sail_model.algorithm_constants = {}
    else:
        raise Exception("Unknown run mode.")

    # If no orbital elements have been specified, use the ones from the constants file
    if (initial_orbital_elements.all() == None):
        initial_orbital_elements = [sail_model.a_0, sail_model.e_0, sail_model.i_0, sail_model.w_0, sail_model.raan_0, sail_model.theta_0]

    # Get optical properties
    if (wings_optical_model.all() != None):
        sail_model.wings_optical_properties = [wings_optical_model] * len(sail_model.wings_coordinates_list)

    if (acs_mode != 'None'):
        if (vanes_optical_model != None):
            if ((vanes_optical_model == ACS3_opt_model_coeffs_set).all()):
                sail_model.vane_optical_model_str = "ACS3_optical_model"
            elif ((vanes_optical_model == single_ideal_opt_model_coeffs_set).all()):
                sail_model.vane_optical_model_str = "single_ideal_optical_model"
            elif ((vanes_optical_model == double_ideal_opt_model_coeffs_set).all()):
                sail_model.vane_optical_model_str = "double_ideal_optical_model"
            else:
                raise Exception("Unrecognised optical coefficients")
            sail_model.vanes_optical_properties = [vanes_optical_model] * len(sail_model.vanes_coordinates_list)

    # Create the folder to store simulation results
    if (not os.path.exists(save_dir) and create_sub_dirs):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/states_history')
        os.makedirs(save_dir + '/dependent_variables_history')
    save_directory = save_dir

    # remove combinations which have already been computed if there is no overwriting
    if (not overwrite_previous):
        new_combs = []
        for comb in all_combinations:
            initial_rotational_velocity = np.array(
                [comb[0] * 2 * np.pi / 3600., comb[1] * 2 * np.pi / 3600, comb[2] * 2 * np.pi / 3600])
            rotations_per_hour = np.round(initial_rotational_velocity * 3600 / (2 * np.pi), 1)
            tentative_file = save_directory + f'/state_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat'
            if (os.path.isfile(tentative_file)):
                # if the file exists, skip this propagation
                continue
            else:
                new_combs.append(comb)
        selected_combinations = new_combs
    else:
        selected_combinations = all_combinations

    # sort the combinations by magnitude to start with the easiest
    temp_sort_array = np.empty((len(selected_combinations), 2), dtype=object)
    for si in range(len(selected_combinations)):
        temp_sort_array[si, 0] = selected_combinations[si]
        temp_sort_array[si, 1] = np.sqrt(selected_combinations[si][0]**2 + selected_combinations[si][1]**2 + selected_combinations[si][2]**2)
    sorted_temp_sort_array = temp_sort_array[np.argsort(temp_sort_array[:, 1])]
    selected_combinations = sorted_temp_sort_array[:, 0]

    # See associated paper for explanations on why these formulas are used
    sail_model.algorithm_constants["sigmoid_scaling_parameter"] = (np.log(1/sigmoid_start_tolerance - 1)
                                                                   * np.pi/(2*vane_speed_rad_s)) # [-]
    sail_model.algorithm_constants["sigmoid_time_shift_parameter"] = vane_speed_rad_s * 2/np.pi  # [s]

    if (vanes_dof.all() != None):
        sail_model.vanes_rotational_dof = vanes_dof

    for counter, combination in enumerate(selected_combinations):
        print(f"--- running {combination}, {100 * ((counter+1)/len(selected_combinations))}% ---")

        # Specify the initial attitude of the sailcraft
        constant_cartesian_position_Sun = spice_interface.get_body_cartesian_state_at_epoch('Sun',
                                                                                            'Earth',
                                                                                            'J2000',
                                                                                            'NONE',
                                                                                            simulation_start_epoch)[:3]
        new_z = constant_cartesian_position_Sun / np.linalg.norm(constant_cartesian_position_Sun)
        new_y = performCrossProduct(np.array([0, 1, 0]), new_z) / np.linalg.norm(performCrossProduct(np.array([0, 1, 0]), new_z))
        new_x = performCrossProduct(new_y, new_z) / np.linalg.norm(performCrossProduct(new_y, new_z))

        inertial_to_body_initial = np.zeros((3, 3))
        inertial_to_body_initial[:, 0] = new_x
        inertial_to_body_initial[:, 1] = new_y
        inertial_to_body_initial[:, 2] = new_z

        inertial_to_body_initial = np.dot(np.dot(inertial_to_body_initial, R.from_euler('z', initial_sun_angles_degrees[1], degrees=True).as_matrix()),
               R.from_euler('y', -initial_sun_angles_degrees[0], degrees=True).as_matrix())

        initial_quaternions = rotation_matrix_to_quaternion_entries(inertial_to_body_initial)
        initial_rotational_velocity = np.array([combination[0] * 2 * np.pi / 3600., combination[1] * 2 * np.pi / 3600, combination[2] * 2 * np.pi / 3600])
        initial_rotational_state = np.concatenate((initial_quaternions, initial_rotational_velocity))

        rotations_per_hour = np.round(initial_rotational_velocity * 3600 / (2 * np.pi), 1)
        tentative_file = save_directory + f'/states_history/state_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat'
        if (os.path.isfile(tentative_file) and overwrite_previous==False):
            # if the file exists, skip this propagation
            continue

        # Define solar sail - see constants file
        acs_object = sail_attitude_control_systems(acs_mode, sail_model.boom_list, sail_model.sail_I, sail_model.algorithm_constants, include_shadow=include_shadow_bool)
        if (acs_mode == 'vanes'):
            acs_object.set_vane_characteristics(sail_model.vanes_coordinates_list,
                                                sail_model.vanes_origin_list,
                                                sail_model.vanes_rotation_matrices_list,
                                                0,
                                                np.array([0, 0, 0]),
                                                sail_model.sail_material_areal_density,
                                                sail_model.vanes_rotational_dof,
                                                sail_model.vane_optical_model_str,  # TODO: should always match the given coefficients
                                                sail_model.wings_coordinates_list,
                                                sail_model.vane_mechanical_rotation_limits,
                                                sail_model.vanes_optical_properties,
                                                torque_allocation_problem_objective_function_weights=[sail_model.algorithm_constants["torque_allocation_problem_target_weight"],
                                                                                                      1-sail_model.algorithm_constants["torque_allocation_problem_target_weight"]])

        sail = sail_craft("ACS3",
                          len(sail_model.wings_coordinates_list),
                          len(sail_model.vanes_origin_list),
                          sail_model.wings_coordinates_list,
                          sail_model.vanes_coordinates_list,
                          sail_model.wings_optical_properties,
                          sail_model.vanes_optical_properties,
                          sail_model.sail_I,
                          sail_model.sail_mass,
                          sail_model.sail_mass_without_wings,
                          sail_model.sail_nominal_CoM,
                          sail_model.sail_material_areal_density,
                          sail_model.sail_material_areal_density,
                          acs_object)
        sail.set_desired_sail_body_frame_inertial_rotational_velocity(np.array([0., 0., 0.]))

        # Initial states
        initial_translational_state = element_conversion.keplerian_to_cartesian_elementwise(
            gravitational_parameter=spice.get_body_gravitational_parameter("Earth"),
            semi_major_axis=initial_orbital_elements[0],
            eccentricity=initial_orbital_elements[1],
            inclination=initial_orbital_elements[2],
            argument_of_periapsis=initial_orbital_elements[3],
            longitude_of_ascending_node=initial_orbital_elements[4],
            true_anomaly=initial_orbital_elements[5])

        sailProp = sailCoupledDynamicsProblem(sail,
                       initial_translational_state,
                       initial_rotational_state,
                       simulation_start_epoch,
                       simulation_end_epoch)

        dependent_variables = sailProp.define_dependent_variables(acs_object)
        bodies, vehicle_target_settings = sailProp.define_simulation_bodies()
        sail.setBodies(bodies)
        termination_settings, integrator_settings = sailProp.define_numerical_environment(integrator_coefficient_set=propagation_setup.integrator.rkf_56,
                                                                                          control_settings=propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(1.0E-12, 1.0E-12),
                                                                                          validation_settings=propagation_setup.integrator.step_size_validation(1E-5, 1E3))
        acceleration_models, torque_models = sailProp.define_dynamical_environment(bodies, acs_object, vehicle_target_settings)
        combined_propagator_settings = sailProp.define_propagators(integrator_settings, termination_settings, acceleration_models,
                                                                   torque_models, dependent_variables,
                                                                   selected_propagator_=propagation_setup.propagator.gauss_modified_equinoctial,
                                                                   output_frequency_in_seconds=output_frequency_in_seconds_)

        t0 = time.time()
        state_history, states_array, dependent_variable_history, dependent_variable_array, number_of_function_evaluations, propagation_outcome = sailProp.run_sim(bodies, combined_propagator_settings)
        t1 = time.time()


        save2txt(state_history, save_directory + f'/states_history/state_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat')
        save2txt(dependent_variable_history, save_directory + f'/dependent_variables_history/dependent_variables_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat')
        print(f'{combination}: {t1-t0}')


def runPropagationAnalysis(all_combinations,
                          optical_model_mode_str,
                          orbital_mode,
                          rank,
                          num_processes,
                          overwrite_previous=False,
                          include_shadow_bool=False,
                          run_mode='vane_detumbling',
                          output_frequency_in_seconds_=1,
                          initial_orientation_str='identity_to_inertial',
                          vane_speed_rad_s=2*np.pi,
                          dof_mode='full_2D'):
    # import different models depending on the mode considered
    max_LTT_case = False
    if ('vane_detumbling' in run_mode):
        import constants as sail_model
        sail_model.analysis_save_data_dir = detumbling_data_directory
        acs_mode = 'vanes'
        keplerian_bool = False
        selected_wings_optical_properties = [np.array(ACS3_opt_model_coeffs_set)] * len(sail_model.wings_coordinates_list)
    elif ('LTT' in run_mode):
        import constants as sail_model
        sail_model.vanes_coordinates_list = []
        sail_model.vanes_origin_list = []
        sail_model.vanes_rotational_dof = np.array([])
        sail_model.vanes_rotation_matrices_list = []
        sail_model.algorithm_constants = {}
        sail_model.analysis_save_data_dir = tumbling_data_directory
        acs_mode = 'None'
        keplerian_bool = False
        selected_vanes_optical_properties = []
        if ('sun_pointing' in run_mode and all_combinations == [(0, 0, 0)]):
            max_LTT_case = True
    else:
        raise Exception("Unknown run mode.")

    if ('keplerian' in run_mode):
        acs_mode = 'None'
        keplerian_bool = True
        all_combinations = [(0, 0, 0)]
        selected_wings_optical_properties = [np.array(double_ideal_opt_model_coeffs_set)] * len(
            sail_model.wings_coordinates_list)
        selected_vanes_optical_properties = []

    # case considered
    eccentricities = [sail_model.e_0, 0.3, 0.6]
    inclinations_deg = [np.rad2deg(sail_model.i_0), 45.0, 0.0]
    sma = ['LEO', 'MEO', 'GEO']
    sma_ecc_inc_combinations = [[sma[0], eccentricities[0], inclinations_deg[0]],
                                [sma[1], eccentricities[0], inclinations_deg[0]],
                                [sma[1], eccentricities[1], inclinations_deg[0]],
                                [sma[2], eccentricities[0], inclinations_deg[0]],
                                [sma[0], eccentricities[0], inclinations_deg[1]],
                                [sma[0], eccentricities[0], inclinations_deg[2]],
                                [sma[1], eccentricities[0], inclinations_deg[1]],
                                [sma[1], eccentricities[0], inclinations_deg[2]],
                                [sma[2], eccentricities[0], inclinations_deg[1]],
                                [sma[2], eccentricities[0], inclinations_deg[2]]]


    sma_mode = sma_ecc_inc_combinations[orbital_mode][0]
    ecc = sma_ecc_inc_combinations[orbital_mode][1]
    inc = sma_ecc_inc_combinations[orbital_mode][2]

    if (sma_mode == 'LEO'):
        initial_sma = sail_model.a_0
    elif (sma_mode == 'MEO'):
        initial_sma = R_E + 10000e3  # m
    elif (sma_mode == 'GEO'):
        initial_sma = R_E + 36000e3  # m

    initial_ecc = ecc
    initial_inc = np.deg2rad(inc)

    if ('orientation' in run_mode):
        analysis_dir = f'OrientationAnalysis/{initial_orientation_str}/'
    elif ('vane_speed' in run_mode):
        analysis_dir = f'VaneSpeedAnalysis/{np.rad2deg(vane_speed_rad_s)}/'
    elif ('reduced_dof' in run_mode):
        analysis_dir = f'ReducedDoFAnalysis/{dof_mode}/'
    elif (max_LTT_case==False and 'sun_pointing' in run_mode):
        if (not ('sun_pointing' in initial_orientation_str)):
            raise Exception('Not sun pointing initial attitude for LTT Sun Pointing analysis')
        analysis_dir = f'Sun_Pointing/'
    else:
        analysis_dir = ''

    if ('few_orbits' in run_mode):
        analysis_dir = f'SingleOrbit/{analysis_dir}'

    # get directory and correct optical properties
    if (optical_model_mode_str == "ACS3_optical_model"):
        if ('vane_detumbling' in run_mode):
            selected_vanes_optical_properties = [np.array(ACS3_opt_model_coeffs_set)] * len(sail_model.vanes_coordinates_list)
        elif ('LTT' in run_mode):
            selected_wings_optical_properties = [np.array(ACS3_opt_model_coeffs_set)] * len(sail_model.wings_coordinates_list)
        save_sub_dir = f'{analysis_dir}{sma_mode}_ecc_{np.round(ecc, 1)}_inc_{np.round(np.rad2deg(initial_inc))}/NoAsymetry_data_ACS3_opt_model_shadow_{bool(include_shadow_bool)}'
    elif (optical_model_mode_str == "double_ideal_optical_model"):
        if ('vane_detumbling' in run_mode):
            selected_vanes_optical_properties = [np.array(double_ideal_opt_model_coeffs_set)] * len(sail_model.vanes_coordinates_list)
        elif ('LTT' in run_mode):
            selected_wings_optical_properties = [np.array(double_ideal_opt_model_coeffs_set)] * len(sail_model.wings_coordinates_list)
        save_sub_dir = f'{analysis_dir}{sma_mode}_ecc_{np.round(ecc, 1)}_inc_{np.round(np.rad2deg(initial_inc))}/NoAsymetry_data_double_ideal_opt_model_shadow_{bool(include_shadow_bool)}'
    elif (optical_model_mode_str == "single_ideal_optical_model"):
        if ('vane_detumbling' in run_mode):
            selected_vanes_optical_properties = [np.array(single_ideal_opt_model_coeffs_set)] * len(sail_model.vanes_coordinates_list)
        elif ('LTT' in run_mode):
            selected_wings_optical_properties = [np.array(single_ideal_opt_model_coeffs_set)] * len(sail_model.wings_coordinates_list)
        save_sub_dir = f'{analysis_dir}{sma_mode}_ecc_{np.round(ecc, 1)}_inc_{np.round(np.rad2deg(initial_inc), 1)}/NoAsymetry_data_single_ideal_opt_model_shadow_{bool(include_shadow_bool)}'
    else:
        raise Exception("Unrecognised optical model mode in detumbling propagation")

    if (not os.path.exists(sail_model.analysis_save_data_dir + f'/{save_sub_dir}') and rank == 0):
        os.makedirs(sail_model.analysis_save_data_dir + f'/{save_sub_dir}/states_history')
        os.makedirs(sail_model.analysis_save_data_dir + f'/{save_sub_dir}/dependent_variable_history')
    save_directory = sail_model.analysis_save_data_dir + f'/{save_sub_dir}'
    print(save_directory)
    if (keplerian_bool == False and max_LTT_case==False):
        # remove combinations which have already been done
        if (not overwrite_previous):
            new_combs = []
            for comb in all_combinations:
                initial_rotational_velocity = np.array(
                    [comb[0] * 2 * np.pi / 3600., comb[1] * 2 * np.pi / 3600, comb[2] * 2 * np.pi / 3600])
                rotations_per_hour = np.round(initial_rotational_velocity * 3600 / (2 * np.pi), 1)
                tentative_file = save_directory + f'/states_history/state_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat'
                if (os.path.isfile(tentative_file)):
                    # if the file exists, skip this propagation
                    continue
                else:
                    new_combs.append(comb)
            all_combinations = new_combs

        # cut into the number of parallel processes and take the required chunk
        chunks_list = divide_list(all_combinations, num_processes)
        selected_combinations = chunks_list[rank]
    else:
        selected_combinations = all_combinations

    # Set simulation start and end epochs
    if ('few_orbits' in run_mode):
        simulation_start_epoch = DateTime(2024, 6, 1, 0).epoch()
        simulation_end_epoch = DateTime(2024, 6, 1, 9).epoch()  # 90 minutes into the future but the simulation will likely finish way earlier
    else:
        simulation_start_epoch = DateTime(2024, 6, 1, 0).epoch()
        simulation_end_epoch = DateTime(2024, 6, 30, 0).epoch()  # 30 days into the future but the simulation will likely finish way earlier

    # sort the combinations by magnitude to start with the easiest
    temp_sort_array = np.empty((len(selected_combinations), 2), dtype=object)
    for si in range(len(selected_combinations)):
        temp_sort_array[si, 0] = selected_combinations[si]
        temp_sort_array[si, 1] = np.sqrt(selected_combinations[si][0]**2 + selected_combinations[si][1]**2 + selected_combinations[si][2]**2)
    sorted_temp_sort_array = temp_sort_array[np.argsort(temp_sort_array[:, 1])]
    selected_combinations = sorted_temp_sort_array[:, 0]

    if (run_mode == 'vane_detumbling_vane_speed'):
        # See paper associated for explanation on why these formulas are used
        sail_model.algorithm_constants["sigmoid_scaling_parameter"] = (np.log(1/sigmoid_start_tolerance - 1)
                                                                       * np.pi/(2*vane_speed_rad_s)) # [-] but is related to the rate of change of the vane angles
        sail_model.algorithm_constants["sigmoid_time_shift_parameter"] = vane_speed_rad_s * 2/np.pi  # [s]
        print(f'time shift: {sail_model.algorithm_constants["sigmoid_scaling_parameter"]}')
        print(f'sigmoid scaling: {sail_model.algorithm_constants["sigmoid_time_shift_parameter"]}')

    if (run_mode == 'vane_detumbling_reduced_dof'):
        #["full_2D", 'Wie2004', '1_stuck_vane']
        if (dof_mode == 'full_2D'):
            sail_model.vanes_rotational_dof = np.array([[True, True], [True, True], [True, True], [True, True]])
        elif (dof_mode == '1_reduced_y'):
            sail_model.vanes_rotational_dof = np.array([[True, False], [True, True], [True, True], [True, True]])
        elif (dof_mode == 'Wie2004'):
            sail_model.vanes_rotational_dof = np.array([[True, False], [False, True], [True, False], [False, True]])
        elif (dof_mode == '1_stuck_vane'):
            sail_model.vanes_rotational_dof = np.array([[False, False], [True, True], [True, True], [True, True]])
        elif (dof_mode == '1_reduced_x'):
            sail_model.vanes_rotational_dof = np.array([[False, True], [True, True], [True, True], [True, True]])
        elif (dof_mode == 'full_1_dof_x'):
            sail_model.vanes_rotational_dof = np.array([[True, False], [True, False], [True, False], [True, False]])
        elif (dof_mode == 'full_1_dof_y'):
            sail_model.vanes_rotational_dof = np.array([[False, True], [False, True], [False, True], [False, True]])

    for counter, combination in enumerate(selected_combinations):
        print(f"--- running {combination}, {100 * ((counter+1)/len(selected_combinations))}% ---")

        # initial rotational state
        constant_cartesian_position_Sun = spice_interface.get_body_cartesian_state_at_epoch('Sun',
                                                                                            'Earth',
                                                                                            'J2000',
                                                                                            'NONE',
                                                                                            simulation_start_epoch)[:3]
        if (initial_orientation_str == 'identity_to_inertial'):
            new_x = np.array([1., 0, 0])
            new_y = np.array([0, 1., 0])
            new_z = np.array([0, 0, 1.])

        elif (initial_orientation_str == 'sun_pointing'):
            new_z = constant_cartesian_position_Sun / np.linalg.norm(constant_cartesian_position_Sun)
            new_y = np.cross(np.array([0, 1, 0]), new_z) / np.linalg.norm(np.cross(np.array([0, 1, 0]), new_z))
            new_x = np.cross(new_y, new_z) / np.linalg.norm(np.cross(new_y, new_z))

        elif (initial_orientation_str == 'edge-on-y'
              or initial_orientation_str == 'alpha_45_beta_90'
              or initial_orientation_str == 'alpha_45_beta_0'):
            new_y = constant_cartesian_position_Sun / np.linalg.norm(constant_cartesian_position_Sun)
            new_z = np.cross(np.array([0, 1, 0]), new_y) / np.linalg.norm(np.cross(np.array([0, 1, 0]), new_y))
            new_x = np.cross(new_y, new_z) / np.linalg.norm(np.cross(new_y, new_z))

        elif (initial_orientation_str == 'edge-on-x'):
            new_x = constant_cartesian_position_Sun / np.linalg.norm(constant_cartesian_position_Sun)
            new_z = np.cross(np.array([0, 1, 0]), new_x) / np.linalg.norm(np.cross(np.array([0, 1, 0]), new_x))
            new_y = np.cross(new_x, new_z) / np.linalg.norm(np.cross(new_x, new_z))

        inertial_to_body_initial = np.zeros((3, 3))
        inertial_to_body_initial[:, 0] = new_x
        inertial_to_body_initial[:, 1] = new_y
        inertial_to_body_initial[:, 2] = new_z

        if (initial_orientation_str == 'alpha_45_beta_90'):
            inertial_to_body_initial = np.dot(inertial_to_body_initial, R.from_euler('x', 45.,
                                                                                     degrees=True).as_matrix())  # rotate by 45 deg around x
        elif (initial_orientation_str == 'alpha_45_beta_0'):
            inertial_to_body_initial = np.dot(np.dot(inertial_to_body_initial, R.from_euler('x', 90., degrees=True).as_matrix()),
                   R.from_euler('y', 45., degrees=True).as_matrix())

        initial_quaternions = rotation_matrix_to_quaternion_entries(inertial_to_body_initial)
        initial_rotational_velocity = np.array([combination[0] * 2 * np.pi / 3600., combination[1] * 2 * np.pi / 3600, combination[2] * 2 * np.pi / 3600])
        initial_rotational_state = np.concatenate((initial_quaternions, initial_rotational_velocity))

        rotations_per_hour = np.round(initial_rotational_velocity * 3600 / (2 * np.pi), 1)
        tentative_file = save_directory + f'/states_history/state_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat'
        if (os.path.isfile(tentative_file) and overwrite_previous==False and keplerian_bool==False and max_LTT_case==False):
            # if the file exists, skip this propagation
            continue

        # Define solar sail - see constants file
        acs_object = sail_attitude_control_systems(acs_mode, sail_model.boom_list, sail_model.sail_I, sail_model.algorithm_constants, include_shadow=include_shadow_bool)
        if (acs_mode == 'vanes'
                or run_mode == 'keplerian_vane_detumbling'
                or run_mode == 'keplerian_vane_detumbling_orientation'
                or run_mode == 'keplerian_vane_detumbling_vane_speed'
                or run_mode == 'keplerian_vane_detumbling_reduced_dof'):
            acs_object.set_vane_characteristics(sail_model.vanes_coordinates_list,
                                                sail_model.vanes_origin_list,
                                                sail_model.vanes_rotation_matrices_list,
                                                0,
                                                np.array([0, 0, 0]),
                                                0.0045,
                                                sail_model.vanes_rotational_dof,
                                                optical_model_mode_str,
                                                sail_model.wings_coordinates_list,
                                                sail_model.vane_mechanical_rotation_limits,
                                                selected_vanes_optical_properties,
                                                torque_allocation_problem_objective_function_weights=[2. / 3., 1. / 3.])

        sail = sail_craft("ACS3",
                          len(sail_model.wings_coordinates_list),
                          len(sail_model.vanes_origin_list),
                          sail_model.wings_coordinates_list,
                          sail_model.vanes_coordinates_list,
                          selected_wings_optical_properties,
                          selected_vanes_optical_properties,
                          sail_model.sail_I,
                          sail_model.sail_mass,
                          sail_model.sail_mass_without_wings,
                          sail_model.sail_nominal_CoM,
                          sail_model.sail_material_areal_density,
                          sail_model.sail_material_areal_density,
                          acs_object)
        sail.set_desired_sail_body_frame_inertial_rotational_velocity(np.array([0., 0., 0.]))

        # Initial states
        initial_translational_state = element_conversion.keplerian_to_cartesian_elementwise(
            gravitational_parameter=398600441500000.0,
            semi_major_axis=initial_sma,
            eccentricity=initial_ecc,
            inclination=initial_inc,
            argument_of_periapsis=sail_model.w_0,
            longitude_of_ascending_node=sail_model.raan_0,
            true_anomaly=sail_model.theta_0)

        sailProp = sailCoupledDynamicsProblem(sail,
                       initial_translational_state,
                       initial_rotational_state,
                       simulation_start_epoch,
                       simulation_end_epoch)

        dependent_variables = sailProp.define_dependent_variables(acs_object, keplerian_bool=keplerian_bool)
        bodies, vehicle_target_settings = sailProp.define_simulation_bodies()
        sail.setBodies(bodies)
        if (keplerian_bool):
            termination_settings, integrator_settings = sailProp.define_numerical_environment(integrator_coefficient_set=propagation_setup.integrator.rkf_56,
                                                                                                control_settings=propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(1.0E-12, 1.0E-12),
                                                                                                validation_settings=propagation_setup.integrator.step_size_validation(
                                                                                                  1E-5, 1))
        else:
            termination_settings, integrator_settings = sailProp.define_numerical_environment(integrator_coefficient_set=propagation_setup.integrator.rkf_56,
                                                                                          control_settings=propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(1.0E-12, 1.0E-12),)
        acceleration_models, torque_models = sailProp.define_dynamical_environment(bodies, acs_object, vehicle_target_settings,
                                                                                   keplerian_bool=keplerian_bool)
        combined_propagator_settings = sailProp.define_propagators(integrator_settings, termination_settings, acceleration_models,
                                                                   torque_models, dependent_variables,
                                                                   selected_propagator_=propagation_setup.propagator.gauss_modified_equinoctial,
                                                                   output_frequency_in_seconds=output_frequency_in_seconds_)

        t0 = time.time()
        state_history, states_array, dependent_variable_history, dependent_variable_array, number_of_function_evaluations, propagation_outcome = sailProp.run_sim(bodies, combined_propagator_settings)
        t1 = time.time()

        if (keplerian_bool):
            save2txt(state_history,
                     save_directory + f'/keplerian_orbit_state_history.dat')
            save2txt(dependent_variable_history,
                     save_directory + f'/keplerian_orbit_dependent_variable_history.dat')
        elif (max_LTT_case):
            save2txt(state_history,
                     save_directory + f'/sun_pointing_state_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat')
            save2txt(dependent_variable_history,
                     save_directory + f'/sun_pointing_dependent_variable_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat')
        else:
            save2txt(state_history, save_directory + f'/states_history/state_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat')
            save2txt(dependent_variable_history, save_directory + f'/dependent_variable_history/dependent_variable_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat')


        print(f'{combination}: {t1-t0}')
