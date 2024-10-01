import numpy as np
from MiscFunctions import all_equal, closest_point_on_a_segment_to_a_third_point, compute_panel_geometrical_properties
from ACS_dynamicalModels import vane_dynamical_model
from vaneControllerMethods import buildEllipseCoefficientFunctions, ellipseCoefficientFunction, vaneTorqueAllocationProblem
from vaneControllerMethods import sigmoid_transition, vaneAngleAllocationScaling
from generalConstants import AMS_directory, c_sol, W, Sun_luminosity
import pygmo as pg
from scipy.optimize import minimize, golden
from time import time
from AMSDerivation.truncatedEllipseCoefficientsFunctions import ellipse_truncated_coefficients_function_shadow_FALSE_double_ideal_optical_model, ellipse_truncated_coefficients_function_shadow_TRUE_double_ideal_optical_model
from AMSDerivation.truncatedEllipseCoefficientsFunctions import ellipse_truncated_coefficients_function_shadow_FALSE_single_ideal_optical_model, ellipse_truncated_coefficients_function_shadow_TRUE_single_ideal_optical_model
from AMSDerivation.truncatedEllipseCoefficientsFunctions import ellipse_truncated_coefficients_function_shadow_FALSE_ACS3_optical_model, ellipse_truncated_coefficients_function_shadow_TRUE_ACS3_optical_model

class sail_attitude_control_systems:
    def __init__(self, ACS_system, booms_coordinates_list, spacecraft_inertia_tensor, algorithm_constants={}, include_shadow=False, sail_craft_name="ACS3", sim_start_epoch=0):
        #TODO: remove booms_coordinates_list as it is not used

        # General
        self.sail_attitude_control_system = ACS_system          # String defining the ACS to be used
        self.bool_mass_based_controller = None                  # Boolean indicating if the ACS concept is mass-based. TODO: should this be depracated?
        self.ACS_mass = 0                                       # [kg] Total mass of the ACS. Initialised to zero and each instance of set_... adds to this sum
        self.ACS_CoM = None                                     # [m] Body-fixed center of mass of the total ACS. Initialised to the center of the spacecraft.
        self.include_shadow = include_shadow
        self.sail_craft_name = sail_craft_name
        self.spacecraft_inertia_tensor = spacecraft_inertia_tensor
        self.latest_updated_time = sim_start_epoch

        # Vanes
        self.number_of_vanes = 0                                # [] Number of vanes of the ACS.
        self.vane_panels_coordinates_list = None                # [m] num_of_vanes long list of (num_of_vanes x 3) arrays of the coordinates of the polygons defining the vanes of the ACS.
        self.vane_reference_frame_origin_list = None            # [m] num_of_vanes long list of (1x3) arrays of the coordinates of the vane coordinate frame origins, around which the vane rotations are defined.
        self.vane_reference_frame_rotation_matrix_list = None   # num_of_vanes long list of (3x3) rotation matrices from the body fixed frame to the vane fixed frame.
        self.vane_material_areal_density = None
        self.vanes_rotational_dof_booleans = None               # num_of_vanes long list of lists of booleans [True, True] stating the rotational degree of freedom of each vane. 0: x and 1:y in vane coordinate frames
        self.vanes_areas_list = None
        self.latest_updated_vane_torques = [None]
        self.latest_updated_optimal_torque_allocation = [None]
        self.vane_mechanical_rotation_limits = None
        self.latest_updated_vane_angles = [[None]]
        self.body_fixed_rotational_velocity_at_last_vane_angle_update = [None]
        self.body_fixed_sunlight_vector_at_last_angle_update = None
        self.allow_update = True

        # Summation variables
        self.ACS_CoM_stationary_components = np.array([0, 0, 0])

        if (ACS_system == "vanes"):
            self.tol_vane_angle_determination_start_golden_section = algorithm_constants["tol_vane_angle_determination_start_golden_section"]
            self.tol_vane_angle_determination_golden_section = algorithm_constants["tol_vane_angle_determination_golden_section"]
            self.tol_vane_angle_determination = algorithm_constants["tol_vane_angle_determination"]

            self.tol_torque_allocation_problem_constraint = algorithm_constants["tol_torque_allocation_problem_constraint"]
            self.tol_torque_allocation_problem_objective = algorithm_constants["tol_torque_allocation_problem_objective"]
            self.tol_torque_allocation_problem_x = algorithm_constants["tol_torque_allocation_problem_x"]

            self.tol_rotational_velocity_orientation_change_update_vane_angles_degrees = algorithm_constants["max_rotational_velocity_orientation_change_update_vane_angles_degrees"]
            self.tol_sunlight_vector_body_frame_orientation_change_update_vane_angles_degrees = algorithm_constants["max_sunlight_vector_body_frame_orientation_change_update_vane_angles_degrees"]
            self.tol_relative_change_in_rotational_velocity_magnitude = algorithm_constants["max_relative_change_in_rotational_velocity_magnitude"]

            self.maximum_vane_torque_orientation_error = algorithm_constants["max_vane_torque_orientation_error"]
            self.maximum_vane_torque_relative_magnitude_error = algorithm_constants["max_vane_torque_relative_magnitude_error"]

            self.vane_controller_shut_down_rotational_velocity_tolerance = algorithm_constants["vane_controller_shut_down_rotational_velocity_tolerance"]

            self.sigmoid_scaling_parameter = algorithm_constants["sigmoid_scaling_parameter"]
            self.sigmoid_time_shift_parameter = algorithm_constants["sigmoid_time_shift_parameter"]

        # Dependent variables dictionaries
        self.actuator_states = {}
        self.actuator_states["vane_rotation_x_default"] = np.zeros((self.number_of_vanes, 1))
        self.actuator_states["vane_rotation_y_default"] = np.zeros((self.number_of_vanes, 1))

        self.actuator_states["vane_rotation_x"] = self.actuator_states["vane_rotation_x_default"]
        self.actuator_states["vane_rotation_y"] = self.actuator_states["vane_rotation_y_default"]

        self.random_variables_dict = {}
        self.random_variables_dict["optimal_torque"] = np.array([0., 0., 0.]).reshape(-1, 1)
        self.random_variables_dict["vane_torques"] = np.array([0., 0., 0.]).reshape(-1, 1)

    def computeBodyFrameTorqueForDetumbling(self, bodies, tau_max, desired_rotational_velocity_vector=np.array([0, 0, 0]), rotational_velocity_tolerance_rotations_per_hour=0.1, timeToPassivateACS=0):
        """
        Function computing the required torque for detumbling the spacecraft to rest. For a time-independent attitude
        control system, this function can be evaluated a single time.

        :param bodies:  tudatpy.kernel.numerical_simulation.environment.SystemOfBodies object containing the information
        on the bodies present in the TUDAT simulation.
        :param tau_max: Maximum input torque of the ACS at a given time.
        :param desired_rotational_velocity_vector=np.array([0, 0, 0]): desired final rotational velocity vector.
        :param rotational_velocity_tolerance=0.1: tolerance on the magnitude of the largest absolute  value of
        the components of the rotational velocity vector. Detumbling torque then becomes zero.
        :param timeToPassivateACS=0: Estimated time to passivate the attitude control system, to avoid a discontinuous
        actuator control.
        :return tau_star: the optimal control torque.

        References:
        Aghili, F. (2009). Time-optimal detumbling control of spacecraft. Journal of guidance, control, and dynamics,
        32(5), 1671-1675.
        """
        body_fixed_angular_velocity_vector = bodies.get_body(self.sail_craft_name).body_fixed_angular_velocity
        if (np.linalg.norm(desired_rotational_velocity_vector - np.array([0, 0, 0]))>1e-15):
            if (len(desired_rotational_velocity_vector[desired_rotational_velocity_vector > 0]) > 1):
                raise Exception("The desired final rotational velocity vector in " +
                                "computeBodyFrameTorqueForDetumbling " +
                                "has more than one non-zero element. Spin-stabilised spacecraft should be about an " +
                                "Eigen-axis.")
            elif (np.count_nonzero(
                    self.spacecraft_inertia_tensor - np.diag(np.diagonal(self.spacecraft_inertia_tensor))) != 0):
                raise Exception("computeBodyFrameTorqueForDetumbling is only valid for " +
                                " axisymmetric spacecrafts.")
        omega_tilted = body_fixed_angular_velocity_vector - desired_rotational_velocity_vector

        if (max(abs(omega_tilted)) * 3600 / (2 * np.pi) < rotational_velocity_tolerance_rotations_per_hour):
            return np.array([0, 0, 0])

        sail_craft_inertia_tensor = self.spacecraft_inertia_tensor

        inertiaTensorTimesAngularVelocity = np.dot(sail_craft_inertia_tensor, omega_tilted)
        predictedTimeToRest = np.linalg.norm(inertiaTensorTimesAngularVelocity)/tau_max
        if ((predictedTimeToRest < timeToPassivateACS) and (timeToPassivateACS != 0)):
            tau_target = (timeToPassivateACS - predictedTimeToRest)/timeToPassivateACS  # Linearly decreasing the torque applied such that the ACS is turned OFF smoothly
        else:
            tau_target = tau_max
        tau_star = - (inertiaTensorTimesAngularVelocity/np.linalg.norm(inertiaTensorTimesAngularVelocity)) * tau_target
        return tau_star.reshape(-1, 1)

    def attitude_control(self,
                         bodies,
                         desired_sail_body_frame_inertial_rotational_velocity,
                         current_time):
        # Returns an empty array if nothing has changed
        wings_coordinates = []
        wings_optical_properties = []
        vanes_coordinates = []
        vanes_optical_properties = []

        moving_masses_CoM_components = np.array([0, 0, 0])
        moving_masses_positions = {}
        self.actuator_states["vane_rotation_x"] = self.actuator_states["vane_rotation_x_default"]
        self.actuator_states["vane_rotation_y"] = self.actuator_states["vane_rotation_y_default"]
        if (bodies != None):
            match self.sail_attitude_control_system:
                case "gimball_mass":
                    self.__pure_gimball_mass(bodies, desired_sail_body_frame_inertial_rotational_velocity)
                    moving_masses_CoM_components = np.zeros([0, 0, 0])
                    moving_masses_positions["gimball_mass"] = np.array([0, 0, 0], dtype="float64")
                case "vanes":
                    # Here comes the controller of the vanes, which will give the rotations around the x and y axis in the
                    # vane coordinate frame
                    sunlight_vector_inertial_frame = (bodies.get_body(self.sail_craft_name).position - bodies.get_body(
                        "Sun").position) / np.linalg.norm(
                        bodies.get_body(self.sail_craft_name).position - bodies.get_body("Sun").position)
                    R_IB = bodies.get_body(self.sail_craft_name).inertial_to_body_fixed_frame
                    sunlight_vector_body_frame = np.dot(R_IB, sunlight_vector_inertial_frame)

                    if (self.body_fixed_rotational_velocity_at_last_vane_angle_update[0] != None):
                        # Check how much the rotational velocity vector orientation has changed
                        c_rotational_velocity_vector_orientation_change = np.dot(
                            bodies.get_body(self.sail_craft_name).body_fixed_angular_velocity,
                            self.body_fixed_rotational_velocity_at_last_vane_angle_update) / (np.linalg.norm(
                            bodies.get_body(self.sail_craft_name).body_fixed_angular_velocity) * np.linalg.norm(
                            self.body_fixed_rotational_velocity_at_last_vane_angle_update))
                        if (abs(c_rotational_velocity_vector_orientation_change-1) < 1e-15):
                            c_rotational_velocity_vector_orientation_change = 1.
                        change_in_rotational_velocity_orientation_rad = np.arccos(c_rotational_velocity_vector_orientation_change)

                        # Check how much the rotational velocity vector magnitude has changed
                        relative_change_in_rotational_velocity_magnitude = ((np.linalg.norm(bodies.get_body(self.sail_craft_name).body_fixed_angular_velocity)
                                                                            -np.linalg.norm(self.body_fixed_rotational_velocity_at_last_vane_angle_update))/
                                                                            np.linalg.norm(self.body_fixed_rotational_velocity_at_last_vane_angle_update))

                        # Check how much the sunlight vector in the body frame has changed
                        cos_sunlight_vector_orientation_change = np.dot(sunlight_vector_body_frame, self.body_fixed_sunlight_vector_at_last_angle_update) / (np.linalg.norm(sunlight_vector_body_frame) * np.linalg.norm(self.body_fixed_sunlight_vector_at_last_angle_update))

                        # handle special cases giving NaN's
                        if (abs(cos_sunlight_vector_orientation_change-1) < 1e-15):
                            cos_sunlight_vector_orientation_change = 1.
                        change_in_body_fixed_sunlight_vector_orientation_rad = np.arccos(cos_sunlight_vector_orientation_change)

                    else:
                        # dummy values to force update
                        change_in_rotational_velocity_orientation_rad = 10.
                        relative_change_in_rotational_velocity_magnitude = 2.
                        change_in_body_fixed_sunlight_vector_orientation_rad = -1.
                        self.latest_updated_vane_angles = np.zeros((self.number_of_vanes, 2))

                    if ((np.rad2deg(change_in_rotational_velocity_orientation_rad) > self.tol_rotational_velocity_orientation_change_update_vane_angles_degrees
                        or np.rad2deg(change_in_rotational_velocity_orientation_rad) < 0
                        or np.rad2deg(change_in_body_fixed_sunlight_vector_orientation_rad) > self.tol_sunlight_vector_body_frame_orientation_change_update_vane_angles_degrees
                        or np.rad2deg(change_in_body_fixed_sunlight_vector_orientation_rad) < 0
                        or relative_change_in_rotational_velocity_magnitude > self.tol_relative_change_in_rotational_velocity_magnitude)
                        and self.allow_update):

                        sail_sun_distance = np.linalg.norm(bodies.get_body(self.sail_craft_name).position - bodies.get_body("Sun").position)
                        current_solar_irradiance = Sun_luminosity/(4 * np.pi * sail_sun_distance**2) # W

                        required_body_torque = self.computeBodyFrameTorqueForDetumbling(bodies,
                                                                                        5e-5,
                                                                                        desired_rotational_velocity_vector=desired_sail_body_frame_inertial_rotational_velocity,
                                                                                        rotational_velocity_tolerance_rotations_per_hour=self.vane_controller_shut_down_rotational_velocity_tolerance,
                                                                                        timeToPassivateACS=0)
                        required_body_torque = required_body_torque.reshape(-1)/(current_solar_irradiance/c_sol)

                        # If the required torque is non-zero, control the vanes accordingly
                        if (np.linalg.norm(required_body_torque) > 1e-15):
                            previous_optimal_torque = self.latest_updated_optimal_torque_allocation if (self.latest_updated_optimal_torque_allocation[0]==None) else self.latest_updated_optimal_torque_allocation * c_sol / current_solar_irradiance

                            controller_vane_angles, vane_torques, optimal_torque_allocation = self.vane_system_angles_from_desired_torque(self,
                                                                                                            self.vane_mechanical_rotation_limits,
                                                                                                            required_body_torque,
                                                                                                            previous_optimal_torque,
                                                                                                            sunlight_vector_body_frame,
                                                                                                            initial_vane_angles_guess_rad=self.latest_updated_vane_angles)
                            controller_vane_angles = controller_vane_angles * self.vanes_rotational_dof_booleans    # Assume that if reduced DoF, the vane angle stays at zero TODO: generalise to any stuck position
                        else:
                            controller_vane_angles = np.zeros((self.number_of_vanes, 2))
                            vane_torques = np.zeros((self.number_of_vanes, 3))
                            optimal_torque_allocation = np.zeros((self.number_of_vanes, 3))
                            previous_optimal_torque = self.latest_updated_vane_torques

                        vane_torques = vane_torques * current_solar_irradiance / c_sol
                        optimal_torque_allocation = optimal_torque_allocation * current_solar_irradiance / c_sol
                        #print(optimal_torque_allocation)
                        #print(f"required torque:{required_body_torque}")
                        #print(f"optimal torque:{optimal_torque_allocation.sum(axis=0)* (current_solar_irradiance / c_sol)**-1}")
                        #print(f"vane torque: {vane_torques.sum(axis=0) * (current_solar_irradiance / c_sol)**-1}")
                        #print(np.rad2deg(controller_vane_angles))
                        #print(f"rotations per hour {bodies.get_body(self.sail_craft_name).body_fixed_angular_velocity * 3600 / (2 * np.pi)}")

                        self.latest_updated_vane_torques = vane_torques.reshape(-1)
                        self.previous_vane_angles = self.latest_updated_vane_angles
                        self.latest_updated_vane_angles = controller_vane_angles
                        self.latest_updated_optimal_torque_allocation = optimal_torque_allocation.reshape(-1)
                        self.body_fixed_rotational_velocity_at_last_vane_angle_update = bodies.get_body(self.sail_craft_name).body_fixed_angular_velocity
                        self.body_fixed_sunlight_vector_at_last_angle_update = sunlight_vector_body_frame
                        self.latest_updated_time = current_time

                    #print( self.latest_updated_vane_angles)
                    sig_vane_angles = sigmoid_transition(current_time,
                                                         self.latest_updated_vane_angles,
                                                         self.latest_updated_time,
                                                         self.previous_vane_angles,
                                                         scaling_parameter=self.sigmoid_scaling_parameter,
                                                         shift_time_parameter=self.sigmoid_time_shift_parameter)

                    # check if sigmoids have converged
                    den = abs(self.latest_updated_vane_angles - self.previous_vane_angles)
                    den[np.where(den<1e-15)] = 1e-8
                    sigmoids_convergence_bool = np.amax(100 * abs((sig_vane_angles - self.latest_updated_vane_angles) / (den))) < 1.
                    if (sigmoids_convergence_bool
                    or np.amax(abs(sig_vane_angles-self.latest_updated_vane_angles)) < 1e-15):
                        self.allow_update = True
                    else:
                        self.allow_update = False
                    vane_x_rotation_degrees, vane_y_rotation_degrees = np.rad2deg(sig_vane_angles[:, 0]),  np.rad2deg(sig_vane_angles[:, 1])
                    self.actuator_states["vane_rotation_x"] = np.deg2rad(vane_x_rotation_degrees.reshape(-1, 1))
                    self.actuator_states["vane_rotation_y"] = np.deg2rad(vane_y_rotation_degrees.reshape(-1, 1))
                    self.random_variables_dict["optimal_torque"] = (self.latest_updated_optimal_torque_allocation.reshape((self.number_of_vanes, 3)).sum(axis=0)).reshape(-1, 1)
                    self.random_variables_dict["vane_torques"] = (self.latest_updated_vane_torques.reshape((self.number_of_vanes, 3)).sum(axis=0)).reshape(-1, 1)
                    vanes_coordinates = self.__vane_dynamics(vane_x_rotation_degrees, vane_y_rotation_degrees)

                case "None":
                    # No attitude control system - the spacecraft remains inert
                    pass
                case _:
                    raise Exception("Selected ACS not available... yet.")

        self.compute_attitude_system_center_of_mass(vanes_coordinates, moving_masses_CoM_components)
        # the attitude-control algorithm should give the output
        panels_coordinates, panels_optical_properties = {}, {}
        panels_coordinates["wings"] = wings_coordinates
        panels_coordinates["vanes"] = vanes_coordinates
        panels_optical_properties["wings"] = wings_optical_properties
        panels_optical_properties["vanes"] = vanes_optical_properties
        return panels_coordinates, panels_optical_properties, self.ACS_CoM, moving_masses_positions  # TODO: Be careful to link self.ACS_CoM with the outgoing variable here

    def compute_attitude_system_center_of_mass(self, vanes_coordinates, moving_masses_CoM_components):
        # Compute the complete ACS center of mass (excluding the components due to the wings)
        if (self.ACS_mass == 0):
            self.ACS_CoM = np.array([0, 0, 0])
        else:
            ACS_CoM = np.array([0, 0, 0], dtype="float64")
            ACS_CoM += self.ACS_CoM_stationary_components
            ACS_CoM += moving_masses_CoM_components
            if (len(vanes_coordinates) != 0):
                for i in range(len(vanes_coordinates)):
                    vane_centroid, vane_area, _ = compute_panel_geometrical_properties(vanes_coordinates[i])
                    ACS_CoM += vane_centroid * (vane_area * self.vane_material_areal_density)

            self.ACS_CoM = ACS_CoM / self.ACS_mass
        return True

    # Controllers
    def __pure_gimball_mass(self, current_sail_state, desired_sail_state):
        return

    # ACS characteristics inputs and dynamics
    def set_vane_characteristics(self, vanes_coordinates_list,
                                 vanes_reference_frame_origin_list,
                                 vanes_reference_frame_rotation_matrix_list,
                                 stationary_system_components_mass,
                                 stationary_system_system_CoM,
                                 vanes_material_areal_density,
                                 vanes_rotational_dof_booleans,
                                 vane_optical_model_str,
                                 wings_coordinates_list,
                                 vane_mechanical_rotation_bounds,
                                 vanes_optical_properties,
                                 torque_allocation_problem_objective_function_weights=[1, 0],
                                 directory_feasibility_ellipse_coefficients=f'{AMS_directory}/Datasets/Ideal_model/vane_1/dominantFitTerms',
                                 number_shadow_mesh_nodes=10):
        """
        Function setting the characteristics of the ACS vanes actuator.
        Should be called a single time.
        :param vanes_coordinates_list:
        :param vanes_reference_frame_origin_list:
        :param vanes_reference_frame_rotation_matrix_list:
        :param stationary_system_components_mass:
        :param stationary_system_system_CoM:
        :param vanes_material_areal_density:
        :return: True if the process was completed successfully
        """
        self.ACS_mass += stationary_system_components_mass  # WITHOUT VANES, which are taken into account below
        self.ACS_CoM_stationary_components += stationary_system_components_mass * stationary_system_system_CoM
        self.number_of_vanes = len(vanes_reference_frame_origin_list)
        self.vane_panels_coordinates_list = vanes_coordinates_list
        self.vane_reference_frame_origin_list = vanes_reference_frame_origin_list
        self.vane_reference_frame_rotation_matrix_list = vanes_reference_frame_rotation_matrix_list
        self.vanes_rotational_dof_booleans = vanes_rotational_dof_booleans
        self.actuator_states["vane_rotation_x_default"] = np.zeros((self.number_of_vanes, 1))
        self.actuator_states["vane_rotation_y_default"] = np.zeros((self.number_of_vanes, 1))
        self.vane_mechanical_rotation_limits = [(vane_mechanical_rotation_bounds[0][i], vane_mechanical_rotation_bounds[1][i]) for i in
                                       range(len(vane_mechanical_rotation_bounds[0]))]

        # Determine vane component of the ACS mass
        vanes_areas = []
        for i in range(len(self.vane_panels_coordinates_list)):
            _, vane_area, _ = compute_panel_geometrical_properties(self.vane_panels_coordinates_list[i])
            vanes_areas.append(vane_area)

        self.vanes_areas_list = vanes_areas
        self.vane_material_areal_density = vanes_material_areal_density
        self.ACS_mass = sum(vanes_areas) * vanes_material_areal_density

        # Determine if a vane is on a boom
        self.vane_is_aligned_on_body_axis = [False] * self.number_of_vanes
        for vane_id, vane in enumerate(self.vane_panels_coordinates_list):
            vane_attachment_point = vane[0, :]  # as per convention
            if (np.shape(np.nonzero(vane_attachment_point)[0])[0]==1):
                self.vane_is_aligned_on_body_axis[vane_id] = True
        self.vane_is_aligned_on_body_axis = np.array(self.vane_is_aligned_on_body_axis)

        # feasible torque ellipse coefficients from pre-computed functions
        if (vane_optical_model_str == "double_ideal_optical_model"):
            vane_has_ideal_model = True
            if (self.include_shadow):
                ellipse_coefficient_functions_list = ellipse_truncated_coefficients_function_shadow_TRUE_double_ideal_optical_model()
            else:
                ellipse_coefficient_functions_list = ellipse_truncated_coefficients_function_shadow_FALSE_double_ideal_optical_model()
        elif (vane_optical_model_str == "single_ideal_optical_model"):
            vane_has_ideal_model = True
            if (self.include_shadow):
                ellipse_coefficient_functions_list = ellipse_truncated_coefficients_function_shadow_TRUE_single_ideal_optical_model()
            else:
                ellipse_coefficient_functions_list = ellipse_truncated_coefficients_function_shadow_FALSE_single_ideal_optical_model()
        elif (vane_optical_model_str == "ACS3_optical_model"):
            vane_has_ideal_model = False
            if (self.include_shadow):
                ellipse_coefficient_functions_list = ellipse_truncated_coefficients_function_shadow_TRUE_ACS3_optical_model()
            else:
                ellipse_coefficient_functions_list = ellipse_truncated_coefficients_function_shadow_FALSE_ACS3_optical_model()
        elif (vane_optical_model_str == "AMS_Derivation"):
            vane_has_ideal_model = False
            ellipse_coefficient_functions_list = []
        else:
            raise Exception("Requested set of ellipse coefficients have not been explicitly implemented yet")

        # vane torque allocation problem
        self.vane_torque_allocation_problem_object = vaneTorqueAllocationProblem(self,
                                                                  wings_coordinates_list,
                                                                  vane_has_ideal_model,
                                                                  self.include_shadow,
                                                                  ellipse_coefficient_functions_list,
                                                                  vanes_optical_properties,
                                                                  w1=torque_allocation_problem_objective_function_weights[0],
                                                                  w2=torque_allocation_problem_objective_function_weights[1],
                                                                  num_shadow_mesh_nodes=number_shadow_mesh_nodes)

        # initial the first vane angles
        self.latest_updated_vane_angles = np.zeros((self.number_of_vanes, 2))
        return True

    def __vane_dynamics(self, rotation_x_deg, rotation_y_deg):
        # Get the vane panel coordinates as a result of the rotation
        # Based on the initial vane position and orientation in the body frame
        if (not all(np.rad2deg(self.vane_mechanical_rotation_limits[0][0]) <= angle <= np.rad2deg(self.vane_mechanical_rotation_limits[0][1]) for angle in rotation_x_deg)
                or not all(np.rad2deg(self.vane_mechanical_rotation_limits[1][0]) <= angle <= np.rad2deg(self.vane_mechanical_rotation_limits[1][1]) for angle in rotation_y_deg)):
            print(all(np.rad2deg(self.vane_mechanical_rotation_limits[0][0]) <= angle <= np.rad2deg(self.vane_mechanical_rotation_limits[0][1]) for angle in rotation_x_deg))
            print(all(np.rad2deg(self.vane_mechanical_rotation_limits[1][1]) <= angle <= np.rad2deg(self.vane_mechanical_rotation_limits[1][1]) for angle in rotation_y_deg))
            raise Exception("Requested vane deflection is not permitted:" + f"x-rotation={rotation_x_deg} degrees and y-rotation={rotation_y_deg} degrees.")

        if (self.vane_reference_frame_origin_list == None
                or self.vane_panels_coordinates_list == None
                or self.vane_reference_frame_rotation_matrix_list == None
                or self.number_of_vanes == 0):
            raise Exception("Vane characteristics have not been set by the user.")

        # update the body fixed coordinates of the vanes
        new_vane_coordinates = vane_dynamical_model(rotation_x_deg,
                                                    rotation_y_deg,
                                                    self.number_of_vanes,
                                                    self.vane_reference_frame_origin_list,
                                                    self.vane_panels_coordinates_list,
                                                    self.vane_reference_frame_rotation_matrix_list)
        return new_vane_coordinates

    def vane_system_angles_from_desired_torque(self, acs_object, vane_angles_bounds, desired_torque, previous_vanes_torque,
                                               sunlight_vector_body_frame, initial_vane_angles_guess_rad=[[None]]):
        # vane torque allocation problem
        tap = acs_object.vane_torque_allocation_problem_object
        tap.set_desired_torque(desired_torque, (
            previous_vanes_torque if (previous_vanes_torque[0] != None) else np.array(
                [0] * 3 * acs_object.number_of_vanes)))
        tap.set_attaignable_moment_set_ellipses(sunlight_vector_body_frame)


        prob = pg.problem(tap)
        nl = pg.nlopt('cobyla')
        nl.xtol_rel = self.tol_torque_allocation_problem_x
        nl.ftol_rel = self.tol_torque_allocation_problem_objective
        algo = pg.algorithm(uda=nl)
        if (previous_vanes_torque[0] == None):
            pop = pg.population(prob, size=1, seed=42)
        else:
            current_bounds = tap.get_bounds()

            if (
            all([0 < previous_vanes_torque[bi] - current_bounds[0][bi] < current_bounds[1][bi] - current_bounds[0][bi] for
             bi in range(3 * acs_object.number_of_vanes)])):
                pop = pg.population(prob)
                pop.push_back(x=previous_vanes_torque)
            else:
                pop = pg.population(prob, size=1, seed=42)
        pop.problem.c_tol = self.tol_torque_allocation_problem_constraint
        pop = algo.evolve(pop)  # Evolve population
        x_final = pop.champion_x
        optimal_torque_allocation = x_final.reshape((acs_object.number_of_vanes, 3))

        t0 = time()
        # Determine associated angles
        torque_from_vane_angles_list = np.zeros((acs_object.number_of_vanes, 3))
        vane_angles_rad = np.zeros((acs_object.number_of_vanes, 2))
        for current_vane_id in range(acs_object.number_of_vanes):
            optimised_vane_torque = optimal_torque_allocation[current_vane_id, :]
            requested_vane_torque = optimised_vane_torque / tap.scaling_list[current_vane_id]
            # determine the required vane angles for this
            vaneAngleProblem = tap.vane_angle_problem_objects_list[current_vane_id]

            run_robust_global_optimisation = True
            if (initial_vane_angles_guess_rad[0][0] != None):
                # try a local optimisation algorithm
                vaneAngleProblem.update_vane_angle_determination_algorithm(requested_vane_torque,
                                                                           sunlight_vector_body_frame)
                vane_angle_allocation_results = minimize(vaneAngleProblem.fitness,
                                                         initial_vane_angles_guess_rad[current_vane_id, :],
                                                         method='Nelder-Mead',
                                                         bounds=vane_angles_bounds,
                                                         tol=self.tol_vane_angle_determination)

                vane_torque = vaneAngleProblem.single_vane_torque(
                    [vane_angle_allocation_results.x[0], vane_angle_allocation_results.x[1]])
                orientation_angle_difference_vane_torque = np.rad2deg(np.arccos(
                    np.dot(requested_vane_torque, vane_torque) / (
                                np.linalg.norm(requested_vane_torque) * np.linalg.norm(vane_torque))))
                relative_magnitude_difference_vane_torque = abs(
                    (np.linalg.norm(vane_torque) - np.linalg.norm(requested_vane_torque)) / np.linalg.norm(
                        requested_vane_torque))
                if (relative_magnitude_difference_vane_torque > self.maximum_vane_torque_relative_magnitude_error
                        or orientation_angle_difference_vane_torque > self.maximum_vane_torque_orientation_error
                        or orientation_angle_difference_vane_torque < 0):
                    pass
                    #print("Local optimisation failed, starting DIRECT algorithm")
                    #print(vaneAngleProblem.single_vane_torque(
                    #    [vane_angle_allocation_results.x[0], vane_angle_allocation_results.x[1]]))
                    #print(optimised_vane_torque / tap.scaling_list[current_vane_id])
                else:
                    run_robust_global_optimisation = False

            if (run_robust_global_optimisation):
                vane_angle_allocation_results = vaneAngleAllocationScaling(1, requested_vane_torque,
                                                                           sunlight_vector_body_frame,
                                                                           vaneAngleProblem,
                                                                           vane_angles_bounds,
                                                                           self.tol_vane_angle_determination,
                                                                           self.tol_vane_angle_determination_start_golden_section)[1]
                if (vane_angle_allocation_results.fun > self.tol_vane_angle_determination_start_golden_section):
                    #print("Scaling the desired torque as it is too large")
                    f_golden = lambda t, Td=requested_vane_torque, \
                                      n_s=sunlight_vector_body_frame, vaneAngProb=vaneAngleProblem, \
                                      vane_bounds=vane_angles_bounds, \
                                      tol_global=self.tol_vane_angle_determination, \
                                      tol_golden=self.tol_vane_angle_determination_start_golden_section: \
                        vaneAngleAllocationScaling(t, Td, n_s, vaneAngProb, vane_bounds, tol_global, tol_golden)[0]

                    # golden section search on scaling factor
                    minimizer = golden(f_golden, brack=(0, 1), tol=self.tol_vane_angle_determination_golden_section)
                    vane_angle_allocation_results = \
                    vaneAngleAllocationScaling(minimizer, optimised_vane_torque / tap.scaling_list[current_vane_id],
                                               sunlight_vector_body_frame,
                                               vaneAngleProblem,
                                               vane_angles_bounds,
                                               self.tol_vane_angle_determination,
                                               self.tol_vane_angle_determination_start_golden_section)[1]

            torque_from_vane_angles = vaneAngleProblem.single_vane_torque([vane_angle_allocation_results.x[0],
                                                                           vane_angle_allocation_results.x[1]]) * \
                                      tap.scaling_list[current_vane_id]
            torque_from_vane_angles_list[current_vane_id, :] = torque_from_vane_angles
            vane_angles_rad[current_vane_id, :] = vane_angle_allocation_results.x[:2]

        resulting_torque_from_angles = torque_from_vane_angles_list.sum(axis=0)
        torque_from_vane_angles_direction = resulting_torque_from_angles / np.linalg.norm(resulting_torque_from_angles)
        desired_torque_direction = desired_torque / np.linalg.norm(desired_torque)
        orientation_difference_degrees = np.rad2deg(
            np.arccos(np.dot(torque_from_vane_angles_direction, desired_torque_direction)))

        #print(f"Time vane controller: {time() - t0}")
        #if (abs((orientation_difference_degrees)) > 5):
            #print("---WARNING, the torque direction is not preserved---")
            #print(torque_from_vane_angles_direction)
            #print(desired_torque_direction)
            #print("---------------------------------------------------")
        return vane_angles_rad, torque_from_vane_angles_list, optimal_torque_allocation

    def is_mass_based(self):
        return self.bool_mass_based_controller

    def set_gimball_mass_chateristics(self, mass_of_gimbaled_ballast):
        self.bool_mass_based_controller = True
        self.gimbaled_mass = mass_of_gimbaled_ballast
        return True

    def get_attitude_system_mass(self):
        return self.ACS_mass

    def initialise_actuator_states_dictionary(self):
        self.actuator_states["vane_rotation_x"] = self.actuator_states["vane_rotation_x_default"]
        self.actuator_states["vane_rotation_y"] = self.actuator_states["vane_rotation_y_default"]
        return True
    def get_attitude_control_system_actuators_states(self):
        # convert the actuator states variable to something compatible with the dependent_variables history
        keys_list = ["vane_rotation_x", "vane_rotation_y", "optimal_torque", "vane_torques"]
        dependent_variable_array = np.array([[0]])
        for key in keys_list:
            if key in self.actuator_states.keys():
                dependent_variable_array = np.vstack((dependent_variable_array, self.actuator_states[key]))
        keys_list = ["optimal_torque", "vane_torques"]
        for key in keys_list:
            if key in self.random_variables_dict.keys():
                dependent_variable_array = np.vstack((dependent_variable_array, self.random_variables_dict[key]))
        dependent_variable_array = dependent_variable_array[1:, 0]
        return dependent_variable_array