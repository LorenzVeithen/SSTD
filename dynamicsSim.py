import sys
from generalConstants import tudat_path
sys.path.insert(0, tudat_path)

# Load standard modules
from attitudeControllersClass import *


# Load tudatpy modules
from generalConstants import *
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.data import save2txt
from tudatpy.kernel.interface import spice_interface


class sailCoupledDynamicsProblem:
    def __init__(self,
                 sail_craft,
                 initial_translational_state,
                 initial_rotational_state,
                 simulation_start_epoch,
                 simulation_end_epoch):
        self.sail_craft = sail_craft
        self.simulation_start_epoch = simulation_start_epoch
        self.simulation_end_epoch = simulation_end_epoch

        # Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
        self.global_frame_origin = "Earth"
        self.global_frame_orientation = "J2000"

        # Define bodies that are propagated
        self.bodies_to_create = ["Earth", "Sun"]
        self.bodies_to_propagate = ["ACS3"]
        self.central_bodies = ["Earth"]

        self.occulting_bodies_dict = dict()
        self.occulting_bodies_dict["Sun"] = ["Earth"]

        # initial states
        self.initial_translational_state = initial_translational_state
        self.initial_rotational_state = initial_rotational_state

    def define_simulation_bodies(self):
        body_settings = environment_setup.get_default_body_settings(
            self.bodies_to_create, self.global_frame_origin, self.global_frame_orientation)
        # Add vehicle object to system of bodies
        body_settings.add_empty_settings("ACS3")

        # Setup that the sail_craft body functions are passed
        body_settings.get("ACS3").rigid_body_settings = environment_setup.rigid_body.custom_time_dependent_rigid_body_properties(
            mass_function=self.sail_craft.get_sail_mass,
            center_of_mass_function=self.sail_craft.get_sail_center_of_mass,
            inertia_tensor_function=self.sail_craft.get_sail_inertia_tensor)

        # Written such that it can be extended in the future
        # Compile the panel properties
        panels_settings = []
        panel_functions_dict = {}
        if( self.sail_craft.get_number_of_vanes() == 0):
            j_max = 1
        else:
            j_max = 2
        for j in range(j_max):
            if (j == 0):
                panel_type_str = "Sail"
                number_of_panels = self.sail_craft.get_number_of_wings()
            else:
                panel_type_str = "Vane"
                number_of_panels = self.sail_craft.get_number_of_vanes()

            for i in range(number_of_panels):
                panel_functions_dict[f"{panel_type_str}_panel_{i}_surface_normal"] = lambda current_panel_i=i, current_panel_type=panel_type_str: self.sail_craft.get_ith_panel_surface_normal(panel_id=current_panel_i, panel_type=current_panel_type)
                panel_functions_dict[f"{panel_type_str}_panel_{i}_centroid"] = lambda current_panel_i=i, current_panel_type=panel_type_str: self.sail_craft.get_ith_panel_centroid(panel_id=current_panel_i, panel_type=current_panel_type)
                panel_functions_dict[f"{panel_type_str}_panel_{i}_area"] = lambda current_panel_i=i, current_panel_type=panel_type_str: self.sail_craft.get_ith_panel_area(panel_id=current_panel_i, panel_type=current_panel_type)
                panel_functions_dict[f"{panel_type_str}_panel_{i}_optical_properties_list"] = [(lambda current_panel_i=i, current_panel_type=panel_type_str, idx=k: self.sail_craft.get_ith_panel_optical_properties(panel_id=current_panel_i, panel_type=current_panel_type)[idx]) for k in range(10)]

                panel_geom = environment_setup.vehicle_systems.time_varying_panel_geometry( surface_normal_function=panel_functions_dict[f"{panel_type_str}_panel_{i}_surface_normal"],
                                                                                            position_vector_function=panel_functions_dict[f"{panel_type_str}_panel_{i}_centroid"],
                                                                                            area_function=panel_functions_dict[f"{panel_type_str}_panel_{i}_area"],
                                                                                            frame_orientation="VehicleFixed")

                reflection_law = environment_setup.radiation_pressure.solar_sail_optical_body_panel_time_varying_reflection(panel_functions_dict[f"{panel_type_str}_panel_{i}_optical_properties_list"][0],
                                                                                                                           panel_functions_dict[f"{panel_type_str}_panel_{i}_optical_properties_list"][1],
                                                                                                                           panel_functions_dict[f"{panel_type_str}_panel_{i}_optical_properties_list"][2],
                                                                                                                           panel_functions_dict[f"{panel_type_str}_panel_{i}_optical_properties_list"][3],
                                                                                                                           panel_functions_dict[f"{panel_type_str}_panel_{i}_optical_properties_list"][4],
                                                                                                                           panel_functions_dict[f"{panel_type_str}_panel_{i}_optical_properties_list"][5],
                                                                                                                           panel_functions_dict[f"{panel_type_str}_panel_{i}_optical_properties_list"][6],
                                                                                                                           panel_functions_dict[f"{panel_type_str}_panel_{i}_optical_properties_list"][7],
                                                                                                                           panel_functions_dict[f"{panel_type_str}_panel_{i}_optical_properties_list"][8],
                                                                                                                           panel_functions_dict[f"{panel_type_str}_panel_{i}_optical_properties_list"][9])

                panel = environment_setup.vehicle_systems.body_panel_settings(panel_type_id=panel_type_str,
                                                                               panel_reflection_law=reflection_law,
                                                                               panel_geometry=panel_geom)
                panels_settings.append(panel)

        panelled_body = environment_setup.vehicle_systems.full_panelled_body_settings(
                panel_settings=panels_settings)  # No rotational models for the panels
        body_settings.get('ACS3').vehicle_shape_settings = panelled_body

        vehicle_target_settings = environment_setup.radiation_pressure.panelled_radiation_target(self.occulting_bodies_dict)

        constant_orientation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # create rotation model settings and assign to body settings of "ACS3"
        body_settings.get("ACS3").rotation_model_settings = environment_setup.rotation_model.constant_rotation_model(
            self.global_frame_orientation,
            "VehicleFixed",
            constant_orientation)

        # Create system of bodies (in this case only Earth)
        bodies = environment_setup.create_system_of_bodies(body_settings)
        self.sail_craft.setBodies(bodies)
        return bodies, vehicle_target_settings

    def define_dependent_variables(self, attitude_control_system_object,
                                   keplerian_bool=False):
        # DEPENDENT VARIABLES
        attitude_control_system_object.initialise_actuator_states_dictionary()
        first_attitude_control_dependent_variable_array = (
            attitude_control_system_object.get_attitude_control_system_actuators_states())

        if (keplerian_bool):
            return [propagation_setup.dependent_variable.keplerian_state('ACS3', 'Earth'),
                    propagation_setup.dependent_variable.relative_position("Sun", "ACS3"),
                    propagation_setup.dependent_variable.relative_position("Sun", "Earth"),
                    propagation_setup.dependent_variable.total_torque_norm("ACS3"),
                    propagation_setup.dependent_variable.custom_dependent_variable(
                        attitude_control_system_object.get_attitude_control_system_actuators_states,
                        np.shape(first_attitude_control_dependent_variable_array)[0]),  # Vane deflections
                    ]
        else:
            return [propagation_setup.dependent_variable.keplerian_state('ACS3', 'Earth'),
                    propagation_setup.dependent_variable.received_irradiance_shadow_function("ACS3", "Sun"),
                    propagation_setup.dependent_variable.single_acceleration(
                        propagation_setup.acceleration.radiation_pressure_type, "ACS3", "Sun"),
                    propagation_setup.dependent_variable.single_torque(
                        propagation_setup.torque.radiation_pressure_type, "ACS3", "Sun"),
                    propagation_setup.dependent_variable.relative_position("Sun", "ACS3"),
                    propagation_setup.dependent_variable.relative_position("Sun", "Earth"),
                    propagation_setup.dependent_variable.total_torque_norm("ACS3"),
                    propagation_setup.dependent_variable.custom_dependent_variable(
                        attitude_control_system_object.get_attitude_control_system_actuators_states,
                        np.shape(first_attitude_control_dependent_variable_array)[0]),       # Vane deflections
                    ]

    def define_numerical_environment(self,
                                     integrator_coefficient_set=propagation_setup.integrator.rkf_56,
                                     initial_time_step=1.0,
                                     control_settings=propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(1.0E-12, 1.0E-12),
                                     validation_settings=propagation_setup.integrator.step_size_validation(1E-5, 3E3)):
        # Create termination settings
        termination_settings = propagation_setup.propagator.time_termination(self.simulation_end_epoch,
                                                                             terminate_exactly_on_final_condition=True)
        # Create numerical integrator settings
        integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
            initial_time_step=initial_time_step,
            coefficient_set=integrator_coefficient_set,
            step_size_control_settings=control_settings,
            step_size_validation_settings=validation_settings)

        return termination_settings, integrator_settings

    def define_dynamical_environment(self, bodies, attitude_control_system_object, vehicle_target_settings,
                                     keplerian_bool=False,
                                     theoretical_detumbling_bool=False):
        environment_setup.add_radiation_pressure_target_model(
            bodies, "ACS3", vehicle_target_settings)

        # Translational dynamics settings
        if (keplerian_bool):
            acceleration_settings_acs3 = dict(
                Earth=[propagation_setup.acceleration.point_mass_gravity()],
                Sun=[],
            )
        else:
            acceleration_settings_acs3 = dict(
                Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(2, 0)],    # include J2
                Sun=[propagation_setup.acceleration.radiation_pressure()]
            )

        acceleration_settings = {"ACS3": acceleration_settings_acs3}
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, self.bodies_to_propagate, self.central_bodies)

        # Rotational dynamics settings
        if (keplerian_bool):
            torque_settings_on_sail = dict()

        elif (theoretical_detumbling_bool):
            optimalDetumblingTorque = lambda t, bd=bodies, tau_m=0.2: (
                attitude_control_system_object.computeBodyFrameTorqueForDetumbling(bd, tau_m,
                  desired_rotational_velocity_vector=np.array([0., 0., 0.])))

            torque_settings_on_sail = dict(
                                            Sun=[propagation_setup.torque.radiation_pressure()],
                                            ACS3=[propagation_setup.torque.custom_torque(optimalDetumblingTorque)],
                                           )  # dict(Earth = [propagation_setup.torque.second_degree_gravitational()])
        else:
            torque_settings_on_sail = dict(
                                            Sun=[propagation_setup.torque.radiation_pressure()],
                                           )
        torque_settings = {'ACS3': torque_settings_on_sail}
        torque_models = propagation_setup.create_torque_models(bodies, torque_settings, self.bodies_to_propagate)
        return acceleration_models, torque_models

    def define_propagators(self, integrator_settings, termination_settings, acceleration_models, torque_models,
                           dependent_variables,
                           selected_propagator_=propagation_setup.propagator.gauss_modified_equinoctial,
                           output_frequency_in_seconds=0):
        # Create propagation settings
        selected_propagator = selected_propagator_

        translational_propagator_settings = propagation_setup.propagator.translational(
            self.central_bodies,
            acceleration_models,
            self.bodies_to_propagate,
            self.initial_translational_state,
            self.simulation_start_epoch,
            integrator_settings,
            termination_settings,
            selected_propagator)

        rotational_propagator_settings = propagation_setup.propagator.rotational(torque_models,
                                                                                 self.bodies_to_propagate,
                                                                                 self.initial_rotational_state,
                                                                                 self.simulation_start_epoch,
                                                                                 integrator_settings,
                                                                                 termination_settings)

        # MULTI-TYPE PROPAGATOR
        propagator_list = [translational_propagator_settings, rotational_propagator_settings]
        combined_propagator_settings = propagation_setup.propagator.multitype(propagator_list,
                                                                              integrator_settings,
                                                                              self.simulation_start_epoch,
                                                                              termination_settings,
                                                                              output_variables=dependent_variables)

        combined_propagator_settings.processing_settings.results_save_frequency_in_steps = 0
        combined_propagator_settings.processing_settings.results_save_frequency_in_seconds = output_frequency_in_seconds
        return combined_propagator_settings

    def run_sim(self, bodies, combined_propagator_settings):
        simulator = numerical_simulation.create_dynamics_simulator(bodies, combined_propagator_settings)
        state_history = simulator.state_history
        dependent_variable_history = simulator.dependent_variable_history
        function_evaluation_dict = simulator.cumulative_number_of_function_evaluations
        number_of_function_evaluations = list(function_evaluation_dict.values())[-1]
        propagation_outcome = simulator.integration_completed_successfully
        # Extract the resulting state history and convert it to a ndarray
        states_array = result2array(state_history)
        dependent_variable_array = result2array(dependent_variable_history)
        return state_history, states_array, dependent_variable_history, dependent_variable_array, number_of_function_evaluations,propagation_outcome

    def write_results_to_file(self, state_history, state_history_file_path, dependent_variable_history,
                              dependent_variable_history_file_path):
        save2txt(state_history, state_history_file_path)
        save2txt(dependent_variable_history, dependent_variable_history_file_path)
        return True

