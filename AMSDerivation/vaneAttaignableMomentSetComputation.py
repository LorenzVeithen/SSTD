from vaneControllerMethods import vaneAnglesAllocationProblem
from vaneControllerMethods import generate_AMS_data, plot_and_files_wrapper_generate_AMS_data
from sailCraftClass import sail_craft
from attitudeControllersClass import sail_attitude_control_systems
from constants import *

import matplotlib.pyplot as plt
import matplotlib
from multiprocessing import Process
matplotlib.pyplot.switch_backend('Agg')

if __name__ == "__main__":
    COMPUTE_DATA = True
    GENERATE_AMS_formula = False
    COMPUTE_MULTIPLE_VANES = True
    if (COMPUTE_DATA):
        # Define solar sail - see constants file
        acs_object = sail_attitude_control_systems("vanes", boom_list, sail_I, algorithm_constants)
        acs_object.set_vane_characteristics(vanes_coordinates_list,
                                            vanes_origin_list,
                                            vanes_rotation_matrices_list,
                                            0,
                                            np.array([0, 0, 0]),
                                            0.0045,
                                            vanes_rotational_dof,
                                            "double_ideal_optical_model",
                                            wings_coordinates_list,
                                            vane_mechanical_rotation_limits,
                                            vanes_optical_properties)

        current_optical_model_str = "Ideal_model"
        sail = sail_craft("ACS3",
                          len(wings_coordinates_list),
                          len(vanes_coordinates_list),
                          wings_coordinates_list,
                          vanes_coordinates_list,
                          wings_optical_properties,
                          [np.array([0., 0., 1., 1., 0., 0., 2/3, 2/3, 1., 1.])] * (5),
                          sail_I,
                          sail_mass,
                          sail_mass_without_wings,
                          sail_nominal_CoM,
                          sail_material_areal_density,
                          sail_material_areal_density,
                          acs_object)

        vane_id = 1
        vaneAngleProblem = vaneAnglesAllocationProblem(vane_id,
                                                       ([-np.pi, -np.pi], [np.pi, np.pi]),
                                                       10,
                                                       wings_coordinates_list,
                                                       acs_object,
                                                       include_shadow=True)
        vaneAngleProblem.update_vane_angle_determination_algorithm(np.array([1e-6, 1e-6, 1e-6]), np.array([0, 0, -1]),
                                                                   vane_variable_optical_properties=True,
                                                                   vane_optical_properties_list=vanes_optical_properties)  # and the next time you can put False

        if (COMPUTE_MULTIPLE_VANES):    # and over a large range of sun angles
            sun_angles_num, vane_angles_num = 37, 100
            sun_angle_alpha_list = np.linspace(-180, 180, sun_angles_num)
            sun_angle_beta_list = np.linspace(-180, 180, sun_angles_num)
            alpha_1_range = np.linspace(-np.pi, np.pi, vane_angles_num)
            alpha_2_range = np.linspace(-np.pi, np.pi, vane_angles_num)
            parallel_processes_lists = [[0, 1], [2, 3]]

            for parallel_processes in parallel_processes_lists:
                processes = []
                for i in parallel_processes:
                    pc = Process(target=plot_and_files_wrapper_generate_AMS_data, args=(i,
                                                                                        vaneAngleProblem,
                                                                                        sun_angle_alpha_list,
                                                                                        sun_angle_beta_list,
                                                                                        alpha_1_range,
                                                                                        alpha_2_range,
                                                                                        current_optical_model_str,
                                                                                        True,
                                                                                        True,
                                                                                        2))
                    pc.start()
                    processes.append(pc)

                # Waiting for all threads to complete
                for pc in processes:
                    pc.join()


