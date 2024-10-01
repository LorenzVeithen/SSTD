import numpy as np
import matplotlib.pyplot as plt
from generalConstants import Project_directory, tumbling_data_directory
from pathlib import Path
import os
from scipy.interpolate import CubicSpline
import matplotlib.cm as cm
cmap = plt.get_cmap('jet')

def absolute_evolution_plot_LTT(x_arrays_list, y_arrays_list, xlabel, ylabel, savefig_label, close_fig_bool=True):
    plt.figure()
    for i in range(len(x_arrays_list)):
        plt.plot(x_arrays_list[i], y_arrays_list[i],
                 c="#E69F00",
                 alpha=0.5,
                 linewidth=1)
    plt.grid(True)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # complete hard-coded, do not reproduce at home
    if ('peri' in savefig_label or 'apo' in savefig_label):
        state_history_array_analytical = np.loadtxt(
            "/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/PropagationData/state_history_omega_x_0.0_omega_y_0.0_omega_z_0.0_reduced_area_LTT.dat")
        dependent_variable_history_array_analytical = np.loadtxt(
            "/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/PropagationData/dependent_variable_history_omega_x_0.0_omega_y_0.0_omega_z_0.0_reduced_area_LTT.dat")

        # Extract dependent variables
        t_hours_analytical = (dependent_variable_history_array_analytical[:, 0] -
                              dependent_variable_history_array_analytical[0, 0]) / (3600 * 24)
        keplerian_state_analytical = dependent_variable_history_array_analytical[:, 1:7]
        pericenter_analytical = keplerian_state_analytical[:, 0] * (1 - keplerian_state_analytical[:, 1]) / 1000
        apocenter_analytical = keplerian_state_analytical[:, 0] * (1 + keplerian_state_analytical[:, 1]) / 1000

        if ('peri' in savefig_label):
            plt.plot(t_hours_analytical, pericenter_analytical, label='Simplified model', linestyle='--', color='k')
        else:
            plt.plot(t_hours_analytical, apocenter_analytical, label='Simplified model', linestyle='--', color='k')
        plt.plot([], [],  c="#E69F00",
                 label=f'Numerical propagations',
                 alpha=0.5,
                 linewidth=1)
        plt.legend()
    plt.savefig(savefig_label, bbox_inches='tight', dpi=300)
    if (close_fig_bool): plt.close()

def relative_evolution_plot_LTT(x_arrays_list, y_arrays_list,
                                xlabel, ylabel,
                                ref_x_array, ref_y_array,
                                colour_bar_values_list, colour_bar_label,
                                savefig_label, close_fig_bool=True):

    c_norm_omega_all = plt.Normalize(np.min(colour_bar_values_list), np.max(colour_bar_values_list))
    c_omega_all = cmap(c_norm_omega_all(colour_bar_values_list))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(x_arrays_list)):
        current_spline = CubicSpline(ref_x_array, ref_y_array)
        diff = y_arrays_list[i] - current_spline(x_arrays_list[i])

        plt.plot(x_arrays_list[i], diff,
                 c=c_omega_all[i],
                 label=f'{i}',
                 linewidth=1)

    plt.grid(True)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    cbar = fig.colorbar(cm.ScalarMappable(norm=c_norm_omega_all, cmap=cmap), ax=ax)
    cbar.set_label(colour_bar_label, rotation=270, labelpad=13)
    plt.savefig(savefig_label, bbox_inches='tight', dpi=300)
    if (close_fig_bool): plt.close()

analysis_save_data_dir = tumbling_data_directory
BIG_PLOTS = True
Maps_2D = True
COMPARISON_PLOTS = True

thinning_factor = 1
keplerian_keys = ['sma', 'ecc', "inc", "aop", "raan", "tranom"]
omega_components_id = ['x', 'y', 'z']
time_label = r"$t$ [days]"

# prepare labels and scaling factors
labels_list = [r"$a$ [km]",
               r"$e$ [-]",
               r"$i$ [deg]",
               r"$\omega$ [deg]",
               r"$\Omega$ [deg]",
               r"$\theta$ [deg]",
               r'$r_a$ [km]',
               r'$r_p$ [km]']

labels_change_list = [r"$\Delta a$ [km]",
                r"$\Delta e$ [-]",
                r"$\Delta i$ [deg]",
                r"$\Delta \omega$, [deg]",
                r"$\Delta \Omega$ [deg]",
                r"$\Delta \theta$ [deg]",
                r'$\Delta r_a$ [km]',
                r'$\Delta r_p$ [km]']

labels_change_final_list = [r"$\Delta a_{\text{fin}}$ [km]",
                r"$\Delta e_{\text{fin}}$ [-]",
                r"$\Delta i_{\text{fin}}$ [deg]",
                r"$\Delta \omega_{\text{fin}}$, [deg]",
                r"$\Delta \Omega_{\text{fin}}$ [deg]",
                r"$\Delta \theta_{\text{fin}}$ [deg]",
                r'$\Delta r_{a, \text{fin}}$ [km]',
                r'$\Delta r_{p, \text{fin}}$ [km]']

labels_change_percent_list = [r"$\Delta a$ [%]",
                r"$\Delta e$ [%]",
                r"$\Delta i$ [%]",
                r"$\Delta \omega$, [%]",
                r"$\Delta \Omega$ [%]",
                r"$\Delta \theta$ [%]",
                r'$\Delta r_a$ [%]',
                r'$\Delta r_p$ [%]']

colour_bar_label = [r'$||\vec{\omega}||$ [deg/s]',
                    r'$\omega_{x}$ [deg/s]',
                    r'$\omega_{y}$ [deg/s]',
                    r'$\omega_{z}$ [deg/s]',
                    r'$||\vec{\omega_{xy}}||$ [deg/s]']
save_fig_cb_label = ["magnitude", "x", "y", "z", "xy_magnitude"]

multiplying_factors = [1./1000, 1., 180./np.pi, 180./np.pi, 180./np.pi, 180./np.pi, 1./1000, 1./1000, 1./1000, 1./1000]
variables_list = ['sma', 'ecc', "inc", "aop", "raan", "tranom", "apo", "peri"]  # 'r', 'v'

def get_dataset_data(current_dataset):
    analysis_data_dir = analysis_save_data_dir + current_dataset
    states_history_dir = analysis_data_dir + "/states_history"
    dependent_variable_history_dir = analysis_data_dir + "/dependent_variable_history"

    # state history files
    p = Path(states_history_dir)
    state_history_files = [x for x in p.iterdir() if (not x.is_dir())]#[:10]

    if ('Sun_Pointing' in current_dataset):
        pass
    else:
        # Add sun_pointing one
        state_history_files += [analysis_data_dir + '/sun_pointing_state_history_omega_x_0.0_omega_y_0.0_omega_z_0.0.dat']

    # add the path with zero rotational velocity in case it was not in the original bundle
    omega_is_zero_vector_data = Path(f'{analysis_data_dir}/states_history/state_history_omega_x_0.0_omega_y_0.0_omega_z_0.0.dat')
    if (((omega_is_zero_vector_data in state_history_files) == False) and (omega_is_zero_vector_data.exists())):
        state_history_files += [omega_is_zero_vector_data]


    LTT_results_dict = {}

    # First handle the Keplerian orbit case for comparison
    keplerian_dependent_variable_history_array = np.loadtxt(f'{analysis_data_dir}/keplerian_orbit_dependent_variable_history.dat')
    keplerian_state_history_array = np.loadtxt(f'{analysis_data_dir}/keplerian_orbit_state_history.dat')

    keplerian_state_history_dict, keplerian_dependent_variables_dict = {}, {}
    for j, c_time in enumerate(keplerian_state_history_array[:, 0]):
        keplerian_state_history_dict[c_time] = keplerian_state_history_array[j, 1:]
        keplerian_dependent_variables_dict[c_time] = keplerian_dependent_variable_history_array[j, 1:]


    LTT_results_dict["keplerian" + "_time_array"] = (keplerian_state_history_array[:, 0] - keplerian_state_history_array[0, 0]) / (24 * 3600)
    for counter in range(6):
        LTT_results_dict["keplerian" + f"_{keplerian_keys[counter]}_array"] = keplerian_dependent_variable_history_array[:, counter+1]
    LTT_results_dict["keplerian" + "_apo_array"] = keplerian_dependent_variable_history_array[:, 1] * (1 + keplerian_dependent_variable_history_array[:, 2])
    LTT_results_dict["keplerian" + "_peri_array"] = keplerian_dependent_variable_history_array[:, 1] * (1 - keplerian_dependent_variable_history_array[:, 2])
    LTT_results_dict[f"keplerian" + "_initial_omega_vector_array"] = np.array([0, 0, 0])
    LTT_results_dict["keplerian" + "_r_array"] = np.sqrt(keplerian_state_history_array[:, 1]**2 + keplerian_state_history_array[:, 2]**2 + keplerian_state_history_array[:, 3]**2)
    LTT_results_dict["keplerian" + "_v_array"] = np.sqrt(keplerian_state_history_array[:, 4]**2 + keplerian_state_history_array[:, 5]**2 + keplerian_state_history_array[:, 6]**2)

    # Initialise some dictionary entries
    LTT_results_dict["all" + "_time_array"] = []
    for counter in range(6):
        LTT_results_dict["all" + f"_{keplerian_keys[counter]}_array"] = []
    LTT_results_dict["all" + "_apo_array"] = []
    LTT_results_dict["all" + "_peri_array"] = []
    LTT_results_dict["all" + "_initial_omega_vector_array"] = []
    LTT_results_dict["all" + "_omega_x_array"] = []
    LTT_results_dict["all" + "_omega_y_array"] = []
    LTT_results_dict["all" + "_omega_z_array"] = []
    LTT_results_dict["all" + "_r_array"] = []
    LTT_results_dict["all" + "_v_array"] = []
    LTT_results_dict["all" + "_a_x_array"] = []
    LTT_results_dict["all" + "_a_y_array"] = []
    LTT_results_dict["all" + "_a_z_array"] = []

    # diffs
    for counter in range(6):
        LTT_results_dict["all" + f"_{keplerian_keys[counter]}_diff_array"] = []
    LTT_results_dict["all" + "_apo_diff_array"] = []
    LTT_results_dict["all" + "_peri_diff_array"] = []
    LTT_results_dict["all" + "_r_diff_array"] = []
    LTT_results_dict["all" + "_v_diff_array"] = []

    for current_state_history_path in state_history_files:
        if (str(current_state_history_path).split('/')[-1][0] == '.'):
            continue
        print(str(current_state_history_path).split('/')[-1])

        # get the initial rotational velocity vector of the propagation
        l_str = str(current_state_history_path)[:-4].split('_')
        omega_z_rph = float(l_str[-1])
        omega_y_rph = float(l_str[-4])
        omega_x_rph = float(l_str[-7])

        current_state_history_array = np.loadtxt(current_state_history_path)

        current_state_history_array = current_state_history_array[::thinning_factor]
        if ('sun_pointing' in str(current_state_history_path) and (not ('Sun_Pointing' in current_dataset))):
            current_dependent_variable_history_path = analysis_data_dir + '/sun_pointing_dependent_variable_history_omega_x_0.0_omega_y_0.0_omega_z_0.0.dat'
            sun_pointing_bool = True
        else:
            current_dependent_variable_history_path = dependent_variable_history_dir + f"/dependent_variable_history_omega_x_{omega_x_rph}_omega_y_{omega_y_rph}_omega_z_{omega_z_rph}.dat"
            sun_pointing_bool = False

        current_dependent_variable_history_array = np.loadtxt(current_dependent_variable_history_path)
        current_dependent_variable_history_array = current_dependent_variable_history_array[::thinning_factor]

        current_state_history_dict, current_dependent_variables_dict = {}, {}
        for j, c_time in enumerate(current_state_history_array[:, 0]):
            current_state_history_dict[c_time] = current_state_history_array[j, 1:]
            current_dependent_variables_dict[c_time] = current_dependent_variable_history_array[j, 1:]

        # Difference between the current propagation and Keplerian
        #diff_to_kep_states_array = result2array(
        #                            compare_results(keplerian_state_history_dict, current_state_history_dict, current_state_history_array[:, 0]))

        #diff_to_kep_dep_var_array = result2array(
        #    compare_results(keplerian_state_history_dict, current_state_history_dict, current_state_history_array[:, 0]))


        omega_vector_deg_per_sec = np.array([omega_x_rph, omega_y_rph, omega_z_rph])/10
        omega_vector_rph = np.array([omega_x_rph, omega_y_rph, omega_z_rph])

        # extract Keplerian elements history
        # 1: Semi-major Axis. 2: Eccentricity. 3: Inclination. 4: Argument of Periapsis.
        # 5. Right Ascension of the Ascending Node. 6: True Anomaly.
        current_time_array = (current_dependent_variable_history_array[:, 0]-current_dependent_variable_history_array[0, 0])/(24*3600)
        current_kep_array = current_dependent_variable_history_array[:, 1:7]

        # apogee and perigee
        current_apo_array = current_kep_array[:, 0] * (1 + current_kep_array[:, 1])
        current_peri_array = current_kep_array[:, 0] * (1 - current_kep_array[:, 1])

        # rotational velocity history
        omega_x_array_deg_s = np.rad2deg(current_state_history_array[:, 11])
        omega_y_array_deg_s = np.rad2deg(current_state_history_array[:, 12])
        omega_z_array_deg_s = np.rad2deg(current_state_history_array[:, 13])

        # Cartesian position
        current_r_array = np.sqrt(current_state_history_array[:, 1]**2 + current_state_history_array[:, 2]**2 + current_state_history_array[:, 3]**2)
        current_v_array = np.sqrt(current_state_history_array[:, 4] ** 2 + current_state_history_array[:,
                                                                   5] ** 2 + current_state_history_array[:, 6] ** 2)

        # SRP acceleration
        spacecraft_srp_acceleration_vector = current_dependent_variable_history_array[:, 8:11]
        current_a_x = spacecraft_srp_acceleration_vector[:, 0]
        current_a_y = spacecraft_srp_acceleration_vector[:, 1]
        current_a_z = spacecraft_srp_acceleration_vector[:, 2]

        # Solar irradiance
        received_irradiance_shadow_function = current_dependent_variable_history_array[:, 7]
        received_irradiance_shadow_function[
            received_irradiance_shadow_function < 1] = 0  # remove any part where it is not
        difference_irradiance = np.diff(received_irradiance_shadow_function)
        t_start_eclipse = current_time_array[np.where(np.diff(received_irradiance_shadow_function) == -1)[0]]
        t_end_eclipse = current_time_array[np.where(np.diff(received_irradiance_shadow_function) == 1)[0]]

        if (len(t_start_eclipse) > len(t_end_eclipse)):
            t_start_eclipse = t_start_eclipse[:-1]
        elif (len(t_end_eclipse) > len(t_start_eclipse)):
            t_end_eclipse = np.append(t_end_eclipse, [current_time_array[-1]])

        eclipse_fraction = np.sum(t_end_eclipse - t_start_eclipse) / (current_time_array[-1] - current_time_array[0])
        print(eclipse_fraction)

        # Difference in Cartesian elements
        #current_r_diff_array = np.sqrt(diff_to_kep_states_array[:, 1]**2 + diff_to_kep_states_array[:, 2]**2 + diff_to_kep_states_array[:, 3]**2)
        #current_v_diff_array = np.sqrt(diff_to_kep_states_array[:, 4] ** 2 + diff_to_kep_states_array[:,
        #                                                          5] ** 2 + diff_to_kep_states_array[:, 6] ** 2)

        # Difference in Keplerian elements
        current_kep_diff_array = current_kep_array - current_kep_array[0, :]

        # Difference in apocenter and pericenter
        current_apo_diff_array = current_apo_array - current_kep_array[0, 0] * (1 + current_kep_array[0, 1])
        current_peri_diff_array = current_peri_array - current_kep_array[0, 0] * (1 - current_kep_array[0, 1])

        if (omega_x_rph==0 and omega_y_rph==0 and omega_z_rph==0):
            if (sun_pointing_bool):
                key = 'sun_pointing'
            else:
                key = 'default'
            LTT_results_dict[key + "_time_array"] = current_time_array
            for counter in range(6):
                LTT_results_dict[key + f"_{keplerian_keys[counter]}_array"] = current_kep_array[:, counter]
            LTT_results_dict[key + "_apo_array"] = current_apo_array
            LTT_results_dict[key + "_peri_array"] = current_peri_array
            LTT_results_dict[key + "_omega_x_array"] = omega_x_array_deg_s
            LTT_results_dict[key + "_omega_y_array"] = omega_y_array_deg_s
            LTT_results_dict[key + "_omega_z_array"] = omega_z_array_deg_s
            LTT_results_dict[key + "_r_array"] = current_r_array
            LTT_results_dict[key + "_v_array"] = current_v_array
            LTT_results_dict[key + "_a_x_array"] = current_a_x
            LTT_results_dict[key + "_a_y_array"] = current_a_y
            LTT_results_dict[key + "_a_z_array"] = current_a_z

            # diffs
            for counter in range(6):
                LTT_results_dict[key + f"_{keplerian_keys[counter]}_diff_array"] = current_kep_diff_array[:, counter]
            LTT_results_dict[key + "_apo_diff__array"] = current_apo_diff_array
            LTT_results_dict[key + "_peri_diff_array"] = current_peri_diff_array
            #LTT_results_dict[key + "_r_diff_array"] = current_r_diff_array
            #LTT_results_dict[key + "_v_diff_array"] = current_v_diff_array

        else:
            LTT_results_dict["all" + "_time_array"].append(current_time_array)
            for i in range(6):
                LTT_results_dict["all" + f"_{keplerian_keys[i]}_array"].append(current_kep_array[:, i])
            LTT_results_dict["all" + "_apo_array"].append(current_apo_array)
            LTT_results_dict["all" + "_peri_array"].append(current_peri_array)
            LTT_results_dict[f"all" + "_initial_omega_vector_array"].append(omega_vector_deg_per_sec)
            LTT_results_dict[f"all" + "_omega_x_array"].append(omega_x_array_deg_s)
            LTT_results_dict[f"all" + "_omega_y_array"].append(omega_y_array_deg_s)
            LTT_results_dict[f"all" + "_omega_z_array"].append(omega_z_array_deg_s)
            LTT_results_dict[f"all" + "_r_array"].append(current_r_array)
            LTT_results_dict[f"all" + "_v_array"].append(current_v_array)
            LTT_results_dict[f"all" + "_a_x_array"].append(current_a_x)
            LTT_results_dict[f"all" + "_a_y_array"].append(current_a_y)
            LTT_results_dict[f"all" + "_a_z_array"].append(current_a_z)

            # diffs
            for i in range(6):
                LTT_results_dict["all" + f"_{keplerian_keys[i]}_diff_array"].append(current_kep_diff_array[:, i])
            LTT_results_dict["all" + "_apo_diff_array"].append(current_apo_diff_array)
            LTT_results_dict["all" + "_peri_diff_array"].append(current_peri_diff_array)
            #LTT_results_dict[f"all" + "_r_diff_array"].append(current_r_diff_array)
            #LTT_results_dict[f"all" + "_v_diff_array"].append(current_v_diff_array)

    if ('Sun_Pointing' in current_dataset):
        LTT_results_dict['sun_pointing' + "_time_array"] = LTT_results_dict['default' + "_time_array"]
        for counter in range(6):
            LTT_results_dict['sun_pointing' + f"_{keplerian_keys[counter]}_array"] = LTT_results_dict['default' + f"_{keplerian_keys[counter]}_array"]
        LTT_results_dict['sun_pointing' + "_apo_array"] = LTT_results_dict['default' + "_apo_array"]
        LTT_results_dict['sun_pointing' + "_peri_array"] = LTT_results_dict['default' + "_peri_array"]
        LTT_results_dict['sun_pointing' + "_omega_x_array"] = LTT_results_dict['default' + "_omega_x_array"]
        LTT_results_dict['sun_pointing' + "_omega_y_array"] = LTT_results_dict['default' + "_omega_y_array"]
        LTT_results_dict['sun_pointing' + "_omega_z_array"] = LTT_results_dict['default' + "_omega_z_array"]
        #LTT_results_dict['sun_pointing' + "_r_array"] = LTT_results_dict['default' + "_r_array"]
        #LTT_results_dict['sun_pointing' + "_v_array"] = LTT_results_dict['default' + "_v_array"]
        LTT_results_dict['sun_pointing' + "_a_x_array"] = LTT_results_dict['default' + "_a_x_array"]
        LTT_results_dict['sun_pointing' + "_a_y_array"] = LTT_results_dict['default' + "_a_y_array"]
        LTT_results_dict['sun_pointing' + "_a_z_array"] = LTT_results_dict['default' + "_a_z_array"]
    processed_array_dict = LTT_results_dict
    # scale variables as desired
    for i in range(len(variables_list)):
        processed_array_dict["default" + f"_{variables_list[i]}_array"] = (
                np.array(processed_array_dict["default" + f"_{variables_list[i]}_array"]) * multiplying_factors[i])
        processed_array_dict["sun_pointing" + f"_{variables_list[i]}_array"] = (
                np.array(processed_array_dict["sun_pointing" + f"_{variables_list[i]}_array"]) * multiplying_factors[i])
        processed_array_dict["keplerian" + f"_{variables_list[i]}_array"] = (
                np.array(processed_array_dict["keplerian" + f"_{variables_list[i]}_array"]) * multiplying_factors[i])
        for j in range(len(processed_array_dict["all" + f"_{variables_list[i]}_array"])):
            processed_array_dict["all" + f"_{variables_list[i]}_array"][j] = (
                    processed_array_dict["all" + f"_{variables_list[i]}_array"][j] * multiplying_factors[i])

        for j in range(len(processed_array_dict["all" + f"_{variables_list[i]}_diff_array"])):
            processed_array_dict["all" + f"_{variables_list[i]}_diff_array"][j] = (
                    processed_array_dict["all" + f"_{variables_list[i]}_diff_array"][j] * multiplying_factors[i])


    # omega lists for colour bars
    initial_omega_norm_list = []
    initial_omega_x_list, initial_omega_y_list, initial_omega_z_list = [], [], []
    all_omega_xy_norm_list = []
    max_position_difference = []
    for l in range(len(LTT_results_dict[f"all" + "_initial_omega_vector_array"])):
        initial_omega_norm_list.append(np.linalg.norm(LTT_results_dict[f"all" + "_initial_omega_vector_array"][l]))
        initial_omega_x_list.append(LTT_results_dict[f"all" + "_initial_omega_vector_array"][l][0])
        initial_omega_y_list.append(LTT_results_dict[f"all" + "_initial_omega_vector_array"][l][1])
        initial_omega_z_list.append(LTT_results_dict[f"all" + "_initial_omega_vector_array"][l][2])
        all_omega_xy_norm_list.append(np.sqrt(LTT_results_dict[f"all" + "_initial_omega_vector_array"][l][0]**2
                                              + LTT_results_dict[f"all" + "_initial_omega_vector_array"][l][1]**2))

    all_data_array = np.empty((len(initial_omega_norm_list), 32), dtype=object)
    all_data_array[:, 0] = initial_omega_norm_list
    all_data_array[:, 1] = initial_omega_x_list
    all_data_array[:, 2] = initial_omega_y_list
    all_data_array[:, 3] = initial_omega_z_list
    all_data_array[:, 4] = all_omega_xy_norm_list
    all_data_array[:, 5] = processed_array_dict["all" + "_omega_x_array"]
    all_data_array[:, 6] = processed_array_dict["all" + "_omega_y_array"]
    all_data_array[:, 7] = processed_array_dict["all" + "_omega_z_array"]
    all_data_array[:, 8] = processed_array_dict["all" + "_time_array"]
    #all_data_array[:, 9] = processed_array_dict["all" + "_r_array"]
    #all_data_array[:, 10] = processed_array_dict["all" + "_v_array"]
    all_data_array[:, 11] = processed_array_dict["all" + f"_sma_array"]
    all_data_array[:, 12] = processed_array_dict["all" + f"_ecc_array"]
    all_data_array[:, 13] = processed_array_dict["all" + f"_inc_array"]
    all_data_array[:, 14] = processed_array_dict["all" + f"_aop_array"]
    all_data_array[:, 15] = processed_array_dict["all" + f"_raan_array"]
    all_data_array[:, 16] = processed_array_dict["all" + f"_tranom_diff_array"]
    all_data_array[:, 17] = processed_array_dict["all" + "_apo_array"]
    all_data_array[:, 18] = processed_array_dict["all" + "_peri_array"]

    #all_data_array[:, 19] = processed_array_dict["all" + "_r_diff_array"]
    #all_data_array[:, 20] = processed_array_dict["all" + "_v_diff_array"]
    all_data_array[:, 21] = processed_array_dict["all" + f"_sma_diff_array"]
    all_data_array[:, 22] = processed_array_dict["all" + f"_ecc_diff_array"]
    all_data_array[:, 23] = processed_array_dict["all" + f"_inc_diff_array"]
    all_data_array[:, 24] = processed_array_dict["all" + f"_aop_diff_array"]
    all_data_array[:, 25] = processed_array_dict["all" + f"_raan_diff_array"]
    all_data_array[:, 26] = processed_array_dict["all" + f"_tranom_diff_array"]
    all_data_array[:, 27] = processed_array_dict["all" + "_apo_diff_array"]
    all_data_array[:, 28] = processed_array_dict["all" + "_peri_diff_array"]

    all_data_array[:, 29] = processed_array_dict["all" + "_a_x_array"]
    all_data_array[:, 30] = processed_array_dict["all" + "_a_y_array"]
    all_data_array[:, 31] = processed_array_dict["all" + "_a_z_array"]


    return processed_array_dict, all_data_array

color_list = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
my_cmap = plt.get_cmap('plasma')
cmap = plt.colormaps["plasma"]
plot_mode = 5
percentage_bool = False
if (plot_mode == 1): # Just a single dataset
    datasets_list = [f'/Sun_Pointing/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_single_ideal_opt_model_shadow_False',
                     f'/Sun_Pointing/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_False',
                     f'/Sun_Pointing/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_ACS3_opt_model_shadow_False'
                     ]
    labels_optical_list = ['SI-SRP', 'DI-SRP', 'O-SRP']
    Comparison_label = 'optical_sun_pointing'
elif (plot_mode == 2):
    datasets_list = [f'/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_False',
                 f'/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_ACS3_opt_model_shadow_False',
                 f'/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_single_ideal_opt_model_shadow_False',]
    labels_optical_list = ['DI-SRP', 'SI-SRP', 'O-SRP']
    Comparison_label = 'optical'
elif (plot_mode == 3):
    datasets_list = [f'/Sun_Pointing/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_False',
                 f'/Sun_Pointing/LEO_ecc_0.0_inc_45.0/NoAsymetry_data_double_ideal_opt_model_shadow_False',
                 f'/Sun_Pointing/LEO_ecc_0.0_inc_0.0/NoAsymetry_data_double_ideal_opt_model_shadow_False',]
    labels_optical_list = [r'$i=98$°', r'$i=45$°', r'$i=0$°']
    Comparison_label = 'inclination'
elif (plot_mode == 4):
    datasets_list = [f'/Sun_Pointing/LEO_ecc_0.0_inc_0.0/NoAsymetry_data_double_ideal_opt_model_shadow_False',
                 f'/Sun_Pointing/MEO_ecc_0.0_inc_0.0/NoAsymetry_data_double_ideal_opt_model_shadow_False',
                 f'/Sun_Pointing/GEO_ecc_0.0_inc_0.0/NoAsymetry_data_double_ideal_opt_model_shadow_False',]
    #percentage_bool = True
    labels_optical_list = [r'LEO', r'MEO', r'GEO']
    Comparison_label = 'orbital_regime'
elif (plot_mode == 5):
    datasets_list = [f'/Sun_Pointing/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_False']
    labels_optical_list = ['DI-SRP']
    Comparison_label = 'None'

num_fine = 100000
t_fine = np.linspace(0, 29, num_fine)


for data_id, dataset in enumerate(datasets_list):
    current_processed_dict, current_all_data_array = get_dataset_data(dataset)
    save_plot_dir = Project_directory + f'/0_FinalPlots/LTT' + dataset
    if (not os.path.exists(save_plot_dir)):
        os.makedirs(save_plot_dir)

    current_last_comp_array = np.empty((len(current_all_data_array[:, 0]), 10), dtype='float64')
    current_last_comp_array[:, 0] = current_all_data_array[:, 1]
    current_last_comp_array[:, 1] = current_all_data_array[:, 2]
    for i in range(len(current_all_data_array[:, 0])):
        current_last_comp_array[i, 2] = current_all_data_array[i, 21][-1]
        current_last_comp_array[i, 3] = current_all_data_array[i, 22][-1]
        current_last_comp_array[i, 4] = current_all_data_array[i, 23][-1]
        current_last_comp_array[i, 5] = current_all_data_array[i, 24][-1]
        current_last_comp_array[i, 6] = current_all_data_array[i, 25][-1]
        current_last_comp_array[i, 7] = current_all_data_array[i, 26][-1]
        current_last_comp_array[i, 8] = current_all_data_array[i, 27][-1]
        current_last_comp_array[i, 9] = current_all_data_array[i, 28][-1]

    for i in range(8):
        current_save_plot_dir = save_plot_dir + '/' + variables_list[i]
        if (not os.path.exists(current_save_plot_dir)):
            os.mkdir(current_save_plot_dir)

        # The (0, 0, 0)
        plt.figure()
        plt.plot(current_processed_dict['sun_pointing_time_array'], current_processed_dict[f'sun_pointing_{variables_list[i]}_array'])
        plt.xlabel(r'$t$ [days]', fontsize=14)
        plt.ylabel(labels_list[i], fontsize=14)
        plt.grid(True)
        plt.savefig(current_save_plot_dir + '/sun_pointing_0_0_0_' + variables_list[i] + '.png')
        plt.close()


        # Only single axis
        nonzero_omega_x_data = current_all_data_array[np.where((current_all_data_array[:, 1] != 0)
                                                             & (np.all(current_all_data_array[:, 2:4] == 0, axis=1)))[0],
                               :]
        nonzero_omega_y_data = current_all_data_array[np.where((current_all_data_array[:, 2] != 0)
                                                             & (current_all_data_array[:, 1] == 0)
                                                             & (current_all_data_array[:, 3] == 0))[0], :]
        plt.figure()
        plt.plot(current_processed_dict['sun_pointing_time_array'],
                 current_processed_dict[f'sun_pointing_{variables_list[i]}_array'], label=r'$||\vec{\omega}_{\mathcal{B}}|| = 0$', linestyle='-', color=color_list[0])
        for p_id in range(len(nonzero_omega_x_data[:, 0])):
            plt.plot(nonzero_omega_x_data[p_id, 8], nonzero_omega_x_data[p_id, 11 + i], alpha=0.2, linestyle='-', color=color_list[1])
        plt.plot([], [], alpha=0.2, linestyle='-', color=color_list[1], label=r'$||\vec{\omega}_{\mathcal{B}}|| = \omega_{x, \mathcal{B}}$')
        for p_id in range(len(nonzero_omega_y_data[:, 0])):
            plt.plot(nonzero_omega_y_data[p_id, 8], nonzero_omega_y_data[p_id, 11 + i], alpha=0.2, linestyle='-', color=color_list[2])
        plt.plot([], [], alpha=0.2, linestyle='-', color=color_list[2], label=r'$||\vec{\omega}_{\mathcal{B}}|| = \omega_{y, \mathcal{B}}$')
        plt.legend()
        plt.xlabel(r'$t$ [days]', fontsize=14)
        plt.ylabel(labels_list[i], fontsize=14)
        plt.grid(True)
        plt.savefig(current_save_plot_dir + '/' + variables_list[i] + '_single_axis_history_comparison_to_initial.png',
                    dpi=600,
                    bbox_inches='tight')

        plt.close()
        print(len(nonzero_omega_x_data[:, 0]))
        print(len(nonzero_omega_y_data[:, 0]))

        if (Maps_2D):   # of the current dataset, not of a comparison
            print('2D Maps and Density maps')
            f, ax = plt.subplots()
            ax.set_xlabel(r'$\omega_{x, \mathcal{B}}$ [deg/s]', fontsize=14)
            ax.set_ylabel(r'$\omega_{y, \mathcal{B}}$ [deg/s]', fontsize=14)
            tpc = ax.tripcolor(current_last_comp_array[:, 0], current_last_comp_array[:, 1], current_last_comp_array[:, i + 2],
                               shading='flat', cmap=my_cmap)
            cbar = f.colorbar(tpc)
            cbar.set_label(labels_change_final_list[i], rotation=270, labelpad=13, fontsize=14)
            plt.savefig(current_save_plot_dir + '/' + variables_list[i] + '_2D.png',
                        dpi=600, bbox_inches='tight')
            plt.close()

            fig, ax = plt.subplots()
            y_fine = np.concatenate([np.interp(t_fine, current_all_data_array[:, 8][j], current_all_data_array[:, 21 + i][j]) for j in
                                     range(len(current_all_data_array[:, 0]))])
            x_fine = np.broadcast_to(t_fine, (len(current_all_data_array[:, 0]), num_fine)).ravel()
            cmap = cmap.with_extremes(bad=cmap(0))
            h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[2000, 250])
            pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap,
                                vmax=1.5e2, rasterized=True)
            fig.colorbar(pcm, ax=ax, label="# points", pad=0)
            ax.set_xlabel(r'$t$ [days]')
            ax.set_ylabel(labels_change_list[i])
            plt.savefig(current_save_plot_dir + '/' + variables_list[i] + '_point_density.png', dpi=300, bbox_inches='tight')
            plt.close()

        if (BIG_PLOTS):
            print('Big plots')
            # Keplerian absolute variations

            save_path = current_save_plot_dir + '/' + variables_list[i] + '.png'
            absolute_evolution_plot_LTT(current_processed_dict[f"all" + "_time_array"],
                                        current_processed_dict["all" + f"_{variables_list[i]}_array"],
                                        time_label,
                                        labels_list[i],
                                        save_path,
                                        close_fig_bool=True)

            # Keplerian change wrt to [0, 0, 0] and to Keplerian orbit with omega magnitude
            for cb_id, cb_value in enumerate(
                    [list(current_all_data_array[:, 0]),
                     list(current_all_data_array[:, 1]),
                     list(current_all_data_array[:, 2]),
                     list(current_all_data_array[:, 3]),
                     list(current_all_data_array[:, 4])]):
                for ref_key in ["default", "keplerian"]:
                    if (ref_key == "default"):  # and omega_is_zero_vector_data.exists()==False
                        continue
                    save_path = current_save_plot_dir + '/' + variables_list[
                        i] + f'_change_wrt_{ref_key}_cbar_omega_{save_fig_cb_label[cb_id]}.png'
                    relative_evolution_plot_LTT(current_processed_dict[f"all" + "_time_array"],
                                                current_processed_dict["all" + f"_{variables_list[i]}_array"],
                                                time_label,
                                                labels_change_list[i],
                                                current_processed_dict[ref_key + "_time_array"],
                                                current_processed_dict[ref_key + f"_{variables_list[i]}_array"],
                                                cb_value,
                                                colour_bar_label[cb_id],
                                                save_path,
                                                close_fig_bool=True)

        if (COMPARISON_PLOTS):
            print('Comparison plots')
            interpolated_history_array = np.array([np.interp(t_fine, current_all_data_array[:, 8][j], current_all_data_array[:, 21+i][j]) for j in range(len(current_all_data_array[:, 0]))])
            min_list, max_list, median_list = [], [], []
            sigma_1_plus_list = []
            sigma_1_minus_list = []
            for t_id in range(len(interpolated_history_array[0, :])):
                min_list.append(np.min(interpolated_history_array[:, t_id]))
                max_list.append(np.max(interpolated_history_array[:, t_id]))
                median_list.append(np.median(interpolated_history_array[:, t_id]))
                sigma_1_plus_list.append(np.percentile(interpolated_history_array[:, t_id], 100-17.5))
                sigma_1_minus_list.append(np.percentile(interpolated_history_array[:, t_id], 17.5))
            min_list, max_list, median_list = np.array(min_list), np.array(max_list), np.array(median_list)
            sigma_1_plus_list, sigma_1_minus_list = np.array(sigma_1_plus_list), np.array(sigma_1_minus_list)
        
            # moving average of each with a window of 10
            N = 100
            min_list = np.convolve(min_list, np.ones(N)/N, mode='valid')
            max_list = np.convolve(max_list, np.ones(N)/N, mode='valid')
            median_list = np.convolve(median_list, np.ones(N) / N, mode='valid')
            sigma_1_plus_list = np.convolve(sigma_1_plus_list, np.ones(N)/N, mode='valid')
            sigma_1_minus_list = np.convolve(sigma_1_minus_list, np.ones(N)/N, mode='valid')
            t_fine_moving = np.convolve(t_fine, np.ones(N)/N, mode='valid')

            if (percentage_bool and variables_list[i] != 'ecc'):
                start_val = current_all_data_array[0, 11+i][0]
                min_list = 100 * min_list/start_val
                max_list = 100 * max_list / start_val
                median_list = 100 * median_list / start_val
                sigma_1_plus_list = 100 * sigma_1_plus_list / start_val
                sigma_1_minus_list = 100 * sigma_1_minus_list / start_val
            plt.figure(1000 + i)
            plt.fill_between(t_fine_moving, min_list, max_list, color=color_list[data_id], alpha=0.2)
            plt.plot(t_fine_moving, min_list, linestyle='-', color=color_list[data_id])
            plt.plot(t_fine_moving, max_list, linestyle='--', color=color_list[data_id])
            plt.plot(t_fine_moving, median_list, linestyle='-.', color=color_list[data_id])
            #plt.plot(t_fine_moving, sigma_1_plus_list, linestyle=':', color=color_list[data_id])
            #plt.plot(t_fine_moving, sigma_1_minus_list, linestyle='-', dashes=[8, 4, 2, 4, 2, 4], color=color_list[data_id])
            plt.plot([], [], linestyle='-', linewidth=7, color=color_list[data_id], label=labels_optical_list[data_id])

if (COMPARISON_PLOTS):
    for i in range(8):
        plt.figure(1000 + i)
        plt.plot([], [], color='k', linestyle='-', label='min')
        plt.plot([], [], color='k', linestyle='-.', label='median')
        plt.plot([], [], color='k', linestyle='--', label='max')

        #plt.plot([], [], color='k', linestyle=':', label=r'$+1-\sigma$')
        #plt.plot([], [], color='k', linestyle='-', dashes=[8, 4, 2, 4, 2, 4], label=r'$-1-\sigma$')
        plt.legend(ncol=2)
        plt.grid(True)
        plt.xlabel(r'$t$ [days]', fontsize=14)
        if (percentage_bool and variables_list[i] != 'ecc'):
            plt.ylabel(labels_change_percent_list[i], fontsize=14)
        else:
            plt.ylabel(labels_change_list[i], fontsize=14)
        plt.savefig(Project_directory + f'/0_FinalPlots/LTT/Comparisons/{Comparison_label}/{variables_list[i]}_{Comparison_label}.png', dpi=600, bbox_inches='tight')
        plt.close()



plt.show()




