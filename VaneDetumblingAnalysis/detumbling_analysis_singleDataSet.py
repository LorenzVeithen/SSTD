import numpy as np
import matplotlib.pyplot as plt
from generalConstants import Project_directory
from vaneDetumbling_ACS3Model import analysis_save_data_dir
from pathlib import Path
from cycler import cycler
import os
from mpl_toolkits.mplot3d import Axes3D

line_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]))
marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))

multiplying_factors = [1./1000, 1., 180./np.pi, 180./np.pi, 180./np.pi, 180./np.pi, 1./1000, 1./1000]
variables_list = ['sma', 'ecc', "inc", "aop", "raan", "tranom", "apo", "peri"]

keplerian_keys = ['sma', 'ecc', "inc", "aop", "raan", "tranom"]
omega_components_id = ['x', 'y', 'z']

# get all data together in gray to be able to point out specific ones to be analysed
rotational_velocity_labels_list = [r"$\omega_{x, \mathcal{B}}$ [deg/s]",
                                   r"$\omega_{y, \mathcal{B}}$ [deg/s]",
                                   r"$\omega_{z, \mathcal{B}}$ [deg/s]"]

thinning_factor = 1
selected_combinations = [(-40.0, -20.0, -40.0),
                         (60.0, 0.0, 0.0),
                         (55.0, 20.0, -30.0),
                         (0.0, 0.0, 70.0),
                         (0.0, 20.0, 0.0),
                         (40.0, 40.0, 40.0)]

def get_dataset_data(dataset):
    analysis_data_dir = analysis_save_data_dir + dataset
    states_history_dir = analysis_data_dir + "/states_history"
    dependent_variable_history_dir = analysis_data_dir + "/dependent_variable_history"

    # state history files
    p = Path(states_history_dir)
    state_history_files = [x for x in p.iterdir() if (not x.is_dir())]#[:10]

    # add the path with zero rotational velocity in case it was not in the original bundle
    omega_is_zero_vector_data = Path(f'{analysis_data_dir}/states_history/state_history_omega_x_0.0_omega_y_0.0_omega_z_0.0.dat')
    if (((omega_is_zero_vector_data in state_history_files) == False) and (omega_is_zero_vector_data.exists())):
        state_history_files += [omega_is_zero_vector_data]


    detumbling_results_dict = {}
    # First handle the Keplerian orbit case for comparison
    keplerian_dependent_variable_history_array = np.loadtxt(f'{analysis_data_dir}/keplerian_orbit_dependent_variable_history.dat')
    keplerian_state_history_array = np.loadtxt(f'{analysis_data_dir}/keplerian_orbit_state_history.dat')

    detumbling_results_dict["keplerian" + "_time_array"] = (keplerian_state_history_array[:, 0] - keplerian_state_history_array[0, 0]) / (24 * 3600)
    for i in range(6):
        detumbling_results_dict["keplerian" + f"_{keplerian_keys[i]}_array"] = keplerian_dependent_variable_history_array[:, i + 1]
    detumbling_results_dict["keplerian" + "_apo_array"] = keplerian_dependent_variable_history_array[:, 1] * (1 + keplerian_dependent_variable_history_array[:, 2])
    detumbling_results_dict["keplerian" + "_peri_array"] = keplerian_dependent_variable_history_array[:, 1] * (1 - keplerian_dependent_variable_history_array[:, 2])
    detumbling_results_dict[f"keplerian" + "_initial_omega_vector"] = np.array([0, 0, 0])

    # Initialise some dictionary entries
    detumbling_results_dict["all" + "_time_array"] = []
    for i in range(6):
        detumbling_results_dict["all" + f"_{keplerian_keys[i]}_array"] = []
    detumbling_results_dict["all" + "_apo_array"] = []
    detumbling_results_dict["all" + "_peri_array"] = []
    detumbling_results_dict["all" + "_detumbling_time_array"] = []
    detumbling_results_dict[f"all" + "_initial_omega_vector"] = []
    detumbling_results_dict[f"all" + "_omega_x_array"] = []
    detumbling_results_dict[f"all" + "_omega_y_array"] = []
    detumbling_results_dict[f"all" + "_omega_z_array"] = []
    detumbling_results_dict[f"all" + "_Tx_array"] = []
    detumbling_results_dict[f"all" + "_Ty_array"] = []
    detumbling_results_dict[f"all" + "_Tz_array"] = []

    selected_indices = []
    ordered_selected_combinations = []
    counter = -1
    for current_state_history_path in state_history_files:
        if (str(current_state_history_path).split('/')[-1][0] == '.'):
            continue
        else:
            counter += 1
        print(str(current_state_history_path).split('/')[-1])
        current_state_history_array = np.loadtxt(current_state_history_path)
        current_state_history_array = current_state_history_array[::thinning_factor]

        if (len(current_state_history_array[:, 0]) < 5):
            # some propagations may be broken, just remove them here
            print("skip")
            continue
        # get the initial rotational velocity vector of the propagation
        l = str(current_state_history_path)[:-4].split('_')
        omega_z_rph = float(l[-1])
        omega_y_rph = float(l[-4])
        omega_x_rph = float(l[-7])

        current_dependent_variable_history_path = dependent_variable_history_dir + f"/dependent_variable_history_omega_x_{omega_x_rph}_omega_y_{omega_y_rph}_omega_z_{omega_z_rph}.dat"
        current_dependent_variable_history_array = np.loadtxt(current_dependent_variable_history_path)
        current_dependent_variable_history_array = current_dependent_variable_history_array[::thinning_factor]
        initial_omega_vector_deg_per_sec = np.array([omega_x_rph, omega_y_rph, omega_z_rph]) / 10.
        initial_omega_vector_rph = np.array([omega_x_rph, omega_y_rph, omega_z_rph])

        print(tuple(initial_omega_vector_rph))
        if (tuple(initial_omega_vector_rph) in selected_combinations):
            selected_indices.append(counter)
            ordered_selected_combinations.append(tuple(initial_omega_vector_rph))

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

        # SRP torque history
        current_Tx_array = current_dependent_variable_history_array[:, 11]
        current_Ty_array = current_dependent_variable_history_array[:, 12]
        current_Tz_array = current_dependent_variable_history_array[:, 13]

        # get detumbling time
        list_indices_zero_angles = np.where(np.sum(abs(current_dependent_variable_history_array[:, 21:29]), axis=1) == 0)[0]
        if (len(list_indices_zero_angles) != 0):
            current_detumbling_time_hours = (current_state_history_array[list_indices_zero_angles[0], 0]
                                              - current_state_history_array[0, 0]) / 3600
        else:
            current_detumbling_time_hours = None

        if (((current_time_array[-1]-current_time_array[0])/3600/24)>1e-15 and current_detumbling_time_hours==None):
            print('stopping!')
            continue    # Did not finish properly
        print(list_indices_zero_angles)
        if (omega_x_rph==0 and omega_y_rph==0 and omega_z_rph==0):
            detumbling_results_dict["default" + "_time_array"] = current_time_array
            for i in range(6):
                detumbling_results_dict["default" + f"_{keplerian_keys[i]}_array"] = current_kep_array[:, i]
            detumbling_results_dict["default" + "_apo_array"] = current_apo_array
            detumbling_results_dict["default" + "_peri_array"] = current_peri_array
            detumbling_results_dict[f"default" + "_omega_x_array"] = omega_x_array_deg_s
            detumbling_results_dict[f"default" + "_omega_y_array"] = omega_y_array_deg_s
            detumbling_results_dict[f"default" + "_omega_z_array"] = omega_z_array_deg_s
            detumbling_results_dict[f"default" + "_Tx_array"] = current_Tx_array
            detumbling_results_dict[f"default" + "_Ty_array"] = current_Ty_array
            detumbling_results_dict[f"default" + "_Tz_array"] = current_Tz_array
        else:
            detumbling_results_dict["all" + "_time_array"].append(current_time_array)
            for i in range(6):
                detumbling_results_dict["all" + f"_{keplerian_keys[i]}_array"].append(current_kep_array[:, i])
            detumbling_results_dict["all" + "_apo_array"].append(current_apo_array)
            detumbling_results_dict["all" + "_peri_array"].append(current_peri_array)
            detumbling_results_dict["all" + "_detumbling_time_array"].append(current_detumbling_time_hours)
            detumbling_results_dict[f"all" + "_initial_omega_vector"].append(initial_omega_vector_deg_per_sec)
            detumbling_results_dict[f"all" + "_omega_x_array"].append(omega_x_array_deg_s)
            detumbling_results_dict[f"all" + "_omega_y_array"].append(omega_y_array_deg_s)
            detumbling_results_dict[f"all" + "_omega_z_array"].append(omega_z_array_deg_s)
            detumbling_results_dict[f"all" + "_Tx_array"].append(current_Tx_array)
            detumbling_results_dict[f"all" + "_Ty_array"].append(current_Ty_array)
            detumbling_results_dict[f"all" + "_Tz_array"].append(current_Tz_array)

    processed_array_dict = detumbling_results_dict

    # scale variables as desired
    for i in range(len(variables_list)):
        #processed_array_dict["default" + f"_{variables_list[i]}_array"] = (
        #        np.array(processed_array_dict["default" + f"_{variables_list[i]}_array"]) * multiplying_factors[i])
        processed_array_dict["keplerian" + f"_{variables_list[i]}_array"] = (
                np.array(processed_array_dict["keplerian" + f"_{variables_list[i]}_array"]) * multiplying_factors[i])
        for j in range(len(processed_array_dict["all" + f"_{variables_list[i]}_array"])):
            processed_array_dict["all" + f"_{variables_list[i]}_array"][j] = (
                    processed_array_dict["all" + f"_{variables_list[i]}_array"][j] * multiplying_factors[i])

    # Make general array to allow easy parsing of the data based on different cases for univariate dict elements
    omega_norm_list = [np.linalg.norm(om) for om in detumbling_results_dict[f"all" + "_initial_omega_vector"]]
    omega_x_list = [om[0] for om in detumbling_results_dict[f"all" + "_initial_omega_vector"]]
    omega_y_list = [om[1] for om in detumbling_results_dict[f"all" + "_initial_omega_vector"]]
    omega_z_list = [om[2] for om in detumbling_results_dict[f"all" + "_initial_omega_vector"]]

    all_data_array = np.empty((len(omega_norm_list), 19), dtype=object)
    all_data_array[:, 0] = omega_norm_list
    all_data_array[:, 1] = omega_x_list
    all_data_array[:, 2] = omega_y_list
    all_data_array[:, 3] = omega_z_list
    all_data_array[:, 4] = detumbling_results_dict["all" + "_detumbling_time_array"]
    all_data_array[:, 5] = processed_array_dict["all" + "_omega_x_array"]
    all_data_array[:, 6] = processed_array_dict["all" + "_omega_y_array"]
    all_data_array[:, 7] = processed_array_dict["all" + "_omega_z_array"]
    all_data_array[:, 8] = processed_array_dict["all" + "_time_array"]
    all_data_array[:, 9] = processed_array_dict["all" + "_Tx_array"]
    all_data_array[:, 10] = processed_array_dict["all" + "_Ty_array"]
    all_data_array[:, 11] = processed_array_dict["all" + "_Tz_array"]   # do not touch anything above here
    all_data_array[:, 12] = processed_array_dict["all" + "_peri_array"]
    all_data_array[:, 13] = processed_array_dict["all" + "_apo_array"]
    all_data_array[:, 14] = processed_array_dict["all" + f"_sma_array"]
    all_data_array[:, 15] = processed_array_dict["all" + f"_ecc_array"]
    all_data_array[:, 16] = processed_array_dict["all" + f"_inc_array"]
    all_data_array[:, 17] = processed_array_dict["all" + f"_aop_array"]
    all_data_array[:, 18] = processed_array_dict["all" + f"_raan_array"]
    return processed_array_dict, all_data_array, selected_indices, ordered_selected_combinations

first_dataset = f'/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_single_ideal_opt_model_shadow_False'
first_processed_array_dict, first_all_data_array, first_selected_indices, first_ordered_selected_combinations = get_dataset_data(first_dataset)

save_plots_dir = Project_directory + '/0_FinalPlots/Detumbling/General' + first_dataset
if (not os.path.exists(save_plots_dir)):
    os.makedirs(save_plots_dir)

# Global scatter plot
zero_omega_x_data = first_all_data_array[np.where((first_all_data_array[:, 1] == 0)
                                                  & (np.all(first_all_data_array[:, 2:4] != 0, axis=1)))[0], :]
zero_omega_y_data = first_all_data_array[np.where((first_all_data_array[:, 2] == 0)
                                                  & (first_all_data_array[:, 1] != 0)
                                                  & (first_all_data_array[:, 3] != 0))[0], :]
zero_omega_z_data = first_all_data_array[np.where((first_all_data_array[:, 3] == 0)
                                                  & (np.all(first_all_data_array[:, 1:3] != 0, axis=1)))[0], :]

nonzero_omega_x_data = first_all_data_array[np.where((first_all_data_array[:, 1] != 0)
                                                     & (np.all(first_all_data_array[:, 2:4] == 0, axis=1)))[0], :]
nonzero_omega_y_data = first_all_data_array[np.where((first_all_data_array[:, 2] != 0)
                                                     & (first_all_data_array[:, 1] == 0)
                                                     & (first_all_data_array[:, 3] == 0))[0], :]
nonzero_omega_z_data = first_all_data_array[np.where((first_all_data_array[:, 3] != 0)
                                                     & (np.all(first_all_data_array[:, 1:3] == 0, axis=1)))[0], :]
no_zero_element = first_all_data_array[np.where(np.all(first_all_data_array[:, 1:4] != 0, axis=1))[0], :]

first_all_data_under_7degs = first_all_data_array[np.where(first_all_data_array[:, 0] < 7)[0], :]
m, b = np.polyfit(list(first_all_data_under_7degs[:, 0]), list(first_all_data_under_7degs[:, 4]), 1)
marker_list = ["4", "2", "3", "1", "+", "x", "."]
color_list = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
plt.rc("axes", prop_cycle=marker_cycler)
plt.figure()
plt.scatter(no_zero_element[:, 0], no_zero_element[:, 4]/24, s=100, label=r'$\vec{\omega}_{0, \mathcal{B}} \in X_{\text{DT-3}}$',
                                                                    marker=marker_list[0],
                                                                    color=color_list[0])# r'$\omega_{x_{0}, \mathcal{B}} \neq 0, \omega_{y_{0}, \mathcal{B}} \neq 0, \omega_{z_{0}, \mathcal{B}} \neq 0$'
plt.scatter(zero_omega_x_data[:, 0], zero_omega_x_data[:, 4]/24, s=100, label=r'$\vec{\omega}_{0, \mathcal{B}} \in X_{\text{DT-2}}: \omega_{y_{0}, \mathcal{B}} \neq 0, \omega_{z_{0}, \mathcal{B}} \neq 0$',
                                                                    marker=marker_list[1],
                                                                    color=color_list[1]) #  \omega_{x_{0}, \mathcal{B}} = 0, \omega_{y_{0}, \mathcal{B}} \neq 0, \omega_{z_{0}, \mathcal{B}} \neq 0
plt.scatter(zero_omega_y_data[:, 0], zero_omega_y_data[:, 4]/24, s=100, label=r'$\vec{\omega}_{0, \mathcal{B}} \in X_{\text{DT-2}}: \omega_{x_{0}, \mathcal{B}} \neq 0, \omega_{z_{0}, \mathcal{B}} \neq 0$',
                                                                    marker=marker_list[2],
                                                                    color=color_list[2])# \omega_{x_{0}, \mathcal{B}} \neq 0, \omega_{y_{0}, \mathcal{B}} = 0, \omega_{z_{0}, \mathcal{B}} \neq 0
plt.scatter(zero_omega_z_data[:, 0], zero_omega_z_data[:, 4]/24, s=100, label=r'$\vec{\omega}_{0, \mathcal{B}} \in X_{\text{DT-2}}: \omega_{x_{0}, \mathcal{B}} \neq 0, \omega_{y_{0}, \mathcal{B}} \neq 0$',
                                                                    marker=marker_list[3],
                                                                    color=color_list[3])# \omega_{x_{0}, \mathcal{B}} \neq 0, \omega_{y_{0}, \mathcal{B}} \neq 0, \omega_{z_{0}, \mathcal{B}} = 0
plt.scatter(nonzero_omega_x_data[:, 0], nonzero_omega_x_data[:, 4]/24, s=100, label=r'$\vec{\omega}_{0, \mathcal{B}} \in X_{\text{DT-1}}: \omega_{x_{0}, \mathcal{B}} \neq 0$',
                                                                    marker=marker_list[4],
                                                                    color=color_list[4])# \omega_{x_{0}, \mathcal{B}} \neq 0, \omega_{y_{0}, \mathcal{B}} = 0, \omega_{z_{0}, \mathcal{B}} = 0
plt.scatter(nonzero_omega_y_data[:, 0], nonzero_omega_y_data[:, 4]/24, s=100, label=r'$\vec{\omega}_{0, \mathcal{B}} \in X_{\text{DT-1}}: \omega_{y_{0}, \mathcal{B}} \neq 0$',
                                                                    marker=marker_list[5],
                                                                    color=color_list[5])# \omega_{x_{0}, \mathcal{B}} = 0, \omega_{y_{0}, \mathcal{B}} \neq 0, \omega_{z_{0}, \mathcal{B}} = 0
plt.scatter(nonzero_omega_z_data[:, 0], nonzero_omega_z_data[:, 4]/24, s=100, label=r'$\vec{\omega}_{0, \mathcal{B}} \in X_{\text{DT-1}}: \omega_{z_{0}, \mathcal{B}} \neq 0$',
                                                                    marker=marker_list[6],
                                                                    color=color_list[6])#\omega_{x_{0}, \mathcal{B}} = 0, \omega_{y_{0}, \mathcal{B}} = 0, \omega_{z_{0}, \mathcal{B}} \neq 0

x_trend = np.linspace(0, 15, 10)
y_trend = m * x_trend + b
plt.grid(True)
plt.xlabel(r"$||\vec{\omega}_{0, \mathcal{B}}||$ [deg/s]", fontsize=14)
plt.ylabel(r'$\Delta t_{\text{detumbling}}$ [days]', fontsize=14)
plt.plot(x_trend, y_trend/24, linestyle='--', color='k', label=f'Linear trend: {round(m/24, 1)}' + r'$||\vec{\omega}_{0, \mathcal{B}}||$' + f' - {abs(round(b/24, 1))}')
plt.legend()
plt.savefig(save_plots_dir +'/general_scatter_detumbling.png',
            dpi=600,
            bbox_inches='tight')

heatmaps_data = np.empty((len(first_all_data_array[:, 0]), 5), dtype='float64')
heatmaps_data[:, 0] = first_all_data_array[:, 0]
heatmaps_data[:, 1:5] = first_all_data_array[:, 1:5]

ht_zero_omega_x_data = heatmaps_data[np.where(heatmaps_data[:, 1] == 0)[0], :]
ht_zero_omega_y_data = heatmaps_data[np.where(heatmaps_data[:, 2] == 0)[0], :]
ht_zero_omega_z_data = heatmaps_data[np.where(heatmaps_data[:, 3] == 0)[0], :]

color_data = heatmaps_data[:, 4] / 24
vmin = np.min(color_data)
vmax = np.max(color_data)

my_cmap = plt.get_cmap('plasma')
f, ax = plt.subplots()
ax.set_xlabel(r'$\omega_{x_{0}, \mathcal{B}}$ [deg/s]', fontsize=14)
ax.set_ylabel(r'$\omega_{y_{0}, \mathcal{B}}$ [deg/s]', fontsize=14)
tpc = ax.tripcolor(ht_zero_omega_z_data[:, 1], ht_zero_omega_z_data[:, 2], ht_zero_omega_z_data[:, 4]/24,
                   shading='flat', cmap=my_cmap, vmin=vmin, vmax=vmax)
cbar = f.colorbar(tpc)
cbar.set_label(r'$\Delta t_{\text{detumbling}}$ [days]', rotation=270, labelpad=15, fontsize=14)
plt.savefig(save_plots_dir +'/xy_2D_heatmap_detumbling.png',
            dpi=600,
            bbox_inches='tight')

f, ax = plt.subplots()
ax.set_xlabel(r'$\omega_{x_{0}, \mathcal{B}}$ [deg/s]', fontsize=14)
ax.set_ylabel(r'$\omega_{z_{0}, \mathcal{B}}$ [deg/s]', fontsize=14)
tpc = ax.tripcolor(ht_zero_omega_y_data[:, 1], ht_zero_omega_y_data[:, 3], ht_zero_omega_y_data[:, 4]/24,
                   shading='flat', cmap=my_cmap, vmin=vmin, vmax=vmax)
cbar = f.colorbar(tpc)
cbar.set_label(r'$\Delta t_{\text{detumbling}}$ [days]', rotation=270, labelpad=15, fontsize=14)
plt.savefig(save_plots_dir +'/xz_2D_heatmap_detumbling.png',
            dpi=600,
            bbox_inches='tight')

f, ax = plt.subplots()
ax.set_xlabel(r'$\omega_{y_{0}, \mathcal{B}}$ [deg/s]', fontsize=14)
ax.set_ylabel(r'$\omega_{z_{0}, \mathcal{B}}$ [deg/s]', fontsize=14)
tpc = ax.tripcolor(ht_zero_omega_x_data[:, 2], ht_zero_omega_x_data[:, 3], ht_zero_omega_x_data[:, 4]/24,
                   shading='flat', cmap=my_cmap, vmin=vmin, vmax=vmax)
cbar = f.colorbar(tpc)
cbar.set_label(r'$\Delta t_{\text{detumbling}}$ [days]', rotation=270, labelpad=15, fontsize=14)
plt.savefig(save_plots_dir +'/yz_2D_heatmap_detumbling.png',
            dpi=600,
            bbox_inches='tight')

# 3D scatter plot with heatmap according to detumbling times
ht_scatter3D = heatmaps_data#[heatmaps_data[:, 4]/24 > 6]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
withcolormap = ax.scatter(ht_scatter3D[:, 1], ht_scatter3D[:, 2], ht_scatter3D[:, 3], c = ht_scatter3D[:, 4]/24, cmap=my_cmap, vmin=vmin, vmax=vmax)
cbar = fig.colorbar(withcolormap, ax=ax, pad=0.1)
cbar.set_label(r'$\Delta t_{\text{detumbling}}$ [days]', rotation=270, labelpad=15, fontsize=14)
ax.set_xlabel(r'$\omega_{x_{0}, \mathcal{B}}$ [deg/s]', fontsize=14)
ax.set_ylabel(r'$\omega_{y_{0}, \mathcal{B}}$ [deg/s]', fontsize=14)
ax.set_zlabel(r'$\omega_{z_{0}, \mathcal{B}}$ [deg/s]', fontsize=14)
plt.savefig(save_plots_dir +'/3D_heatmap_detumbling.png',
            dpi=600,
            bbox_inches='tight')

# Single axis detumbling only
plt.figure()
plt.scatter(nonzero_omega_x_data[:, 0], nonzero_omega_x_data[:, 4]/24, s=100, label=r'$\omega_{x_{0}, \mathcal{B}} \neq 0, \omega_{y_{0}, \mathcal{B}}, \omega_{z_{0}, \mathcal{B}} = 0$',
                                                                    marker=marker_list[4],
                                                                    color=color_list[4])
plt.scatter(nonzero_omega_y_data[:, 0], nonzero_omega_y_data[:, 4]/24, s=100, label=r'$\omega_{y_{0}, \mathcal{B}} \neq 0, \omega_{x_{0}, \mathcal{B}}, \omega_{z_{0}, \mathcal{B}} = 0$',
                                                                    marker=marker_list[5],
                                                                    color=color_list[5])
plt.scatter(nonzero_omega_z_data[:, 0], nonzero_omega_z_data[:, 4]/24, s=100, label=r'$\omega_{z_{0}, \mathcal{B}} \neq 0, \omega_{x_{0}, \mathcal{B}}, \omega_{y_{0}, \mathcal{B}} = 0$',
                                                                    marker=marker_list[6],
                                                                    color=color_list[6])
plt.grid(True)
plt.xlabel(r"$||\vec{\omega}_{0, \mathcal{B}}||$ [deg/s]", fontsize=14)
plt.ylabel(r'$\Delta t_{\text{detumbling}}$ [days]', fontsize=14)
plt.legend()
plt.savefig(save_plots_dir +'/scatter_single_axis_detumbling.png',
            dpi=600,
            bbox_inches='tight')

ALL_COMBINED = True
if (ALL_COMBINED):
    # Combined general scatter plot for different optical properties
    for mode in range(2):
        if (mode==0):
            dataset_labels = ['SI-SRP', 'DI-SRP', 'O-SRP']
            comparison_label = 'optical_sh_False'
            files_list = [f'/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_single_ideal_opt_model_shadow_False',
                            f'/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_False',
                            f'/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_ACS3_opt_model_shadow_False',
                            ]
            reference_dataset_id = 2

        elif (mode==1):
            dataset_labels = ['SI-SRP', 'DI-SRP', 'O-SRP']
            comparison_label = 'optical_sh_True'
            files_list = [f'/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_single_ideal_opt_model_shadow_True',
                            f'/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_True',
                            f'/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_ACS3_opt_model_shadow_True',
                        ]
            reference_dataset_id = 2
        elif (mode==2):
            dataset_labels = ['Without self-shadowing constraint', 'With self-shadowing constraint']
            comparison_label = 'shadow_SI'
            files_list = [f'/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_single_ideal_opt_model_shadow_False',
                        f'/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_single_ideal_opt_model_shadow_True',
                        ]
            reference_dataset_id = 0
        elif (mode==3):
            dataset_labels = ['Without self-shadowing constraint', 'With self-shadowing constraint']
            comparison_label = 'shadow_ACS3'
            files_list = [f'/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_ACS3_opt_model_shadow_False',
                        f'/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_ACS3_opt_model_shadow_True',
                        ]
            reference_dataset_id = 0

        datasets_array_list = []
        datasets_array_sizes_list = []
        for cd_id, current_data_set_filter in enumerate(files_list):
            if current_data_set_filter == first_dataset:
                current_processed_dict_filter, current_all_array_filter = first_processed_array_dict, first_all_data_array
            else:
                current_processed_dict_filter, current_all_array_filter, _, _ = get_dataset_data(current_data_set_filter)

            datasets_array_sizes_list.append(len(current_all_array_filter[:, 0]))
            datasets_array_list.append(current_all_array_filter)

        # Find the dataset with the smallest size (most constraining dataset)
        most_constraining_dataset = datasets_array_list[datasets_array_sizes_list.index(min(datasets_array_sizes_list))]
        key_to_mc_row = {tuple(row[:4]): row for row in datasets_array_list[reference_dataset_id]}
        print(datasets_array_sizes_list)
        print(min(datasets_array_sizes_list))

        filtered_datasets_list = []
        # Filter each dataset in datasets_array_list
        for dataset in datasets_array_list:
            filtered_dataset = np.array([row for row in dataset if
                                         any(np.array_equal(row[:4], mc_row[:4]) for mc_row in most_constraining_dataset)])
            filtered_datasets_list.append(filtered_dataset)


        # Optionally, verify the sizes of filtered datasets
        for idx, filtered_dataset in enumerate(filtered_datasets_list):
            print(f"Filtered dataset {idx} size: {filtered_dataset.shape}")

        percentage_differences_lists = []
        omega_lists_diffs = []
        for cd_id, current_all_array in enumerate(filtered_datasets_list):
            percentage_diff_list_to_mc = []
            percentage_diff_to_mc_omega_list = []
            for row in current_all_array:
                key = tuple(row[:4])
                if key in key_to_mc_row:
                    percentage_diff_to_mc_omega_list.append(row[0])
                    percentage_diff_list_to_mc.append(100*(row[4]-key_to_mc_row[key][4])/key_to_mc_row[key][4])
                    # filter to only use the datasets available in the first one
            omega_lists_diffs.append(percentage_diff_to_mc_omega_list)
            percentage_differences_lists.append(percentage_diff_list_to_mc)
            plt.figure(111+mode*100)     # General scatter
            plt.scatter(current_all_array[:, 0], current_all_array[:, 4]/24, s=25, label=dataset_labels[cd_id],
                                                                                marker=marker_list[-3 + cd_id],
                                                                                color=color_list[cd_id])
            plt.figure(112+mode*100)     # Only with single axis
            nonzero_omega_x_data = current_all_array[np.where((current_all_array[:, 1] != 0)
                                                                 & (np.all(current_all_array[:, 2:4] == 0, axis=1)))[0], :]
            nonzero_omega_y_data = current_all_array[np.where((current_all_array[:, 2] != 0)
                                                                 & (current_all_array[:, 1] == 0)
                                                                 & (current_all_array[:, 3] == 0))[0], :]
            nonzero_omega_z_data = current_all_array[np.where((current_all_array[:, 3] != 0)
                                                                 & (np.all(current_all_array[:, 1:3] == 0, axis=1)))[0], :]

            plt.scatter(nonzero_omega_x_data[:, 0], nonzero_omega_x_data[:, 4] / 24, s=25,
                        marker='v',
                        color=color_list[cd_id])
            plt.scatter(nonzero_omega_y_data[:, 0], nonzero_omega_y_data[:, 4] / 24, s=25,
                        marker='1',
                        color=color_list[cd_id])
            plt.scatter(nonzero_omega_z_data[:, 0], nonzero_omega_z_data[:, 4] / 24, s=25,
                        marker='d',
                        color=color_list[cd_id])
            plt.plot([], [], linestyle='-', linewidth=7, label=dataset_labels[cd_id], color=color_list[cd_id])

            if (cd_id != reference_dataset_id):
                plt.figure(113 + mode * 100)
                plt.scatter(percentage_diff_to_mc_omega_list, percentage_diff_list_to_mc,
                            label=dataset_labels[cd_id],
                            color=color_list[cd_id],
                            marker=marker_list[-3 + cd_id])
                print('HERE')
                print(np.min(percentage_diff_list_to_mc), np.mean(percentage_diff_list_to_mc), np.max(percentage_diff_list_to_mc))


        plt.figure(111+mode*100)
        plt.grid(True)
        plt.xlabel(r"$||\vec{\omega}_{0, \mathcal{B}}||$ [deg/s]", fontsize=14)
        plt.ylabel(r'$\Delta t_{\text{detumbling}}$ [days]', fontsize=14)
        plt.legend()
        plt.savefig(Project_directory + f'/0_FinalPlots/Detumbling/General/{comparison_label}_general_scatter_detumbling.png',
                    dpi=600,
                    bbox_inches='tight')

        plt.figure(112+mode*100)
        plt.scatter([], [], marker='v', label=r'$\omega_{x_{0}, \mathcal{B}} \neq 0$', color='k')
        plt.scatter([], [], marker='1', label=r'$\omega_{y_{0}, \mathcal{B}} \neq 0$', color='k')
        plt.scatter([], [], marker='d', label=r'$\omega_{z_{0}, \mathcal{B}} \neq 0$', color='k')
        plt.grid(True)
        plt.xlabel(r"$||\vec{\omega}_{0, \mathcal{B}}||$ [deg/s]", fontsize=14)
        plt.ylabel(r'$\Delta t_{\text{detumbling}}$ [days]', fontsize=14)
        plt.legend(ncol=2)
        plt.savefig(Project_directory + f'/0_FinalPlots/Detumbling/General/{comparison_label}_single_scatter_detumbling.png',
                    dpi=600,
                    bbox_inches='tight')

        plt.figure(113 + mode * 100)
        plt.grid(True)
        plt.xlabel(r"$||\vec{\omega}_{0, \mathcal{B}}||$ [deg/s]", fontsize=14)
        plt.ylabel(r'$\Delta t_{\text{detumbling}}$ Difference [%]', fontsize=14)
        plt.legend()


ALL_PLOTS = False
if (ALL_PLOTS):

    # General scatter plots
    plt.figure()
    plt.scatter(first_all_data_array[:, 1], first_all_data_array[:, 4], s=3)
    plt.grid(True)
    plt.xlabel(r"$\omega_{x_{0}, \mathcal{B}}$ [deg/s]", fontsize=14)
    plt.ylabel(r"$\Delta t_{detumbling}$ [hours]", fontsize=14)

    plt.figure()
    plt.scatter(first_all_data_array[:, 2], first_all_data_array[:, 4], s=3)
    plt.grid(True)
    plt.xlabel(r"$\omega_{y_{0}, \mathcal{B}}$ [deg/s]", fontsize=14)
    plt.ylabel(r"$\Delta t_{detumbling}$ [hours]", fontsize=14)

    plt.figure()
    plt.scatter(first_all_data_array[:, 3], first_all_data_array[:, 4], s=3)
    plt.grid(True)
    plt.xlabel(r"$\omega_{z_{0}, \mathcal{B}}$ [deg/s]", fontsize=14)
    plt.ylabel(r"$\Delta t_{detumbling}$ [hours]", fontsize=14)

    ordered_selected_combinations_temp = np.array(first_ordered_selected_combinations)
    ordered_selected_combinations_norm = np.sqrt(ordered_selected_combinations_temp[:, 0]**2
                                                    + ordered_selected_combinations_temp[:, 1]**2
                                                    + ordered_selected_combinations_temp[:, 2]**2)

    temp_to_sort = np.empty((len(first_selected_indices), 4), dtype=object)
    temp_to_sort[:, 0] = ordered_selected_combinations_norm
    temp_to_sort[:, 1] = first_selected_indices
    temp_to_sort[:, 2] = first_ordered_selected_combinations
    temp_to_sort[:, 3] = [x.count(0) for x in first_ordered_selected_combinations]

    temp_to_sort = temp_to_sort[np.lexsort((temp_to_sort[:,0][::-1], temp_to_sort[:,3]))]
    first_selected_indices = temp_to_sort[:, 1]
    first_ordered_selected_combinations = temp_to_sort[:, 2]

    for k_id, omega_key in enumerate(["x", "y", "z"]):
        plt.figure()
        for i, current_t_array in enumerate(first_processed_array_dict["all" + "_time_array"]):
            plt.plot(current_t_array, first_processed_array_dict["all" + f"_omega_{omega_key}_array"][i], alpha=0.5, color="darkgrey")
        for s_id, selected_id in enumerate(first_selected_indices):
            plt.plot(first_processed_array_dict["all" + "_time_array"][selected_id],
                     first_processed_array_dict["all" + f"_omega_{omega_key}_array"][selected_id],
                     label=f'{first_ordered_selected_combinations[s_id]}')
        plt.legend()
        plt.xlabel(r"Time, $t$, [days]", fontsize=14)
        plt.ylabel(rotational_velocity_labels_list[k_id], fontsize=14)
        plt.grid(True)
        plt.xlim((0, max(first_processed_array_dict["all" + "_detumbling_time_array"]) / 24.))


    # omega_x as only non-zero and positive
    dataset_singleAxisOmegaX = first_all_data_array[(first_all_data_array[:, 1] == first_all_data_array[:, 0]), :]
    dataset_singleAxisOmegaY = first_all_data_array[(first_all_data_array[:, 2] == first_all_data_array[:, 0]), :]
    dataset_singleAxisOmegaZ = first_all_data_array[(first_all_data_array[:, 3] == first_all_data_array[:, 0]), :]

    plt.figure()
    plt.scatter(dataset_singleAxisOmegaX[:, 0], dataset_singleAxisOmegaX[:, 4], label=r'$\omega_{y_{0}, \mathcal{B}} = \omega_{z_{0}, \mathcal{B}} = 0$')
    plt.scatter(dataset_singleAxisOmegaY[:, 0], dataset_singleAxisOmegaY[:, 4], label=r'$\omega_{x_{0}, \mathcal{B}} = \omega_{z_{0}, \mathcal{B}} = 0$')
    plt.scatter(dataset_singleAxisOmegaZ[:, 0], dataset_singleAxisOmegaZ[:, 4], label=r'$\omega_{x_{0}, \mathcal{B}} = \omega_{y_{0}, \mathcal{B}} = 0$')
    plt.grid(True)
    plt.legend()
    plt.xlabel(r"$||\vec{\omega}_{0, \mathcal{B}}||$ [deg/s]", fontsize=14)
    plt.ylabel(r"$\Delta t_{detumbling}$ [hours]", fontsize=14)

    suptitles_list = [r'$\omega_{x_{0}, \mathcal{B}} \neq 0$', r'$\omega_{y_{0}, \mathcal{B}} \neq 0$', r'$\omega_{z_{0}, \mathcal{B}} \neq 0$']
    for j, dataset in enumerate([dataset_singleAxisOmegaX, dataset_singleAxisOmegaY, dataset_singleAxisOmegaZ]):
        max_detumbling_time_days = max(dataset[:, 4] / 24.)

        fig, axs = plt.subplots(1, 3)
        (ax1, ax2, ax3) = axs
        fig.suptitle(suptitles_list[j])
        for i in range(len(dataset[:, 0])):
            ax1.plot(dataset[:, 8][i], dataset[:, 5][i])
            ax2.plot(dataset[:, 8][i], dataset[:, 6][i])
            ax3.plot(dataset[:, 8][i], dataset[:, 7][i])

        ax1.grid(True)
        ax1.set_xlabel(r"$t$ [days]", fontsize=14)
        ax1.set_ylabel(r"$\omega_{x, \mathcal{B}}$ [deg/s]", fontsize=14)

        ax2.grid(True)
        ax2.set_xlabel(r"$t$ [days]", fontsize=14)
        ax2.set_ylabel(r"$\omega_{y, \mathcal{B}}$ [deg/s]", fontsize=14)

        ax3.grid(True)
        ax3.set_xlabel(r"$t$ [days]", fontsize=14)
        ax3.set_ylabel(r"$\omega_{z, \mathcal{B}}$ [deg/s]", fontsize=14)
        fig.tight_layout()
        custom_xlim = (0, max_detumbling_time_days * 1.05)
        plt.setp(axs, xlim=custom_xlim)
    """
        fig, axs = plt.subplots(1, 3)
        (ax1, ax2, ax3) = axs
        fig.suptitle(suptitles_list[j])
        for i in range(len(dataset[:, 0])):
            ax1.plot(dataset[:, 8][i], dataset[:, 9][i])
            ax2.plot(dataset[:, 8][i], dataset[:, 10][i])
            ax3.plot(dataset[:, 8][i], dataset[:, 11][i])
    
        ax1.grid(True)
        ax1.set_xlabel(r"Time, $t$, [days]", fontsize=14)
        ax1.set_ylabel(r"X-axis body torque, $T_{x}$, [Nm]", fontsize=14)
    
        ax2.grid(True)
        ax2.set_xlabel(r"Time, $t$, [days]", fontsize=14)
        ax2.set_ylabel(r"Y-axis body torque, $T_{y}$, [Nm]", fontsize=14)
    
        ax3.grid(True)
        ax3.set_xlabel(r"Time, $t$, [days]", fontsize=14)
        ax3.set_ylabel(r"Z-axis body torque, $T_{z}$, [Nm]", fontsize=14)
        fig.tight_layout()
        #custom_xlim = (0, max_detumbling_time_days * 1.05)
        plt.setp(axs, xlim=custom_xlim)
    """
print(m, b)
plt.show()
