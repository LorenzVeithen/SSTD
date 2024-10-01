import numpy as np
from generalConstants import Project_directory
import matplotlib.pyplot as plt
from pathlib import Path


color_list = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
line_style_loop = ["-", "--", "-.", ":", "-", "--", "-."]
labels = ['SI-SRP', 'DI-SRP', 'O-SRP']
analysis_dirs = [f'/0_GeneratedData/DetumblingAnalysis/SingleOrbit/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_single_ideal_opt_model_shadow_True',
                 f'/0_GeneratedData/DetumblingAnalysis/SingleOrbit/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_True',
                 f'/0_GeneratedData/DetumblingAnalysis/SingleOrbit/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_ACS3_opt_model_shadow_True']
plt.figure(1)
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='--', label="Effectivity threshold")
plt.xlabel(r'$||\vec{\omega}_{0, \mathcal{B}}||$ [deg/s]', fontsize=14)
plt.ylabel(r'Average $\Delta ||\vec{\omega}_{\mathcal{B}}||$ per orbit [deg/s]', fontsize=14)

for count, ad in enumerate(analysis_dirs):
    analysis_data_dir = Project_directory + ad
    states_history_dir = analysis_data_dir + "/states_history"

    # state history files
    p = Path(states_history_dir)
    state_history_files = [x for x in p.iterdir() if (not x.is_dir())]


    initial_omega_list_two_axis = []
    initial_omega_list_three_axis = []
    final_omega_list_two_axis = []
    final_omega_list_three_axis = []
    for current_state_history_path in state_history_files:
        if (str(current_state_history_path).split('/')[-1][0] == '.'):
            continue
        current_state_history_array = np.loadtxt(current_state_history_path)

        if (len(current_state_history_array[:, 0]) < 5):
            # some propagations may be broken, just remove them here
            print("skip")
            continue
        # get the initial rotational velocity vector of the propagation
        l = str(current_state_history_path)[:-4].split('_')
        omega_z_rph = float(l[-1])
        omega_y_rph = float(l[-4])
        omega_x_rph = float(l[-7])

        if (omega_x_rph == omega_y_rph == 0 or omega_z_rph == 0):
            continue

        initial_omega_vector_deg_s = np.array([omega_x_rph, omega_y_rph, omega_z_rph]) / 10.
        initial_omega_vector_rph = np.array([omega_x_rph, omega_y_rph, omega_z_rph])

        print(tuple(initial_omega_vector_rph))

        # extract Keplerian elements history
        # 1: Semi-major Axis. 2: Eccentricity. 3: Inclination. 4: Argument of Periapsis.
        # 5. Right Ascension of the Ascending Node. 6: True Anomaly.
        current_time_array = (current_state_history_array[:, 0]-current_state_history_array[0, 0])/(60)

        # rotational velocity history
        omega_x_array_deg_s = np.rad2deg(current_state_history_array[:, 11])
        omega_y_array_deg_s = np.rad2deg(current_state_history_array[:, 12])
        omega_z_array_deg_s = np.rad2deg(current_state_history_array[:, 13])

        omega_norm_deg_s = np.sqrt(omega_x_array_deg_s**2 + omega_y_array_deg_s**2 + omega_z_array_deg_s**2)

        if (omega_z_rph == 0):
            final_omega_list_two_axis.append(omega_norm_deg_s[-1])
            initial_omega_list_two_axis.append(np.linalg.norm(initial_omega_vector_deg_s))
        else:
            final_omega_list_three_axis.append(omega_norm_deg_s[-1])
            initial_omega_list_three_axis.append(np.linalg.norm(initial_omega_vector_deg_s))

    final_omega_list_two_axis = np.array(final_omega_list_two_axis)
    initial_omega_list_two_axis = np.array(initial_omega_list_two_axis)
    final_omega_list_three_axis = np.array(final_omega_list_three_axis)
    initial_omega_list_three_axis = np.array(initial_omega_list_three_axis)

    relative_change_two_axis = (final_omega_list_two_axis - initial_omega_list_two_axis)/6   # * 100/initial_omega_list_two_axis  # %
    relative_change_three_axis = (final_omega_list_three_axis - initial_omega_list_three_axis)/6   # * 100/initial_omega_list_three_axis  # %

    combined_data_three_axis = np.zeros((len(initial_omega_list_three_axis[initial_omega_list_three_axis < 100]), 2))
    combined_data_three_axis[:, 0] = initial_omega_list_three_axis[initial_omega_list_three_axis < 100]
    combined_data_three_axis[:, 1] = relative_change_three_axis[initial_omega_list_three_axis < 100]
    combined_data_three_axis = combined_data_three_axis[combined_data_three_axis[:, 0].argsort()]

    plt.figure(1)
    #plt.scatter(initial_omega_list_two_axis, relative_change_two_axis, label='2-axis')
    plt.plot(combined_data_three_axis[:, 0], combined_data_three_axis[:, 1],
                c=color_list[count],
                label=labels[count],
                linestyle=line_style_loop[count]
             )

    plt.figure()
    #plt.scatter(initial_omega_list_two_axis, relative_change_two_axis, label='2-axis')
    plt.scatter(initial_omega_list_three_axis[initial_omega_list_three_axis < 100], relative_change_three_axis[initial_omega_list_three_axis < 100],
                c="#E69F00")

    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', label="Effectivity threshold")
    plt.xlabel(r'$||\vec{\omega}_{0, \mathcal{B}}||$ [deg/s]', fontsize=14)
    plt.ylabel(r'Average $\Delta ||\vec{\omega}_{\mathcal{B}}||$ per orbit [deg/s]', fontsize=14)
    plt.legend(prop={'size': 8})
    plt.savefig('/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/Misc'
                + f'/MaxOmega_{labels[count]}.png',
                dpi=600,
                bbox_inches='tight')

plt.figure(1)
plt.legend(prop={'size': 8})
plt.savefig('/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/Misc'
            + f'/MaxOmega_combined.png',
            dpi=600,
            bbox_inches='tight')
plt.show()