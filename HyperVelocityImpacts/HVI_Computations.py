import numpy as np
import itertools
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
from generalConstants import Project_directory
from numba import jit
from scipy.interpolate import griddata

my_cmap = plt.get_cmap('plasma')

@jit(nopython=True, cache=True)
def projectile_direction_body_frame(theta, phi):
    # This is the same as the sunlight vector, but with theta and phi instead!
    return np.array([np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    -np.cos(theta)])

@jit(nopython=True, cache=True)
def generate_HVI_comparison_data(projectile_mass_list, projectile_velocity_list, beta_list,
                                 impact_position_vector_list, theta_list, phi_list, sail_I):
    # Prepare array of all combinations
    HVI_data_arr = np.empty((num_combinations, 11), dtype=float)
    counter = 0
    for mp in projectile_mass_list:  # Mass of the projectile
        for velocity in projectile_velocity_list:  # Projectile velocity
            for beta in beta_list:  # Momentum enhancement factor
                p_enhanced = (mp * velocity) * beta  # Enhanced linear momentum of the projectile
                for impact_body_fixed_position in impact_position_vector_list:  # Position on the given boom
                    for theta_id, theta in enumerate(theta_list):  # First direction angle
                        for phi_id, phi in enumerate(phi_list):  # Second direction angle

                            # Store data
                            HVI_data_arr[counter, 0] = mp
                            HVI_data_arr[counter, 1] = velocity
                            HVI_data_arr[counter, 2] = beta
                            if (impact_body_fixed_position[0] != 0. and impact_body_fixed_position[1] != 0):
                                HVI_data_arr[counter, 3] = 0.
                            else:
                                HVI_data_arr[counter, 3] = 0. if (
                                            impact_body_fixed_position[0] != 0.) else 1.  # 0 for X-axis and 1 for Y-axis
                            HVI_data_arr[counter, 4] = sum(impact_body_fixed_position)
                            HVI_data_arr[counter, 5] = theta
                            HVI_data_arr[counter, 6] = phi

                            # Compute attitude perturbation
                            transferred_momentum_vector = p_enhanced * projectile_direction_body_frame(theta, phi)
                            omega_increment = np.dot(np.linalg.inv(sail_I), (
                                np.cross(impact_body_fixed_position, transferred_momentum_vector)))

                            # Store attitude perturbation
                            HVI_data_arr[counter, 7] = omega_increment[0]
                            HVI_data_arr[counter, 8] = omega_increment[1]
                            HVI_data_arr[counter, 9] = omega_increment[2]
                            HVI_data_arr[counter, 10] = np.linalg.norm(omega_increment)

                            counter += 1
    return HVI_data_arr



# First considering the initial rotational velocity
beta_list = np.array([1., 2., 3., 4., 5.], dtype=float)
position_on_boom_list = list(np.linspace(-7, 7, 15))
#position_on_boom_list.remove(0)
position_on_boom_list += [-0.75, -0.5, -0.25, -0.125, -0.01, -0.001, 0.001, 0.01, 0.125, 0.25, 0.5, 0.75]
boom_direction_list = [np.array([1., 0., 0.])]#np.array([0., 1., 0.])    # First do the X-booms and then the Y-booms
impact_position_vector_list = []
for comb in list(itertools.product(position_on_boom_list, boom_direction_list)):
    impact_position_vector_list.append(comb[0] * comb[1])
impact_position_vector_list = np.array(impact_position_vector_list, dtype=float)

# Necessary sail characteristics
sail_I = np.zeros((3, 3))
sail_I[0, 0] = 10.5
sail_I[1, 1] = 10.5
sail_I[2, 2] = 21
booms_length = 7.    # m

# Orbital debris - assuming aluminium
rho_debris = 2700   # kg/m^3
projectile_diameter = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1., 10, 100], dtype=float) * 1e-3    # m  from 0.1 \micro m to 10 cm
projectile_mass_list = (4 / 3) * np.pi * ((projectile_diameter/2) ** 3) * rho_debris
projectile_velocity_list = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 30., 40, 50., 60., 70., 75.], dtype=float) * 1e3  # m/s
projectile_velocity_list_selected = np.array([1., 5., 10., 15., 20.], dtype=float) * 1e3  # m/s
theta_list = list(np.linspace(-np.pi, np.pi, 91))
phi_list = list(np.linspace(-np.pi, np.pi, 91))

if (not (0. in theta_list)):
    theta_list += [0.]
if (not (0. in phi_list)):
    phi_list += [0.]

num_combinations = len(projectile_mass_list) * len(projectile_velocity_list) * len(beta_list) * len(impact_position_vector_list) * len(theta_list) * len(phi_list)

GENERATE_DATA = False
if (GENERATE_DATA):
    HVI_data_arr = generate_HVI_comparison_data(projectile_mass_list,
                                                projectile_velocity_list,
                                                beta_list,
                                                impact_position_vector_list,
                                                theta_list,
                                                phi_list,
                                                sail_I)

    np.save(Project_directory + '/HyperVelocityImpacts/HVI_arr.npy', HVI_data_arr)
else:
    HVI_data_arr = np.load(Project_directory + '/HyperVelocityImpacts/HVI_arr.npy', mmap_mode='r')

print('Loaded data')
colors_list = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
# First plot - influence of the mass and relative velocity on the omega increment
plt.figure()
# Filter the dataset to the ones that are interesting
mp_HVI_data_arr = HVI_data_arr[HVI_data_arr[:, 5] == 0]   # Keep theta=0 for this analysis
mp_HVI_data_arr = mp_HVI_data_arr[mp_HVI_data_arr[:, 6] == 0]   # Keep phi=0 for this analysis
mp_HVI_data_arr = mp_HVI_data_arr[mp_HVI_data_arr[:, 3] == 0]     # Focus on the 'x' axis
mp_HVI_data_arr = mp_HVI_data_arr[mp_HVI_data_arr[:, 4] == 7]   # Focus on the ones at the tip of the booms
for mp_id, mp in enumerate(projectile_mass_list):
    current_HVI_data_arr_mp = mp_HVI_data_arr[mp_HVI_data_arr[:, 0] == mp]
    current_HVI_data_arr_mp_min_beta = current_HVI_data_arr_mp[current_HVI_data_arr_mp[:, 2] == 1]
    current_HVI_data_arr_mp_max_beta = current_HVI_data_arr_mp[current_HVI_data_arr_mp[:, 2] == 5]
    plt.plot(current_HVI_data_arr_mp_min_beta[:, 1]/1000, np.rad2deg(current_HVI_data_arr_mp_min_beta[:, 10]),
             color=colors_list[mp_id])
    plt.plot(current_HVI_data_arr_mp_max_beta[:, 1]/1000, np.rad2deg(current_HVI_data_arr_mp_max_beta[:, 10]),
             color=colors_list[mp_id])
    plt.fill_between(current_HVI_data_arr_mp_max_beta[:, 1]/1000,
                     np.rad2deg(current_HVI_data_arr_mp_max_beta[:, 10]),
                     np.rad2deg(current_HVI_data_arr_mp_min_beta[:, 10]),
                     color=colors_list[mp_id],
                     alpha=0.5,
                     label=r'$d_{p}$ = ' + f"{'{:0.0e}'.format(projectile_diameter[mp_id])} m")


plt.grid(True)
plt.xlabel(r'$\vec{v}_{p}$ [km/s]', fontsize=14)
plt.ylabel(r'$||\vec{\omega}_{0, \mathcal{B}}||$ [deg/s]', fontsize=14)
plt.axhline(y=0.01*np.sqrt(3), color='k', linestyle='--', label='Significance level')
plt.axhline(y=26, color='k', linestyle=':', label='Capability level')
plt.yscale('log')
plt.legend(ncol=3, loc='lower left', prop={'size': 8})
plt.ylim(1e-18, np.rad2deg(max(HVI_data_arr[:, 10]))*2)
plt.savefig("/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/HVI_Plots/HVI_proj_diam_speed.png",
            dpi=600,
            bbox_inches='tight')

# Second figure: linear momentum and beta
plt.figure()
# Filter the dataset to the ones that are interesting
beta_HVI_data_arr = HVI_data_arr[HVI_data_arr[:, 5] == 0]   # Keep theta=0 for this analysis
beta_HVI_data_arr = beta_HVI_data_arr[beta_HVI_data_arr[:, 6] == 0]   # Keep phi=0 for this analysis
beta_HVI_data_arr = beta_HVI_data_arr[beta_HVI_data_arr[:, 3] == 0]     # Focus on the 'x' axis
beta_HVI_data_arr = beta_HVI_data_arr[beta_HVI_data_arr[:, 4] == 7]   # Focus on the ones at the tip of the booms
for beta_id, beta in enumerate(beta_list):
    current_HVI_data_arr_beta = beta_HVI_data_arr[np.where(beta_HVI_data_arr[:, 2] == beta)[0], :]
    p = current_HVI_data_arr_beta[:, 0] * current_HVI_data_arr_beta[:, 1]   # Linear momentum of the considered particles

    reduced_arr = np.empty((len(p), 2))
    reduced_arr[:, 0] = p
    reduced_arr[:, 1] = current_HVI_data_arr_beta[:, 10]
    reduced_arr = reduced_arr[reduced_arr[:, 0].argsort()]  # order by the linear momentum
    plt.plot(reduced_arr[:, 0], np.rad2deg(reduced_arr[:, 1]),
             color=colors_list[beta_id], label=rf'$\beta$ = {beta}')
plt.grid(True)
plt.xlabel(r'$||\vec{p}_{p, \mathcal{B}}||$ [kg m/s]', fontsize=14)
plt.ylabel(r'$||\vec{\omega}_{0, \mathcal{B}}||$ [deg/s]', fontsize=14)
plt.axhline(y=0.01*np.sqrt(3), color='k', linestyle='--', label='Significance level')
plt.axhline(y=26, color='k', linestyle=':', label='Capability level')
plt.yscale('log')
plt.xscale('log')
plt.legend(ncol=1, prop={'size': 8}, loc='lower right')
plt.savefig("/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/HVI_Plots/HVI_beta.png",
            dpi=600,
            bbox_inches='tight')

# Third figure: dependence of the initial rotational velocity on the moment arm
plt.figure()
# Filter the dataset to the ones that are interesting
arm_HVI_data_arr = HVI_data_arr[HVI_data_arr[:, 5] == 0]   # Keep theta=0 for this analysis
arm_HVI_data_arr = arm_HVI_data_arr[arm_HVI_data_arr[:, 6] == 0]   # Keep phi=0 for this analysis
arm_HVI_data_arr = arm_HVI_data_arr[arm_HVI_data_arr[:, 3] == 0]  # x-axis only
arm_HVI_data_arr = arm_HVI_data_arr[arm_HVI_data_arr[:, 1] == 10000]   # Focus on collisions with Vp=10 km/s
for mp_id, mp in enumerate(projectile_mass_list):
    current_HVI_data_arr_arm = arm_HVI_data_arr[arm_HVI_data_arr[:, 0] == mp]

    current_HVI_data_arr_arm_beta_min = current_HVI_data_arr_arm[current_HVI_data_arr_arm[:, 2] == 1]
    current_HVI_data_arr_arm_beta_max = current_HVI_data_arr_arm[current_HVI_data_arr_arm[:, 2] == 5]

    current_HVI_data_arr_arm_beta_min = current_HVI_data_arr_arm_beta_min[
        current_HVI_data_arr_arm_beta_min[:, 4].argsort()]  # order by the moment arm
    current_HVI_data_arr_arm_beta_max = current_HVI_data_arr_arm_beta_max[
        current_HVI_data_arr_arm_beta_max[:, 4].argsort()]  # order by moment arm

    print(current_HVI_data_arr_arm_beta_min[:, 4])
    plt.plot(current_HVI_data_arr_arm_beta_min[:, 4], np.rad2deg(current_HVI_data_arr_arm_beta_min[:, 10]),
             color=colors_list[mp_id])
    plt.plot(current_HVI_data_arr_arm_beta_max[:, 4], np.rad2deg(current_HVI_data_arr_arm_beta_max[:, 10]),
             color=colors_list[mp_id])
    plt.fill_between(current_HVI_data_arr_arm_beta_max[:, 4],
                     np.rad2deg(current_HVI_data_arr_arm_beta_max[:, 10]),
                     np.rad2deg(current_HVI_data_arr_arm_beta_min[:, 10]),
                     color=colors_list[mp_id],
                     alpha=0.5,
                     label=r'$d_{p}$ = ' + f"{'{:0.0e}'.format(projectile_diameter[mp_id])} m")
plt.grid(True)
plt.xlabel(r'$r_{arm}$ [m]', fontsize=14)
plt.ylabel(r'$||\vec{\omega}_{0, \mathcal{B}}||$ [deg/s]', fontsize=14)
plt.axhline(y=0.01*np.sqrt(3), color='k', linestyle='--', label='Significance level')
plt.axhline(y=26, color='k', linestyle=':', label='Capability level')
plt.yscale('log')
plt.legend(ncol=1, prop={'size': 8})
plt.savefig("/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/HVI_Plots/HVI_proj_diam_arm.png",
            dpi=600,
            bbox_inches='tight')

# Fourth figure: dependence of the initial rotational velocity on the angles
orientation_HVI_data_arr = HVI_data_arr[HVI_data_arr[:, 3] == 0]     # Focus on the 'x' axis
orientation_HVI_data_arr = orientation_HVI_data_arr[orientation_HVI_data_arr[:, 0] == projectile_mass_list[-3]]     # 1 mm diameter
orientation_HVI_data_arr = orientation_HVI_data_arr[orientation_HVI_data_arr[:, 1] == 10000]    # 10 km/s
orientation_HVI_data_arr = orientation_HVI_data_arr[orientation_HVI_data_arr[:, 2] == 5]        # beta=5
orientation_HVI_data_arr = orientation_HVI_data_arr[orientation_HVI_data_arr[:, 4] == 7]   # Focus on the ones at the tip of the booms
f, ax = plt.subplots()
ax.set_xlabel(r'$\theta_{p, \mathcal{B}}$ [deg]', fontsize=14)
ax.set_ylabel(r'$\phi_{p, \mathcal{B}}$ [deg]', fontsize=14)
tpc = ax.tripcolor(np.rad2deg(orientation_HVI_data_arr[:, 5]),
                   np.rad2deg(orientation_HVI_data_arr[:, 6]),
                   np.rad2deg(orientation_HVI_data_arr[:, 10]),
                   shading='flat', cmap=my_cmap)
cbar = f.colorbar(tpc)
cbar.set_label(r'$||\vec{\omega}_{0, \mathcal{B}}||$ [deg/s]', rotation=270, labelpad=17, fontsize=14)
# Generate a grid for interpolation
grid_x, grid_y = np.mgrid[-180:180:10000j, -180:180:10000j]

# Interpolate the values on the grid
contour_data = orientation_HVI_data_arr#[orientation_HVI_data_arr[:, 8] != 0]
#contour_data = contour_data[contour_data[:, 9] != 0]
z = np.rad2deg(contour_data[:, 9]) #/contour_data[:, 8]   # omega_z/omega_x
grid_z = griddata((np.rad2deg(contour_data[:, 5]), np.rad2deg(contour_data[:, 6])), z, (grid_x, grid_y), method='linear')

CS = ax.contour(grid_x, grid_y, grid_z, 10, colors='k')
ax.clabel(CS, inline=1, fontsize=8)
plt.savefig("/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/HVI_Plots/HVI_col_geom.png",
            dpi=600,
            bbox_inches='tight')

# Fifth figure: linear momentum and velocity and mass
fig, ax1 = plt.subplots()
# Filter the dataset to the ones that are interesting
p_HVI_data_arr = HVI_data_arr[HVI_data_arr[:, 5] == 0]             # Keep theta=0 for this analysis
p_HVI_data_arr = p_HVI_data_arr[p_HVI_data_arr[:, 6] == 0]         # Keep phi=0 for this analysis
p_HVI_data_arr = p_HVI_data_arr[p_HVI_data_arr[:, 3] == 0]         # Focus on the 'x' axis
p_HVI_data_arr = p_HVI_data_arr[p_HVI_data_arr[:, 4] == 7]         # Focus on the ones at the tip of the booms
p_HVI_data_arr = p_HVI_data_arr[p_HVI_data_arr[:, 2] == 5]         # beta=5
for v_id, vel in enumerate(projectile_velocity_list_selected):
    current_HVI_data_arr_p = p_HVI_data_arr[np.where(p_HVI_data_arr[:, 1] == vel)[0], :]
    p = current_HVI_data_arr_p[:, 0] * current_HVI_data_arr_p[:, 1]   # Linear momentum of the considered particles

    reduced_arr = np.empty((len(p), 2))
    reduced_arr[:, 0] = current_HVI_data_arr_p[:, 0]
    reduced_arr[:, 1] = p
    reduced_arr = reduced_arr[reduced_arr[:, 0].argsort()]  # order by the linear momentum
    ax1.plot(reduced_arr[:, 0], reduced_arr[:, 1],
             color=colors_list[v_id], label=r'$||\vec{v}_{p}||$ =' + f'{vel/1000} km/s')
ax1.grid(True)
ax1.set_ylabel(r'$||\vec{p}_{p, \mathcal{B}}||$ [kg m/s]', fontsize=14)
ax1.set_xlabel(r'$m_{p}$ [kg]', fontsize=14)

f = lambda x: (x * (8 * 3/(4*np.pi*2700)))**(1/3)
g = lambda x: (4/(3*8))*np.pi*(x)**3 * 2700
ax2 = ax1.secondary_xaxis("top", functions=(f,g))
ax2.set_xlabel(r'$d_{p}$ [m]', fontsize=14)
ax1.axhline(y=0.136135*5, color='grey', linestyle='-')#, label=r'Capability level: $\beta_{E} = 1$'
ax1.axhline(y=0.136135, color='grey', linestyle='-')#, label=r'Capability level: $\beta_{E} = 5$'
ax1.axhline(y=9 * 10**(-5), color='k', linestyle='--', label=r'Minimum significance level ($\beta_{E} = 5$)')
ax1.fill_between(x=projectile_mass_list, y1=0.136135*5, y2=0.136135, color='grey',  interpolate=True, alpha=.5, label='Capability level range')

ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend(ncol=2, prop={'size': 8}, loc='lower right')
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1.set_xlim(10**(-16), 1.5)
fig.savefig("/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/HVI_Plots/HVI_p_v.png",
            dpi=600,
            bbox_inches='tight')
plt.show()

