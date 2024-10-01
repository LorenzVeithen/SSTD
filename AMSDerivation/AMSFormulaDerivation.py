import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.optimize import curve_fit
from scipy.linalg import lstsq
import pygmo as pg
import os
import itertools
from time import time
from matplotlib import cm

from constants import *
from generalConstants import AMS_directory
from MiscFunctions import compute_panel_geometrical_properties, sun_angles_from_sunlight_vector
from vaneControllerMethods import constrainedEllipseCoefficientsProblem, rotated_ellipse_coefficients_wrt_vane_1
from vaneControllerMethods import fourierSumFunction, fourierSeriesFunction, combinedFourierFitFunction, cart_to_pol, fit_2d_ellipse, get_ellipse_pts
from vaneControllerMethods import generate_AMS_data, vaneAnglesAllocationProblem, buildEllipseCoefficientFunctions, ellipseCoefficientFunction
from attitudeControllersClass import sail_attitude_control_systems
from sailCraftClass import sail_craft
from numpy.linalg import lstsq
from vaneControllerMethods import combinedFourierFitDesignMatrix
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, roc_auc_score
from matplotlib import rc

# activate latex text rendering
rc('text', usetex=True)

#import h5py

PLOT = True
COMPUTE_DATA = True
COMPUTE_ELLIPSES = True
COMPUTE_ELLIPSE_FOURIER_FORMULA = False
order_i = 15     # 8
order_m = 15     # 6
order_n = 15     # 6
cut_off = 0

SAVE_DATA = False
ALL_HULLS = False
FOURIER_ELLIPSE_AVAILABLE = True
GENERATE_FOURIER_TEST_DATA = False

if (GENERATE_FOURIER_TEST_DATA):
    coefficients_file_differentiator = '_test_data'
else:
    coefficients_file_differentiator = ''

# Define solar sail - see constants file
#vanes_optical_properties = [np.array([0., 0., 1., 1., 0., 0., 2/3, 2/3, 1., 1.])] * len(vanes_origin_list)
#vanes_optical_properties = [np.array([0., 0., 1., 0., 0., 0., 2/3, 2/3, 1., 1.])] * len(vanes_origin_list)
vanes_optical_properties = [np.array([0.1, 0.57, 0.74, 0.23, 0.16, 0.2, 2/3, 2/3, 0.03, 0.6])] * len(vanes_origin_list)
vane_optical_model_str = "ACS3_optical_model"
sh_comp = 1 # Define whether to work with or without shadow effects
vane_id = 1
cdir = f"{AMS_directory}/Datasets/{vane_optical_model_str}/vane_{vane_id}"
target_hull = "TyTz"

if (FOURIER_ELLIPSE_AVAILABLE):
    ellipse_coefficient_functions_list = []
    for i in range(6):
        filename = f'{AMS_directory}/Datasets/{vane_optical_model_str}/vane_1/dominantFitTerms/{["A", "B", "C", "D", "E", "F"][i]}_shadow_{bool(sh_comp)}.txt'
        built_function = buildEllipseCoefficientFunctions(filename)
        ellipse_coefficient_functions_list.append(lambda aps, bes, f=built_function: ellipseCoefficientFunction(aps,bes, f))

if (COMPUTE_DATA):
    if (sh_comp != 0 and sh_comp != 1):
        raise Exception(
            "Error.AMS generation for both shadow TRUE and FALSE is not supported in AMSFormulaDerivation." +
            "Do one after the other")

    acs_object = sail_attitude_control_systems("vanes", boom_list, sail_I, algorithm_constants)
    acs_object.set_vane_characteristics(vanes_coordinates_list,
                                    vanes_origin_list,
                                    vanes_rotation_matrices_list,
                                    0,
                                    np.array([0, 0, 0]),
                                    0.0045,
                                    vanes_rotational_dof,
                                    "AMS_Derivation",
                                    wings_coordinates_list,
                                    vane_mechanical_rotation_limits,
                                    vanes_optical_properties)

    sail = sail_craft("ACS3",
                      len(wings_coordinates_list),
                      len(vanes_coordinates_list),
                      wings_coordinates_list,
                      vanes_coordinates_list,
                      wings_optical_properties,
                      vanes_optical_properties,
                      sail_I,
                      sail_mass,
                      sail_mass_without_wings,
                      sail_nominal_CoM,
                      sail_material_areal_density,
                      sail_material_areal_density,
                      acs_object)

    vaneAngleProblem = vaneAnglesAllocationProblem(vane_id,
                                                   ([-np.pi, -np.pi], [np.pi, np.pi]),
                                                   10,
                                                   wings_coordinates_list,
                                                   acs_object,
                                                   include_shadow=True)
    vaneAngleProblem.update_vane_angle_determination_algorithm(np.array([0, 1, 0]), np.array([0, 0, -1]),
                                                               vane_variable_optical_properties=True, vane_optical_properties_list=vanes_optical_properties)  # and the next time you can put False

    sun_angles_num, vane_angles_num = 21, 100

    if (not GENERATE_FOURIER_TEST_DATA):
        sun_angle_alpha_list = np.linspace(-180, 180, sun_angles_num)
        sun_angle_beta_list = np.linspace(-180, 180, sun_angles_num)
    else:    # shift the grid and generate the same data
        sun_angle_alpha_list = np.linspace(-180+1, 180+1, sun_angles_num)[:-1]
        sun_angle_beta_list = np.linspace(-180+1, 180+1, sun_angles_num)[:-1]

    alpha_1_range = np.linspace(-np.pi, np.pi, vane_angles_num)
    alpha_2_range = np.linspace(-np.pi, np.pi, vane_angles_num)

t0 = time()-1
if (COMPUTE_ELLIPSES):
    # Extract data
    optimised_ellipse_coefficients_stack_without_shadow = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    optimised_ellipse_coefficients_stack_with_shadow = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    if (COMPUTE_DATA):
        loop_list = list(itertools.product(sun_angle_alpha_list, sun_angle_beta_list))
    else:
        all_files_list = os.listdir(cdir)
        number_files = len(all_files_list)
        loop_list = os.listdir(cdir)
    for i, input in enumerate(loop_list):
        print(time() - t0)
        t0 = time()
        if (COMPUTE_DATA):
            print(f'{100 * (i / len(list(loop_list)))} %')
            ams_data, shadow_list = generate_AMS_data(vane_id, vaneAngleProblem,
                                                  input[0],
                                                  input[1],
                                                  alpha_1_range,
                                                  alpha_2_range,
                                                  optical_model_str=vane_optical_model_str,
                                                  savefig=False, savedat=False,
                                                  shadow_computation=sh_comp)
            ams_data = ams_data[0]
        else:   # retrieve data from file
            print(f'{100 * (i / number_files)} %')
            if (os.path.isdir(cdir + "/" + input)):
                continue
            ams_data = np.genfromtxt(cdir + "/" + input, delimiter=',')
            shadow_list = []

        alpha_1_deg, alpha_2_deg = np.rad2deg(ams_data[:, 0]), np.rad2deg(ams_data[:, 1])
        alpha_sun_deg, beta_sun_deg = np.rad2deg(ams_data[0, 2]), np.rad2deg(ams_data[0, 3])
        alpha_s_rad, beta_s_rad = ams_data[0, 2], ams_data[0, 3]
        n_s = np.array([np.sin(alpha_s_rad) * np.cos(beta_s_rad),
                        np.sin(alpha_s_rad) * np.sin(beta_s_rad),
                        -np.cos(alpha_s_rad)])  # In the body reference frame
        point_in_shadow = ams_data[:, 4]    # watch out, for now this is only the torque magnitude

        Tx, Ty, Tz = ams_data[:, 5].T, ams_data[:, 6].T, ams_data[:, 7].T
        Tx, Ty, Tz = Tx[point_in_shadow == 0], Ty[point_in_shadow == 0], Tz[point_in_shadow == 0]   # Only select points which are not in the shadow
        Tx, Ty, Tz = Tx[..., None], Ty[..., None], Tz[..., None]

        # Compute complex hull
        match target_hull:
            case 'TxTy':
                AMS_points = np.hstack((Tx, Ty))
                numerical_hull = ConvexHull(AMS_points)
                numerical_hull_points = numerical_hull.points[numerical_hull.vertices]
                plot_labels = [r'$\tilde{T}_{x}$ [-]',
                               r'$\tilde{T}_{y}$ [-]']
            case 'TxTz':
                AMS_points = np.hstack((Tx, Tz))
                numerical_hull = ConvexHull(AMS_points)
                numerical_hull_points = numerical_hull.points[numerical_hull.vertices]
                plot_labels = [r'$\tilde{T}_{x}$ [-]',
                               r'$\tilde{T}_{z}$ [-]']
            case 'TyTz':
                AMS_points = np.hstack((Ty, Tz))
                numerical_hull = ConvexHull(AMS_points)
                numerical_hull_points = numerical_hull.points[numerical_hull.vertices]
                plot_labels = [r'$\tilde{T}_{y, \mathcal{B}}$ [-]',
                               r'$\tilde{T}_{z, \mathcal{B}}$ [-]']
            case 'TxyTz':
                AMS_points = np.hstack((np.sqrt(Tx**2 + Ty**2), Tz))
                numerical_hull = ConvexHull(AMS_points)
                numerical_hull_points = numerical_hull.points[numerical_hull.vertices]
                plot_labels = ['Txy [-]', 'Tz [-]']
            case 'TxTyTz':
                AMS_points = np.hstack((Tx, Ty, Tz))
                numerical_hull = ConvexHull(AMS_points)
                numerical_hull_points = numerical_hull.points[numerical_hull.vertices]
                plot_labels = ['TxTyTz [-]', 'Tz [-]']
            case _:
                raise Exception('No feasible numerical hull selected')

        # Fitted ellipse
        weights_chosen_hull_fit = fit_2d_ellipse(numerical_hull_points[:, 0], numerical_hull_points[:, 1])
        x0, y0, ap, bp, e, phi = cart_to_pol(weights_chosen_hull_fit)
        x_chosen_hull, y_chosen_hull = get_ellipse_pts((x0, y0, ap, bp, e, phi), npts=500)

        # Find the best ellipse circumscribed by the convex hull (and does not go over the boundary)
        # For TyTz of vane 1, then the coefficients will be reused for other vanes
        myConstrainedEllipseCoefficientsProblem = constrainedEllipseCoefficientsProblem(numerical_hull_points)

        A0 = myConstrainedEllipseCoefficientsProblem.ellipse_area(weights_chosen_hull_fit)
        myConstrainedEllipseCoefficientsProblem.sef_initial_area(A0)

        obj0 = myConstrainedEllipseCoefficientsProblem.fitness(weights_chosen_hull_fit)[0]
        worst_inequality_constraint0 = max(myConstrainedEllipseCoefficientsProblem.fitness(weights_chosen_hull_fit)[1:])

        prob = pg.problem(myConstrainedEllipseCoefficientsProblem)
        prob.c_tol = 0
        pop = pg.population(prob=prob)
        pop.push_back(x=weights_chosen_hull_fit)  # Defining the initial guess
        scp = pg.algorithm(pg.scipy_optimize(method="SLSQP"))
        scp.set_verbosity(1)
        result = scp.evolve(pop)

        optimised_ellipse_coefficients = pop.champion_x
        A_final = myConstrainedEllipseCoefficientsProblem.ellipse_area(pop.champion_x)
        obj_final = myConstrainedEllipseCoefficientsProblem.fitness(pop.champion_x)[0]
        worst_inequality_constraint_final = max(myConstrainedEllipseCoefficientsProblem.fitness(pop.champion_x)[1:])

        [x0, y0, ap, bp, e, phi] = cart_to_pol(pop.champion_x)
        x_opt, y_opt = get_ellipse_pts([x0, y0, ap, bp, e, phi], npts=1000)

        if (abs(A_final) > abs(A0)):
            print("Final area is larger than the initial fit")
        if (not prob.feasibility_x(pop.champion_x)):
            print("Constraint violation")
        if ((abs(A_final) > abs(A0)) or (not prob.feasibility_x(pop.champion_x))):
            print(input)
            print(abs(A_final), abs(A0))
            print(worst_inequality_constraint_final)

        current_stack = np.hstack((np.deg2rad(alpha_sun_deg), np.deg2rad(beta_sun_deg), optimised_ellipse_coefficients))
        if (COMPUTE_DATA):
            array_storage_selection = str(bool(sh_comp))
        else:
            array_storage_selection = input[-8:-4]
        if (array_storage_selection == 'True'):
            # Store ellipse coefficients and write file
            optimised_ellipse_coefficients_stack_with_shadow = np.vstack((optimised_ellipse_coefficients_stack_with_shadow,
                                                              current_stack))
        else:
            # Store ellipse coefficients and write file
            optimised_ellipse_coefficients_stack_without_shadow = np.vstack((optimised_ellipse_coefficients_stack_without_shadow,
                                                              current_stack))

        if (PLOT):
            if (FOURIER_ELLIPSE_AVAILABLE):
                fourier_ellipse_coefficient = []
                R_VBi = np.linalg.inv(vanes_rotation_matrices_list[vane_id])
                current_alpha_s, current_beta_s = sun_angles_from_sunlight_vector(R_VBi, n_s)
                for func in ellipse_coefficient_functions_list:
                    fourier_ellipse_coefficient.append(func(current_alpha_s, current_beta_s))
                A, B, C, D, E, F = fourier_ellipse_coefficient

                Tx_tuple, Ty_tuple = rotated_ellipse_coefficients_wrt_vane_1(R_VBi, (A, B, C, D, E, F))
                if (target_hull == 'TxTz'):
                    (A, B, C, D, E, F) = Tx_tuple
                elif (target_hull == 'TyTz'):
                    (A, B, C, D, E, F) = Ty_tuple

                params = cart_to_pol((A, B, C, D, E, F))
                x_fourier, y_fourier = get_ellipse_pts(params, npts=250)

            # Plots
            plt.figure()
            plt.grid(True)
            plt.scatter(AMS_points[:, 0], AMS_points[:, 1], s=1, color='grey', label="Vane points")
            for simplex in numerical_hull.simplices:
                plt.plot(AMS_points[simplex, 0], AMS_points[simplex, 1], 'k--', zorder=1*5)
            plt.plot([], [], 'k--', label="Convex Hull")
            plt.plot(x_chosen_hull, y_chosen_hull, 'g', linewidth=2, linestyle=':', label="Fitted ellipse", zorder=2*5)
            plt.plot(x_opt, y_opt, 'b', linestyle='-.', label="Optimised ellipse", zorder=4*5)
            if (FOURIER_ELLIPSE_AVAILABLE):
                plt.plot(x_fourier, y_fourier, color='r', label="Fourier ellipse", zorder=3*5)
            plt.scatter(numerical_hull_points[:, 0], numerical_hull_points[:, 1], color='orange', s=15, label='Hull points')
            plt.xlabel(plot_labels[0], fontsize=14)
            plt.ylabel(plot_labels[1], fontsize=14)
            #plt.title(r"$\alpha_{s}$=" + f'{round(alpha_sun_deg, 1)}°' + r" and $\beta_{s}$=" + f'{round(beta_sun_deg, 1)}°')
            plt.legend()
            if (not FOURIER_ELLIPSE_AVAILABLE):
                if (not COMPUTE_DATA):
                    plt.savefig(f"{AMS_directory}/Plots/{vane_optical_model_str}/vane_{vane_id}/Fitted_Ellipses/{input[:-4]}_{vane_optical_model_str}_hull_{target_hull}.png",
                               bbox_inches='tight', dpi=600)
                else:
                    plt.savefig(f"{AMS_directory}/Plots/{vane_optical_model_str}/vane_{vane_id}/Fitted_Ellipses/AMS_{vane_optical_model_str}_alpha_{round(alpha_sun_deg, 1)}_beta_{round(beta_sun_deg, 1)}_shadow_{str(bool(sh_comp))}_hull_{target_hull}.png",
                                bbox_inches='tight', dpi=600
                                )

            else:
                if (not COMPUTE_DATA):
                    plt.savefig(f"{AMS_directory}/Plots/{vane_optical_model_str}/vane_{vane_id}/Fourier_Ellipses/{input[:-4]}_{vane_optical_model_str}_hull_{target_hull}.png",
                                bbox_inches='tight', dpi=600
                                )
                else:
                    plt.savefig(
                        f"{AMS_directory}/Plots/{vane_optical_model_str}/vane_{vane_id}/Fourier_Ellipses/AMS_{vane_optical_model_str}_alpha_{round(alpha_sun_deg, 1)}_beta_{round(beta_sun_deg, 1)}_shadow_{str(bool(sh_comp))}_hull_{target_hull}.png",
                        bbox_inches='tight', dpi=600
                    )
            #plt.show()
            plt.close()

    if (len(np.shape(optimised_ellipse_coefficients_stack_with_shadow)) == 1):
        optimised_ellipse_coefficients_stack_with_shadow = optimised_ellipse_coefficients_stack_with_shadow[..., None]
    if (len(np.shape(optimised_ellipse_coefficients_stack_without_shadow)) == 1):
        optimised_ellipse_coefficients_stack_without_shadow = optimised_ellipse_coefficients_stack_without_shadow[..., None]
    optimised_ellipse_coefficients_stack_with_shadow = optimised_ellipse_coefficients_stack_with_shadow[1:, :]
    optimised_ellipse_coefficients_stack_without_shadow = optimised_ellipse_coefficients_stack_without_shadow[1:, :]

    if (SAVE_DATA):
        if (np.shape(optimised_ellipse_coefficients_stack_with_shadow)[0] > 1e2):
            np.savetxt(
                cdir + f"/ellipseCoefficients/" + f"AMS_ellipse_coefficients_shadow_True_hull_{target_hull}{coefficients_file_differentiator}.csv",
                optimised_ellipse_coefficients_stack_with_shadow, delimiter=",",
                header='alpha_sun, beta_sun, A, B, C, D, E, F')

        if (np.shape(optimised_ellipse_coefficients_stack_without_shadow)[0] > 1e2):
            np.savetxt(
                cdir + f"/ellipseCoefficients/" + f"AMS_ellipse_coefficients_shadow_False_hull_{target_hull}{coefficients_file_differentiator}.csv",
                optimised_ellipse_coefficients_stack_without_shadow, delimiter=",",
                header='alpha_sun, beta_sun, A, B, C, D, E, F')


if (COMPUTE_ELLIPSE_FOURIER_FORMULA):
    ellipse_coefficients_data = np.genfromtxt(
        cdir + f"/ellipseCoefficients/" + f"AMS_ellipse_coefficients_shadow_{bool(sh_comp)}_hull_{target_hull}.csv", delimiter=',')
    sorted_indices = np.lexsort((ellipse_coefficients_data[:, 1], ellipse_coefficients_data[:, 0]))
    ellipse_coefficients_data = ellipse_coefficients_data[sorted_indices]

    alpha_sun_deg = ellipse_coefficients_data[:, 0]
    beta_sun_deg = ellipse_coefficients_data[:, 1]

    sun_angles = np.stack([alpha_sun_deg, beta_sun_deg], axis=1)
    Y = ellipse_coefficients_data[:, 2:]  # Take all the ellipse coefficients

    current_fit_coefficients = []

    num_coefficients = (order_i * 2 - 1) ** 2 - (order_i - 2) * 2 + (
                ((order_n) * (order_m) - 1) * 4 * (order_n > 1 or order_m > 1))

    M = combinedFourierFitDesignMatrix(sun_angles, *[1] * (num_coefficients), order=order_i, order_n=order_n,
                                       order_m=order_m)[0]
    # We perform singular-value decomposition of M https://stackoverflow.com/questions/18452633/how-do-i-associate-which-singular-value-corresponds-to-what-entry
    # significant_entries = small_singular_value_entries(M, threshold=0)
    for i in range(np.shape(Y)[1]):
        # popt, pcov = curve_fit(fourierSeriesFunc, xdata=sun_angles, ydata=Y[:, i], p0=[1] * (num_coefficients))
        coefficients, residuals, rank, singular_values = lstsq(M, ellipse_coefficients_data[:, i + 2], rcond=1e-10)
        # Ridge regression with cross-validation
        ridge = Ridge(alpha=1.0)  # alpha is the regularization strength

        # KFold cross-validator with shuffling
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        scores = cross_val_score(ridge, M, ellipse_coefficients_data[:, i + 2], cv=kf, scoring='neg_mean_squared_error')  # 5-fold cross-validation
        ridge.fit(M, ellipse_coefficients_data[:, i + 2])
        coefficients = ridge.coef_
        coefficients[0] = ridge.intercept_

        print(f"Cross-validated scores: {scores}")
        print(f"Mean score: {np.mean(scores)}")
        print(f"STD score: {np.std(scores)}")
        current_fit_coefficients.append(coefficients)

    # Evaluate the fit quality - start with A
    fourier_ellipse_coefficients = np.zeros(np.shape(ellipse_coefficients_data[:, :]))
    fourier_ellipse_coefficients[:, 0] = alpha_sun_deg
    fourier_ellipse_coefficients[:, 1] = beta_sun_deg

    avg_term_magnitude, terms_list = combinedFourierFitFunction(sun_angles, *([1] * num_coefficients), order=order_i,
                                                                order_n=order_n, order_m=order_m)[1:3]

    stacked_terms = np.vstack((avg_term_magnitude, np.array(terms_list)), dtype=object)
    list_dominance_fit_terms = []
    for i in range(6):
        print(['A', 'B', 'C', 'D', 'E', 'F'][i])
        new_stacked_terms = np.copy(stacked_terms)
        new_stacked_terms = np.vstack((current_fit_coefficients[i], new_stacked_terms), dtype=object)
        new_stacked_terms[1, :] = new_stacked_terms[1, :] * current_fit_coefficients[i]

        fourier_ellipse_coefficients[:, 2 + i] = np.dot(M[:, abs(new_stacked_terms[1, :])>cut_off],
                                                        current_fit_coefficients[i][abs(new_stacked_terms[1, :]) > cut_off])
        new_stacked_terms = new_stacked_terms[:, abs(new_stacked_terms[1, :]) > cut_off]
        #  combinedFourierFitFunction(sun_angles, *current_fit_coefficients[i], order=order_i, order_n=order_n, order_m=order_m)[0]
        list_dominance_fit_terms.append(new_stacked_terms[:, (-abs(new_stacked_terms[1])).argsort()])

    for i, dominance_terms in enumerate(list_dominance_fit_terms):
        np.savetxt(
            f"{AMS_directory}/Datasets/{vane_optical_model_str}/vane_1/dominantFitTerms/{['A', 'B', 'C', 'D', 'E', 'F'][i]}_shadow_{bool(sh_comp)}.txt",
            dominance_terms.T, delimiter=',', fmt='%s')

    # Sort the indices by alpha and beta to ensure a fair comparison
    sorted_indices = np.lexsort((fourier_ellipse_coefficients[:, 1], fourier_ellipse_coefficients[:, 0]))
    fourier_ellipse_coefficients = fourier_ellipse_coefficients[sorted_indices]
    print(np.shape(fourier_ellipse_coefficients))
    area_diff_rel = []
    for i in range(len(fourier_ellipse_coefficients[:, 0])):
        a, b, c, d, e, f = fourier_ellipse_coefficients[i, 2:]
        _, _, ap, bp, _, _ = cart_to_pol(fourier_ellipse_coefficients[i, 2:])
        A_fourier = np.pi * ap * bp
        _, _, ap, bp, _, _ = cart_to_pol(ellipse_coefficients_data[i, 2:])
        A_real = np.pi * ap * bp
        area_diff_rel.append(100 * (A_fourier - A_real) / A_real)
    area_diff_rel = np.array(area_diff_rel)

    n_data_points = len(alpha_sun_deg)
    x = np.reshape(np.rad2deg(alpha_sun_deg), (int(np.sqrt(n_data_points)), int(np.sqrt(n_data_points))))
    y = np.reshape(np.rad2deg(beta_sun_deg), (int(np.sqrt(n_data_points)), int(np.sqrt(n_data_points))))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.3,
            alpha=0.2)
    my_cmap = plt.get_cmap('jet')
    ax.scatter(np.rad2deg(alpha_sun_deg), np.rad2deg(beta_sun_deg), area_diff_rel,
               c=area_diff_rel, cmap=my_cmap, alpha=0.5, label="Data")
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    ax.set_zlabel(f"Relative area difference, %")

    for i in range(6):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.grid(b=True, color='grey',
                linestyle='-.', linewidth=0.3,
                alpha=0.2)
        my_cmap = plt.get_cmap('jet')
        z = np.reshape(fourier_ellipse_coefficients[:, i + 2],
                       (int(np.sqrt(n_data_points)), int(np.sqrt(n_data_points))))
        ax.plot_surface(x, y, z, cmap=cm.coolwarm, alpha=0.5, label="Fourier fit")
        ax.scatter(np.rad2deg(alpha_sun_deg), np.rad2deg(beta_sun_deg), ellipse_coefficients_data[:, i + 2],
                   c=ellipse_coefficients_data[:, i + 2], cmap=my_cmap, alpha=0.5, label="Data")
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')
        ax.set_zlabel(f"Ellipse coefficient {['A', 'B', 'C', 'D', 'E', 'F'][i]}")
        ax.legend()
        plt.close()

    for i in range(6):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')


        z = (fourier_ellipse_coefficients[:, i + 2] - ellipse_coefficients_data[:, i + 2])
        id = z.argmax(axis=0)
        ax.scatter(np.rad2deg(alpha_sun_deg), np.rad2deg(beta_sun_deg), z,
                   c=z, cmap=my_cmap, alpha=0.5, label="Data")
        ax.set_zlabel(f'{["A", "B", "C", "D", "E", "F"][i]} absolute error')
        ax.set_title(f'{np.mean(abs(ellipse_coefficients_data[:, i + 2]))}')

    plt.show()
    #plt.close()