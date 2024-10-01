import numpy as np
import matplotlib
from numba import jit
from scipy.spatial import Delaunay
from scipy.optimize import golden, direct, minimize
from scipy.interpolate import make_interp_spline, PPoly
import matplotlib.pyplot as plt
import pygmo as pg

#from constants import *
from generalConstants import AMS_directory
from generalConstants import default_ellipse_bounding_box_margin
from ACS_dynamicalModels import vane_dynamical_model
from MiscFunctions import *
import itertools
from time import time

#matplotlib.pyplot.switch_backend('Agg')

class vaneAnglesAllocationProblem:
    # Solving the angle allocation problem of the vanes for a single vane
    def __init__(self, vane_id, bounds, num_vane_edge_mesh_nodes, sail_wings_coordinates, acs_object, include_shadow=True):
        self.vane_id = vane_id
        self.bounds = bounds    # [[-np.pi, np.pi], [-np.pi, np.pi]]
        self.vane_target_torque = None
        self.unit_direction_vane_target_torque = None
        self.sun_direction_body_frame = None
        self.sail_wings_coordinates = sail_wings_coordinates
        self.include_shadow = include_shadow
        self.hull = None

        self.R_BV = acs_object.vane_reference_frame_rotation_matrix_list[self.vane_id]

        self.vane_origin = acs_object.vane_reference_frame_origin_list[self.vane_id]
        self.vane_nominal_coordinates = acs_object.vane_panels_coordinates_list[self.vane_id]

        # Vane optical properties
        self.alpha_front = None
        self.alpha_back = None
        self.rho_s_front = None
        self.rho_s_back = None
        self.rho_d_front = None
        self.rho_d_back = None
        self.B_front = None
        self.B_back = None
        self.emissivity_front = None
        self.emissivity_back = None
        self.absorption_reemission_ratio = None

        # Mesh the vane edges for shadow determination
        all_meshed_points = np.array([[0, 0, 0]])
        for i in range(1, np.shape(self.vane_nominal_coordinates)[0] + 1):
            if (i == np.shape(self.vane_nominal_coordinates)[0]):
                delta_vec = self.vane_nominal_coordinates[0, :] - self.vane_nominal_coordinates[i - 1, :]
            else:
                delta_vec = self.vane_nominal_coordinates[i, :] - self.vane_nominal_coordinates[i - 1, :]
            meshed_points = np.zeros((num_vane_edge_mesh_nodes + 1, 3))

            for j in range(num_vane_edge_mesh_nodes + 1):
                meshed_points[j, :] = self.vane_nominal_coordinates[i - 1, :] + j * delta_vec / num_vane_edge_mesh_nodes
            all_meshed_points = np.vstack((all_meshed_points, meshed_points))
        self.meshed_vane_coordinates = all_meshed_points[1:, :]

    def fitness(self, x):
        """
        Fitness function to be minimised.
        :param x: variables to be optimised. x[0]: rotation around vane x-axis in radians; x[1]: rotation around vane
        y-axis in radiants.
        :return: list containing the objective function and equality constraints.
        """
        torque_x = self.single_vane_torque(x)
        obj = [(1./3.) * np.sum((torque_x - self.vane_target_torque)**2)]
        return obj

    def get_bounds(self):
        """
        Function returning the bounds on the variables to be optimised.
        :return: self.bounds, a list of two lists: the lower and upper bounds on each variable to be optimised
        """
        return self.bounds

    def get_nec(self):
        """
        Function returning the number of equality constraints of the optimisation
        :return: the number of equality constraints (int)
        """
        return 0

    def get_nic(self):
        """
        Function returning the number of inequality constraints of the optimisation
        :return: the number of inequality constraints (int)
        """
        return 0

    def get_name(self):
        return "Vane angles allocation problem"


    def single_vane_torque(self, x):
        """
        Torque performed by the vane alone, in the body frame. Therefore, even for zero rotation angles, the torque will
        be non-zero.
        :param x: variables to be optimised. x[0]: rotation around vane x-axis in radians; x[1]: rotation around vane
        y-axis in radians.
        :return: Torque in the body frame, or a np.array([1e23, 1e23, 1e23]) if the vane is casting some or is in
         the shadow.
        """
        rotated_points_body_frame = vane_dynamical_model([np.rad2deg(x[0])],
                                                         [np.rad2deg(x[1])],
                                                         1,
                                                         [self.vane_origin],
                                                         [self.meshed_vane_coordinates],
                                                         [self.R_BV])[0]

        centroid_body_frame, vane_area, surface_normal_body_frame = compute_panel_geometrical_properties(
            rotated_points_body_frame)  # This is all in the body frame
        c_theta = np.dot(surface_normal_body_frame, -self.sun_direction_body_frame) / (
                    np.linalg.norm(surface_normal_body_frame) * np.linalg.norm(self.sun_direction_body_frame))

        # Get the vane torque according to the optical model, in the body frame
        if (c_theta >= 0):  # the front is exposed
            # W * vane_area/ c_sol *
            f = (abs(c_theta)) * ((
              self.alpha_front * self.absorption_reemission_ratio - 2 * self.rho_s_front * c_theta - self.rho_d_front * self.B_front) * surface_normal_body_frame + (
              self.alpha_front + self.rho_d_front) * self.sun_direction_body_frame)
        else:
            # W * vane_area/ c_sol *
            f = (abs(c_theta)) * ((
              self.alpha_back * self.absorption_reemission_ratio - 2 * self.rho_s_back * c_theta + self.rho_d_back * self.B_back) * surface_normal_body_frame + (
              self.alpha_back + self.rho_d_back) * self.sun_direction_body_frame)

        force_on_vane_body_frame = f
        torque_on_body_from_vane = (1/np.linalg.norm(self.vane_origin)) * np.cross(centroid_body_frame, force_on_vane_body_frame)
        result = torque_on_body_from_vane
        if (self.include_shadow):
            shadow_bool = self.vane_shadow(rotated_points_body_frame[2:-1, :],
                                           self.hull)  # The hull is updated at each iteration of the propagation
            if (shadow_bool):
                result = np.array([1e23, 1e23, 1e23])  # Penalty if in shadow
        return result   # Non-dimensional, physical torque is given when multiplicating by norm(r_arm) * W * vane_area/c_sol

    def update_vane_angle_determination_algorithm(self, Td, n_s, vane_variable_optical_properties=False, vane_optical_properties_list=[None]):
        """
        Function permitting to update the properties of the class object at each iteration of the propagation to target
        different target torques under different sunlight conditions or vane optical properties.
        :param Td: np.array[(1, 3)], target torque in Nm at the time of evaluating the optimisation.
        :param n_s: np.array[(1, 3)], unit vector of the sunlight direction in the body fixed frame.
        :param vane_variable_optical_properties: boolean, indicating if the optical properties of the vanes can change
        as a function of time.
        :return: True, if the process was completed successfully
        """
        # Target torque
        self.vane_target_torque = Td

        # Sun direction in the body frame
        self.sun_direction_body_frame = n_s

        # Spacecraft shadow hull
        vstack_stacking = np.array([[0, 0, 0]])
        for wing in self.sail_wings_coordinates:
            vstack_stacking = np.vstack((vstack_stacking, wing))
        relative_sun_vector_same_shape = np.zeros(np.shape(vstack_stacking[1:, :]))
        relative_sun_vector_same_shape[:, :3] = self.sun_direction_body_frame
        total_hull = np.vstack((vstack_stacking[1:, :] + relative_sun_vector_same_shape * 20,
                                vstack_stacking[1:, :] - relative_sun_vector_same_shape * 20))

        if (abs(n_s[2]) < 1e-15):   # Coplanar hull, no real shadow
            self.sunlight_is_in_sail_plane = True
            self.hull = total_hull
        else:
            self.sunlight_is_in_sail_plane = False
            self.hull = Delaunay(total_hull)

        self.total_hull = total_hull
        self.all_wing_points = vstack_stacking[1:, :]

        # Surface optical properties
        if (vane_variable_optical_properties):
            vane_optical_properties = vane_optical_properties_list[self.vane_id] #self.sail_craft.get_ith_panel_optical_properties(self.vane_id, "Vane")
            self.alpha_front = vane_optical_properties[0]
            self.alpha_back = vane_optical_properties[1]
            self.rho_s_front = vane_optical_properties[2]
            self.rho_s_back = vane_optical_properties[3]
            self.rho_d_front = vane_optical_properties[4]
            self.rho_d_back = vane_optical_properties[5]
            self.B_front = vane_optical_properties[6]
            self.B_back = vane_optical_properties[7]
            self.emissivity_front = vane_optical_properties[8]
            self.emissivity_back = vane_optical_properties[9]
            self.absorption_reemission_ratio = (self.emissivity_back * self.B_back - self.emissivity_front * self.B_front) / (
                                                       self.emissivity_back + self.emissivity_front)
        return True

    def set_shadow_bool(self, sb):
        self.include_shadow = sb
        return True

    def vane_shadow(self, p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        if (self.sunlight_is_in_sail_plane):
            return False
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)

        return any(hull.find_simplex(p) >= 0)

    def plot_shadow_hull(self):
        """
        Simple 3D plot of the shadow hull
        :return: ax, the axis object permitting to plot other items in the figure
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.total_hull[:, 0], self.total_hull[:, 1], self.total_hull[:, 2])
        ax.scatter(self.all_wing_points[:, 0], self.all_wing_points[:, 1], self.all_wing_points[:, 2])
        for simplex in self.hull.simplices:
            plt.plot(self.total_hull[simplex, 0], self.total_hull[simplex, 1], self.total_hull[simplex, 2], 'k-')
        return ax


def vaneAngleAllocationScaling(t, desired_torque, n_s, vaneAngleProblem, vane_angle_determination_bounds, tol_global_search, tol_golden_search):
    scaled_desired_torque = desired_torque * t
    vaneAngleProblem.update_vane_angle_determination_algorithm(scaled_desired_torque, n_s)
    fit_function = lambda x: vaneAngleProblem.fitness(x)[0]
    optRes = direct(fit_function, bounds=vane_angle_determination_bounds,
                    len_tol=tol_global_search)
    obtainedFitness = vaneAngleProblem.fitness([optRes.x[0], optRes.x[1]])[0]
    if (obtainedFitness < tol_golden_search):
        return 1, optRes
    else:
        return -1, optRes


class vaneTorqueAllocationProblem:
    def __init__(self,
                 acs_object,
                 sail_wings_coordinates,
                 vane_has_ideal_model,
                 include_shadow,
                 vanes_AMS_coefficient_functions,
                 vanes_optical_properties,
                 w1=2./3.,
                 w2=1./3.,
                 num_shadow_mesh_nodes=10):
        self.acs_object = acs_object    # Object with all vane characteristics
        self.sail_wings_coordinates = sail_wings_coordinates
        self.vane_has_ideal_model = vane_has_ideal_model
        self.include_shadow = include_shadow
        self.vanes_AMS_coefficient_functions = vanes_AMS_coefficient_functions
        self.w_1, self.w_2 = w1, w2
        self.previous_x = None
        self.current_torque = None
        self.desired_torque = None
        self.default_vane_torque_body_frame = None
        self.bounds = ([-100] * (self.acs_object.number_of_vanes * 3), [100] * (self.acs_object.number_of_vanes * 3))
        self.scaling_list = []
        for vid in range(len(self.acs_object.vanes_areas_list)):
            self.scaling_list.append(self.acs_object.vanes_areas_list[vid] * np.linalg.norm(self.acs_object.vane_reference_frame_origin_list[vid]))


        self.vane_angle_problem_objects_list = []
        for vane_id in range(self.acs_object.number_of_vanes):
            vaneAngleProblem_obj_i = vaneAnglesAllocationProblem(vane_id,
                                        self.acs_object.vane_mechanical_rotation_limits,
                                        num_shadow_mesh_nodes,
                                        self.sail_wings_coordinates,
                                        self.acs_object,
                                        include_shadow=self.include_shadow)
            vaneAngleProblem_obj_i.update_vane_angle_determination_algorithm(np.array([0, 0, 0]), np.array([0, 0, 0]),
                                                                           vane_variable_optical_properties=True, vane_optical_properties_list=vanes_optical_properties)
            self.vane_angle_problem_objects_list.append(vaneAngleProblem_obj_i)

        # Determine if the degrees of freedom and position of each vane allows a moment around the x, y, or z axis
        # Make an array of booleans with True if the related torque should be equal to zero
        self.eq_constraints_vane_capabilities = np.array([0] * (self.acs_object.number_of_vanes * 3))
        for i in range(self.acs_object.number_of_vanes):
            if (not self.acs_object.vanes_rotational_dof_booleans[i][0]):       # DoF constraints
                self.eq_constraints_vane_capabilities[i * 3 + 2] = 2            # Body torque around Z is not possible

            if (not self.acs_object.vanes_rotational_dof_booleans[i][1]):       # DoF constraints
                # Only certain combinations of torques in Z and X/Y are feasible
                self.eq_constraints_vane_capabilities[i * 3] = 2                # X
                self.eq_constraints_vane_capabilities[i * 3 + 1] = 2            # Y

            # A torque around X, Y or both is possible based on the DoF.
            # Check with the vane orientation (position will be taken care of with the attaignable moment set)
            idx = np.where(abs(abs(np.linalg.inv(self.acs_object.vane_reference_frame_rotation_matrix_list[i])[:, 1]) - 1) < 1e-15)[0]
            if (len(idx) !=0):
                if (idx[0] == 0):
                    self.eq_constraints_vane_capabilities[i * 3 + 1] = 1     # Torque fully around X, no torque around Y
                elif (idx[0] == 1):
                    self.eq_constraints_vane_capabilities[i * 3] = 1         # Torque fully around Y, no torque around X
                # Else, no constraint
    def fitness(self, x):
        # x is 3 * self.acs_object.number_of_vanes long array with the x-y-z body torques for each vane in the same
        # order as the rest of the vane lists
        self.current_torque = x.reshape((self.acs_object.number_of_vanes, 3)).sum(axis=0)
        obj = self.w_1 * (np.linalg.norm(self.current_torque - self.desired_torque) ** 2) + self.w_2 * (np.linalg.norm(x - self.previous_x) ** 2)

        eq_constraint_list = []
        # The total torque needs to be the desired torque - became an objective
        if (self.w_1 == 0):
            eq_constraint_list.append((self.current_torque - self.desired_torque)[0])  # = 0
            eq_constraint_list.append((self.current_torque - self.desired_torque)[1])  # = 0
            eq_constraint_list.append((self.current_torque - self.desired_torque)[2])  # = 0
        else:
            if (np.linalg.norm(self.desired_torque)>1e-15 and np.linalg.norm(self.current_torque)>1e-15):
                # the torque selected should be in the desired torque direction
                current_torque_direction = self.current_torque/np.linalg.norm(self.current_torque)
                desired_torque_direction = self.desired_torque / np.linalg.norm(self.desired_torque)
                eq_constraint_list.append((current_torque_direction - desired_torque_direction)[0])  # = 0
                eq_constraint_list.append((current_torque_direction - desired_torque_direction)[1])  # = 0
                eq_constraint_list.append((current_torque_direction - desired_torque_direction)[2])  # = 0
            else:

                eq_constraint_list.append((self.current_torque - self.desired_torque)[0])  # = 0
                eq_constraint_list.append((self.current_torque - self.desired_torque)[1])  # = 0
                eq_constraint_list.append((self.current_torque - self.desired_torque)[2])  # = 0

        # Degree of freedom limitations and constraints based on the orientation of the vanes
        eq_constraint_list = eq_constraint_list + list(x[self.eq_constraints_vane_capabilities == 1])

        # Ellipse inequality constraints and
        ineq_constraint_list = []
        for vane_id, (vane_AMS_coeff_x, vane_AMS_coeff_y) in enumerate(zip(self.vanes_AMS_coefficients_x, self.vanes_AMS_coefficients_y)):
            # Get torque on the current vane
            T_current_vane = x.reshape((self.acs_object.number_of_vanes), 3)[vane_id, :]
            Tx = T_current_vane[0]
            Ty = T_current_vane[1]
            Tz = T_current_vane[2]

            if (self.acs_object.vanes_rotational_dof_booleans[vane_id][0] and self.acs_object.vanes_rotational_dof_booleans[vane_id][1]):
                # Given vane has two degrees of freedom, therefore the AMS is necessary
                if (self.eq_constraints_vane_capabilities[vane_id * 3] != 1):
                    if (self.eq_constraints_vane_capabilities[vane_id * 3] == 0):  # is an inequality constraint
                        ineq_constraint_list.append(np.dot(np.array([[Tx ** 2, Tx * Tz, Tz ** 2, Tx, Tz, 1]]), vane_AMS_coeff_x)[0])

                if (self.eq_constraints_vane_capabilities[vane_id * 3 + 1] != 1):
                    if (self.eq_constraints_vane_capabilities[vane_id * 3 + 1] == 0):  # is an inequality constraint
                        ineq_constraint_list.append(np.dot(np.array([[Ty ** 2, Ty * Tz, Tz ** 2, Ty, Tz, 1]]), vane_AMS_coeff_y)[0])

            elif (self.acs_object.vanes_rotational_dof_booleans[vane_id][0]==False and self.acs_object.vanes_rotational_dof_booleans[vane_id][1]==False):
                Tdefault = self.default_vane_torque_body_frame[vane_id]
                eq_constraint_list.append((Tx - Tdefault[0]))   # = 0
                eq_constraint_list.append((Ty - Tdefault[1]))   # = 0
                eq_constraint_list.append((Tz - Tdefault[2]))   # = 0

            else:
                list_alpha_tuples = []
                if (self.acs_object.vanes_rotational_dof_booleans[vane_id][0] == False):
                    # No rotation around the vane x-axis, alpha_1 is constrained to zero
                    (fTx_list_alpha_1, fTy_list_alpha_1, fTz_list_alpha_1) = self.vanes_spline_constraints_alpha_1_list[
                        vane_id]
                    list_alpha_tuples.append((fTx_list_alpha_1, fTy_list_alpha_1, fTz_list_alpha_1))
                if (self.acs_object.vanes_rotational_dof_booleans[vane_id][1] == False):
                    # No rotation around the vane x-axis, alpha_1 is constrained to zero
                    (fTx_list_alpha_2, fTy_list_alpha_2, fTz_list_alpha_2) = self.vanes_spline_constraints_alpha_2_list[
                        vane_id]
                    list_alpha_tuples.append((fTx_list_alpha_2, fTy_list_alpha_2, fTz_list_alpha_2))

                for alpha_tuple in list_alpha_tuples:
                    (fTx_list_alpha, fTy_list_alpha, fTz_list_alpha) = alpha_tuple
                    if (self.eq_constraints_vane_capabilities[vane_id * 3] == 1):
                        # Torque fully around Y, no torque around X. A constraint has already been placed on that
                        # Only give an additional equality constraint on Y
                        current_abscissa_ = [Ty]
                        abscissa_list_ = [fTy_list_alpha]
                        image_list_ = [fTz_list_alpha]
                    elif (self.eq_constraints_vane_capabilities[vane_id * 3 + 1] == 1):
                        # Torque fully around X, no torque around Y. A constraint has already been placed on that
                        # Only give an additional equality constraint on X
                        current_abscissa_ = [Tx]
                        abscissa_list_ = [fTx_list_alpha]
                        image_list_ = [fTz_list_alpha]
                    else:
                        # Put a constraint on both X and Y
                        current_abscissa_ = [Tx, Ty]
                        abscissa_list_ = [fTx_list_alpha, fTy_list_alpha]
                        image_list_ = [fTz_list_alpha, fTz_list_alpha]

                    for current_abscissa, abscissa_list, image_list in zip(current_abscissa_, abscissa_list_, image_list_):
                        min_distance_to_spline = 1E23
                        current_equality_constraint = 0
                        for s_a, s_b in zip(abscissa_list, image_list):
                            k_int = min(len(s_a.t) - 1, 3)  # aim for cubicsplines but go smaller if necessary
                            u0 = PPoly.from_spline((s_a.t, s_a.c - current_abscissa, k_int), extrapolate=False).roots()
                            if (len(u0) != 0):
                                min_distance_to_spline = min(min(abs(s_b(u0) - Tz * np.ones_like(u0))), min_distance_to_spline)
                                if (len(np.where(abs(s_b(u0) - Tz) == min_distance_to_spline)[0]) != 0):
                                    current_equality_constraint = \
                                    (s_b(u0) - Tz)[abs(s_b(u0) - Tz) == min_distance_to_spline][0]
                        eq_constraint_list.append(current_equality_constraint)  # add to the equality constraints list
        fitness_array = [obj] + eq_constraint_list + ineq_constraint_list
        return fitness_array

    def get_bounds(self):
        return self.bounds

    def get_nec(self):
        count_0Dof_vane = 0
        for sublist in self.acs_object.vanes_rotational_dof_booleans:
            unique, counts = np.unique(sublist, return_counts=True)
            sublist_dict = dict(zip(unique, counts))
            if False in sublist_dict:
                if (sublist_dict[False] == 2):
                    count_0Dof_vane += 1

        count_1Dof_vane_on_body_axis = 0
        for sublist in self.acs_object.vanes_rotational_dof_booleans[self.acs_object.vane_is_aligned_on_body_axis == True]:
            unique, counts = np.unique(sublist, return_counts=True)
            sublist_dict = dict(zip(unique, counts))
            if True in sublist_dict:
                if sublist_dict[True] == 1:
                    count_1Dof_vane_on_body_axis += 1

        count_1Dof_vane_off_body_axis = 0
        for sublist in self.acs_object.vanes_rotational_dof_booleans[self.acs_object.vane_is_aligned_on_body_axis == False]:
            unique, counts = np.unique(sublist, return_counts=True)
            sublist_dict = dict(zip(unique, counts))
            if True in sublist_dict:
                if sublist_dict[True] == 1:
                    count_1Dof_vane_off_body_axis += 1

        num_ec = 3                                  # in the same direction as the desired torque
        num_ec += 3 * count_0Dof_vane               # torque of 0DoF vane is fully known
        num_ec += len(self.acs_object.vane_is_aligned_on_body_axis[self.acs_object.vane_is_aligned_on_body_axis == True])   # Constraint on one of the torques due to position
        num_ec += 1 * count_1Dof_vane_on_body_axis  # spline constraints when on a body axis
        num_ec += 2 * count_1Dof_vane_off_body_axis # spline constraints when off the body axes
        return num_ec

    def get_nic(self):
        count_on_body_axis = 0
        for sublist in self.acs_object.vanes_rotational_dof_booleans[self.acs_object.vane_is_aligned_on_body_axis == True]:
            unique, counts = np.unique(sublist, return_counts=True)
            sublist_dict = dict(zip(unique, counts))
            if True in sublist_dict:
                if sublist_dict[True] == 2:
                    count_on_body_axis += 1

        count_off_body_axis = 0
        for sublist in self.acs_object.vanes_rotational_dof_booleans[self.acs_object.vane_is_aligned_on_body_axis == False]:
            unique, counts = np.unique(sublist, return_counts=True)
            sublist_dict = dict(zip(unique, counts))
            if True in sublist_dict:
                if sublist_dict[True] == 2:
                    count_off_body_axis += 1

        # 1 ellipse when on body axis, 2 otherwise
        num_ic = 1 * count_on_body_axis + 2 * count_off_body_axis
        return num_ic

    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)

    def set_desired_torque(self, desired_torque, previous_x):
        """
        et les docstrings wesh? comment la pouponne va comprendre sinon?(déjà qu'elle écoute rien)

        :param Td: s
        :return:
        """
        self.previous_x = previous_x
        self.desired_torque = desired_torque
        return True

    def set_attaignable_moment_set_ellipses(self,
                                            n_s_body_frame,
                                            default_x_rotation_deg=0,
                                            default_y_rotation_deg=0,
                                            ellipse_bounding_box_margin=default_ellipse_bounding_box_margin):
        """

        :param n_s_body_frame:
        :return:
        """

        # Compute default torque values
        self.default_vane_config_torque_body_frame(n_s_body_frame,
                                                   default_alpha_1_deg=default_x_rotation_deg,
                                                   default_alpha_2_deg=default_y_rotation_deg)
        # Use pre-computed ellipses
        self.vanes_AMS_coefficients_x = []
        self.vanes_AMS_coefficients_y = []
        for vane_id in range(self.acs_object.number_of_vanes):
            R_VBi = np.linalg.inv(self.acs_object.vane_reference_frame_rotation_matrix_list[vane_id])
            alpha_s_rad_vane_reference_frame, beta_s_rad_vane_reference_frame = sun_angles_from_sunlight_vector(
                R_VBi, n_s_body_frame)
            fourier_ellipse_coefficient = []


            for func in self.vanes_AMS_coefficient_functions:
                fourier_ellipse_coefficient.append(func(alpha_s_rad_vane_reference_frame, beta_s_rad_vane_reference_frame))

            Tx_tuple, Ty_tuple = rotated_ellipse_coefficients_wrt_vane_1(R_VBi, tuple(fourier_ellipse_coefficient))
            scaling = self.scaling_list[vane_id]
            self.vanes_AMS_coefficients_x.append(ellipse_stretching(scaling, scaling, Tx_tuple))
            self.vanes_AMS_coefficients_y.append(ellipse_stretching(scaling, scaling, Ty_tuple))

        # Ensure that the weights have the right sign
        for m, weights in enumerate((self.vanes_AMS_coefficients_x + self.vanes_AMS_coefficients_y)):
            if (not self.check_inequality_sign(weights)):
                weights *= -1
                if (m > (self.acs_object.number_of_vanes-1)):
                    self.vanes_AMS_coefficients_y[int(m-self.acs_object.number_of_vanes)] = weights
                else:
                    self.vanes_AMS_coefficients_x[m] = weights

        # If the degrees of freedom of the vanes are limited, compute relevant constraints and adapt the bounds of the
        # optimisation variables
        new_bounds = (1E23 * np.ones_like(self.bounds[0]), -1E23 * np.ones_like(self.bounds[1]))
        self.vanes_spline_constraints_alpha_1_list = []
        self.vanes_spline_constraints_alpha_2_list = []
        for vane_id, vane_rotational_dof_booleans in enumerate(self.acs_object.vanes_rotational_dof_booleans):
            vaneAngleProblem_obj = self.vane_angle_problem_objects_list[vane_id]
            if (vane_rotational_dof_booleans[0] == False and vane_rotational_dof_booleans[1] == False):
                # Trivial case, skip time-expensive computations
                vane_feasible_torque = self.default_vane_torque_body_frame[vane_id]
                for j in range(3):
                    new_bounds[0][vane_id * 3 + j] = vane_feasible_torque[j]
                    new_bounds[1][vane_id * 3 + j] = vane_feasible_torque[j]
                self.vanes_spline_constraints_alpha_1_list.append((None, None, None))
                self.vanes_spline_constraints_alpha_2_list.append((None, None, None))
            elif (vane_rotational_dof_booleans[0] == True and vane_rotational_dof_booleans[1] == True):
                # Determine the bounding box around the ellipse to simplify the optimisation
                if (self.eq_constraints_vane_capabilities[vane_id * 3] != 1):
                    # Torque around X, and there is an ellipse for TxTz
                    x0_TxTz, y0_TxTz, ap_TxTz, bp_TxTz, e_TxTz, phi_TxTz = cart_to_pol(
                        self.vanes_AMS_coefficients_x[vane_id])
                    min_Tx, min_Tz1, max_Tx, max_Tz1 = get_ellipse_bb(x0_TxTz, y0_TxTz, ap_TxTz, bp_TxTz,
                                                                      np.rad2deg(phi_TxTz))
                    dx = max_Tx-min_Tx
                    dz = max_Tz1 - min_Tz1
                    new_bounds[0][vane_id * 3 + 0] = min_Tx - dx * ellipse_bounding_box_margin
                    new_bounds[1][vane_id * 3 + 0] = max_Tx + dx * ellipse_bounding_box_margin
                    new_bounds[0][vane_id * 3 + 2] = min_Tz1 - dz * ellipse_bounding_box_margin
                    new_bounds[1][vane_id * 3 + 2] = max_Tz1 + dz * ellipse_bounding_box_margin
                else:
                    new_bounds[0][vane_id * 3 + 0] = 0
                    new_bounds[1][vane_id * 3 + 0] = 0

                if (self.eq_constraints_vane_capabilities[vane_id * 3 + 1] != 1):
                    x0_TyTz, y0_TyTz, ap_TyTz, bp_TyTz, e_TyTz, phi_TyTz = cart_to_pol(
                        self.vanes_AMS_coefficients_y[vane_id])
                    min_Ty, min_Tz2, max_Ty, max_Tz2 = get_ellipse_bb(x0_TyTz, y0_TyTz, ap_TyTz, bp_TyTz,
                                                                      np.rad2deg(phi_TyTz))
                    dy = max_Ty-min_Ty
                    dz = max_Tz2 - min_Tz2
                    new_bounds[0][vane_id * 3 + 1] = min_Ty - dy * ellipse_bounding_box_margin
                    new_bounds[1][vane_id * 3 + 1] = max_Ty + dy * ellipse_bounding_box_margin
                    new_bounds[0][vane_id * 3 + 2] = min_Tz2 - dz * ellipse_bounding_box_margin
                    new_bounds[1][vane_id * 3 + 2] = max_Tz2 + dz * ellipse_bounding_box_margin
                else:
                    new_bounds[0][vane_id * 3 + 1] = 0
                    new_bounds[1][vane_id * 3 + 1] = 0

                if (self.eq_constraints_vane_capabilities[vane_id * 3] != 1 and
                        self.eq_constraints_vane_capabilities[vane_id * 3 + 1] != 1):
                    if (abs(min_Tz2 - min_Tz1)>1e-15 or abs(max_Tz2 - max_Tz1)>1e-15):
                        print(min_Tz2, min_Tz1)
                        print(max_Tz2, max_Tz1)
                        raise Exception("Tz different in TxTz and TyTz ellipses")
                self.vanes_spline_constraints_alpha_1_list.append((None, None, None))
                self.vanes_spline_constraints_alpha_2_list.append((None, None, None))
            else:
                if (vane_rotational_dof_booleans[0] == False):
                    tuple_fT_alpha_1 = self.reducedDOFConstraintSplines(vaneAngleProblem_obj,
                                                                        case=0,
                                                                        default_alpha_1_deg=default_x_rotation_deg,
                                                                        default_alpha_2_deg=default_y_rotation_deg,
                                                                        n_points=150)

                    for j, fT_list in enumerate(tuple_fT_alpha_1):
                        for fT in fT_list:
                            interpolation_range = fT(fT.t)
                            new_bounds[0][vane_id * 3 + j] = min(min(interpolation_range), new_bounds[0][vane_id * 3 + j])
                            new_bounds[1][vane_id * 3 + j] = max(max(interpolation_range), new_bounds[0][vane_id * 3 + j])
                    # alpha_1 is zero
                    if (self.vane_has_ideal_model==True):   # ideal model - Tz is zero no matter what
                        fTz_list_alpha_1 = [lambda a: np.zeros_like(a)]        # Tz
                        tuple_fT_alpha_1 = (tuple_fT_alpha_1[0], tuple_fT_alpha_1[1], fTz_list_alpha_1)
                    self.vanes_spline_constraints_alpha_1_list.append(tuple_fT_alpha_1)
                else:
                    self.vanes_spline_constraints_alpha_1_list.append((None, None, None))

                if (vane_rotational_dof_booleans[1] == False):
                    tuple_fT_alpha_2 = self.reducedDOFConstraintSplines(vaneAngleProblem_obj,
                                                                        case=1,
                                                                        default_alpha_1_deg=default_x_rotation_deg,
                                                                        default_alpha_2_deg=default_y_rotation_deg,
                                                                        n_points=150)

                    for j, fT_list in enumerate(tuple_fT_alpha_2):
                        for fT in fT_list:
                            interpolation_range = fT(fT.t)
                            new_bounds[0][vane_id * 3 + j] = min(min(interpolation_range), new_bounds[0][vane_id * 3 + j])
                            new_bounds[1][vane_id * 3 + j] = max(max(interpolation_range), new_bounds[1][vane_id * 3 + j])
                    self.vanes_spline_constraints_alpha_2_list.append(tuple_fT_alpha_2)
                else:
                    self.vanes_spline_constraints_alpha_2_list.append((None, None, None))
        # Modify the bounds of the optimisation problem
        bound_stack0 = np.vstack((new_bounds[0], new_bounds[0]))
        bound_stack1 = np.vstack((new_bounds[1], new_bounds[1]))
        min_array = bound_stack0.max(axis=0)          # Take the maximum of the minimum bounds
        max_array = bound_stack1.min(axis=0)          # Take the minimum of the maximum bounds
        min_array[abs(min_array) < 1e-15] = 0
        max_array[abs(max_array) < 1e-15] = 0
        if (any(max_array-min_array) < 0):
            raise Exception("Incompatible bounds")
        self.bounds = (min_array, max_array)                 # Redefine the bounds of the search such that the equality constraint is always valid
        return True

    def check_inequality_sign(self, weights):
        inf_point = np.array([1e23, 1e23])
        D_inf = np.array(
            [[inf_point[0] ** 2, inf_point[0] * inf_point[1], inf_point[1] ** 2, inf_point[0], inf_point[1], 1]])
        return np.dot(D_inf, weights)[0] > 0  # return True if infinity is outside, False if it is inside

    def reducedDOFConstraintSplines(self, vaneAngleProblem_obj, case=0, default_alpha_1_deg=0, default_alpha_2_deg=0, n_points = 150):
        """

        :param: vaneAngleProblem_obj: object from the vaneAnglesAllocationProblem class used to compute the torque of
        the considered vane in a given configuration.
        :param: vane_id: the identification number of the vane for which the constraint is computed.
        :param case: 0 for alpha_1=0 (rotation around vane x-axis), 1 for alpha_2=0 (rotation around vane y-axis).
        :param: default_alpha_1_deg: the default angle of rotation around the vane x-axis.
        :param: default_alpha_2_deg: the default angle of rotation around the vane y-axis.
        :param: n_points: the number of points used to compute the CubicSpline representing the solution.
        :return:
        """
        current_vane_id = vaneAngleProblem_obj.vane_id
        alpha_range = np.linspace(-np.pi, np.pi, n_points)
        Tx_, Ty_, Tz_ = np.zeros(np.shape(alpha_range)), np.zeros(np.shape(alpha_range)), np.zeros(np.shape(alpha_range))
        for k, alpha in enumerate(alpha_range):
            if (case == 0):
                T = vaneAngleProblem_obj.single_vane_torque([np.deg2rad(default_alpha_1_deg), alpha]) * self.scaling_list[current_vane_id]
            elif (case == 1):
                T = vaneAngleProblem_obj.single_vane_torque([alpha, np.deg2rad(default_alpha_2_deg)]) * self.scaling_list[current_vane_id]
            else:
                raise Exception('Error. Unsupported case for reduced degree of freedom constraint calculations')
            Tx_[k] = T[0]
            Ty_[k] = T[1]
            Tz_[k] = T[2]
        alpha_range = alpha_range[Tx_ < 1e20]
        Tx_ = Tx_[Tx_ < 1e20]
        Ty_ = Ty_[Ty_ < 1e20]
        Tz_ = Tz_[Tz_ < 1e20]

        # Determine the boundaries of the shadow in terms of vane angles
        diff_alpha = np.diff(alpha_range)
        cut_indices = np.where(diff_alpha > 1.1 * (2 * np.pi) / n_points)[0]
        alpha_range = alpha_range[..., None]
        alpha_range = alpha_range
        data_points = np.hstack((alpha_range, np.column_stack((Tx_, Ty_, Tz_))))

        # Handle cases where the cut point is at the beginning or end of the sequence
        if (len(cut_indices) != 0):
            if (cut_indices[0] == 0):
                cut_indices = cut_indices[1:] - 1
                data_points = data_points[1:, :]
            elif (cut_indices[-1] == (n_points - 1)):
                cut_indices = cut_indices[:-1]
                data_points = data_points[:-1, :]
        split_arrays = np.split(data_points, cut_indices + 1)
        fTx_list, fTy_list, fTz_list = [], [], []
        if ((len(cut_indices) == 0) and (abs(alpha_range[0]) == np.pi) and (
                abs(alpha_range[0]) - abs(alpha_range[-1])) < 1e-15):
            boundary_type = "periodic"
        else:
            boundary_type = "not-a-knot"

        for i, sp in enumerate(split_arrays):
            k_int = min(np.shape(sp)[0] - 1, 3)  # aim for cubicsplines but go smaller if necessary
            fTx = make_interp_spline(sp[:, 0], sp[:, 1], k=k_int, bc_type=boundary_type)
            fTy = make_interp_spline(sp[:, 0], sp[:, 2], k=k_int, bc_type=boundary_type)
            fTz = make_interp_spline(sp[:, 0], sp[:, 3], k=k_int, bc_type=boundary_type)
            fTx_list.append(fTx)
            fTy_list.append(fTy)
            fTz_list.append(fTz)
        return (fTx_list, fTy_list, fTz_list)

    def default_vane_config_torque_body_frame(self, n_s, default_alpha_1_deg=0, default_alpha_2_deg=0):
        self.default_vane_torque_body_frame = np.array([None] * self.acs_object.number_of_vanes)
        for vane_id in range(self.acs_object.number_of_vanes):
            vaneAngleProblem_obj = self.vane_angle_problem_objects_list[vane_id]
            vaneAngleProblem_obj.update_vane_angle_determination_algorithm(np.array([0, 1, 0]), n_s)    # give random target torque to compute the default
            self.default_vane_torque_body_frame[vane_id] = self.scaling_list[vane_id] * vaneAngleProblem_obj.single_vane_torque([np.deg2rad(default_alpha_1_deg), np.deg2rad(default_alpha_2_deg)])
        return True

class constrainedEllipseCoefficientsProblem:
    def __init__(self, convex_hull_points):
        self.convex_hull_points = convex_hull_points
        self.A0 = None

        # Vector of inequality constraints
        D = np.array([0, 0, 0, 0, 0, 0])
        for xp in convex_hull_points[0:, :]:
            D = np.vstack((D, np.array([xp[0] ** 2, xp[0] * xp[1], xp[1] ** 2, xp[0], xp[1], 1])))
        self.D = D[1:, :]

    def fitness(self, x):
        [a, b, c, d, e, f] = x
        if (self.check_inequality_sign(x)):
            x *= 1  # all rows < 0
        else:
            x *= -1  # all rows < 0
        ineq_constraint = list(np.dot(-self.D, x))  # all rows < 0
        A = self.ellipse_area(x)
        self.previous_A = A
        obj = -A
        return [obj] + [a + c - 1] + (ineq_constraint + [b ** 2 - 4 * a * c])

    def ellipse_area(self, x):
        [a, b, c, d, e, f] = x
        if ((b ** 2 - 4 * a * c) < 0):
            x0, y0, ap, bp, e, phi = cart_to_pol([a, b, c, d, e, f])
            A = np.pi * ap * bp
        else:
            A = self.previous_A
        return A

    def get_bounds(self):
        return (
        [-20, -21, -22, -23, -24, -25], [20, 21, 22, 23, 24, 25])  # Generic bounds as they are necessary for pygmo

    def get_nic(self):
        return np.shape(self.D)[0] + 1

    def get_nec(self):
        return 1

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

    def sef_initial_area(self, a0):
        self.A0 = a0

    def check_inequality_sign(self, current_weights):
        inf_point = np.array([1e23, 1e23])
        D_inf = np.array(
            [[inf_point[0] ** 2, inf_point[0] * inf_point[1], inf_point[1] ** 2, inf_point[0], inf_point[1], 1]])
        return np.dot(D_inf, current_weights)[0] > 0  # return True if infinity is outside, False if it is inside


def generate_AMS_data(vane_id, vaneAngleProblem, current_sun_angle_alpha_deg, current_sun_angle_beta_deg,
                          alpha_1_range, alpha_2_range, optical_model_str="Ideal_model", savefig=True, savedat=False,
                          shadow_computation=0):
    """

    :param vane_id:
    :param vaneAngleProblem:
    :param current_sun_angle_alpha_deg:
    :param current_sun_angle_beta_deg:
    :param alpha_1_range:
    :param alpha_2_range:
    :param optical_model_str:
    :param savefig:
    :param savedat:
    :param shadow_computation:
    :return:
    """
    # shadow computation; 0: without shadow effects, 1: with shadow effects, 2: both
    xlabels = ["Tx", "Tx", "Ty"]
    ylabels = ["Ty", "Tz", "Tz"]

    alpha_1_range_grid, alpha_2_range_grid = np.meshgrid(alpha_1_range, alpha_2_range)
    flattened_alpha_1_range_grid = alpha_1_range_grid.reshape(-1)
    flattened_alpha_2_range_grid = alpha_2_range_grid.reshape(-1)

    # Current sun angles
    alpha_s_rad = np.deg2rad(current_sun_angle_alpha_deg)
    beta_s_rad = np.deg2rad(current_sun_angle_beta_deg)

    # Compute sun vector in body frame
    n_s = np.array([np.sin(alpha_s_rad) * np.cos(beta_s_rad),
                    np.sin(alpha_s_rad) * np.sin(beta_s_rad),
                    -np.cos(alpha_s_rad)])   # In the body reference frame

    vaneAngleProblem.update_vane_angle_determination_algorithm(np.array([0, 1, 0]), n_s)  # False at the next call
    print(f'vane_id={vane_id}, alpha_s_deg={round(np.rad2deg(alpha_s_rad), 1)}, beta_s_deg={round(np.rad2deg(beta_s_rad), 1)}')
    if (savefig): current_figs = [plt.figure(1), plt.figure(2), plt.figure(3)]
    if (shadow_computation==2):
        shadow_l = [0, 1]
    else:
        shadow_l = [shadow_computation]

    returned_arrays = []
    for m in shadow_l:
        if (m==1):
            SHADOW_BOOL = True
            current_fig_label = "With Shadow"
        elif (m==0):
            SHADOW_BOOL = False
            current_fig_label = "No shadow"
        else:
            raise Exception("Error. Wrong shadow condition in generate_AMS_data.")

        vaneAngleProblem.set_shadow_bool(SHADOW_BOOL)
        T_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
        Tx_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
        Ty_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
        Tz_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
        C_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
        for i, alpha_1 in enumerate(alpha_1_range):
            for j, alpha_2 in enumerate(alpha_2_range):
                T = vaneAngleProblem.single_vane_torque([alpha_1, alpha_2])
                T_[i, j] = np.linalg.norm(T)
                Tx_[i, j], Ty_[i, j], Tz_[i, j], C_[i, j] = T[0], T[1], T[2], np.sum(T ** 2) * (1 / 3)

        # Remove the shadow points, if any
        BT = T_
        BT[BT < 1e20], BT[BT > 1e20] = 0, 1
        if (SHADOW_BOOL):
            T_[T_ > 1e20], Tx_[Tx_ > 1e20], Ty_[Ty_ > 1e20], Tz_[Tz_ > 1e20], C_[C_ > 1e20]  = None, None, None, None, None
        flattened_BT, flattened_Tx, flattened_Ty, flattened_Tz  = BT.reshape(-1), Tx_.reshape(-1), Ty_.reshape(-1), Tz_.reshape(-1)

        # Write data to file for further processing
        array_to_save = np.stack([flattened_alpha_1_range_grid, flattened_alpha_2_range_grid,
                                  alpha_s_rad * np.ones(np.shape(flattened_alpha_1_range_grid)),
                                  beta_s_rad * np.ones(np.shape(flattened_alpha_1_range_grid)), flattened_BT,
                                  flattened_Tx, flattened_Ty, flattened_Tz], axis=1)
        returned_arrays.append(array_to_save)
        if (savedat):
            np.savetxt(
                f"{AMS_directory}/Datasets/{optical_model_str}/vane_{vane_id}/AMS_alpha_{round(np.rad2deg(alpha_s_rad), 1)}_beta_{round(np.rad2deg(beta_s_rad), 1)}_shadow_{str(SHADOW_BOOL)}.csv",
                array_to_save, delimiter=",",
                header='alpha_1, alpha_2, alpha_sun, beta_sun, Shadow_bool, Tx, Ty, Tz')

        if (savefig):
            # Plot the scatter-data
            plt.figure(1)
            plt.scatter(flattened_Tx, flattened_Ty, s=1, label=current_fig_label)
            plt.figure(2)
            plt.scatter(flattened_Tx, flattened_Tz, s=1, label=current_fig_label)
            plt.figure(3)
            plt.scatter(flattened_Ty, flattened_Tz, s=1, label=current_fig_label)

    if (savefig):
        for i in range(1, 4):
            plt.figure(i)
            plt.title(f'vane {vane_id}: alpha_s_deg={round(np.rad2deg(alpha_s_rad), 1)}, beta_s_deg={round(np.rad2deg(beta_s_rad), 1)}')
            plt.xlabel(xlabels[i-1])
            plt.ylabel(ylabels[i-1])
            plt.legend(loc='lower left')
            plt.savefig(f'{AMS_directory}/Plots/{optical_model_str}/vane_{vane_id}/plot_{i}/AMS_{i}_alpha_{round(np.rad2deg(alpha_s_rad), 1)}_beta_{round(np.rad2deg(beta_s_rad), 1)}.png')
            plt.close(current_figs[i-1])
    return returned_arrays, shadow_l    # shadow_l returned to indicate the shadow condition used in the calculation

def plot_and_files_wrapper_generate_AMS_data(vane_id,
                              vaneAngleProblem,
                              sun_angle_alpha_range,
                              sun_angle_beta_range,
                              alpha_1_range,
                              alpha_2_range,
                              optical_model_str = "Ideal_model",
                              savefig=True,
                              savedat=False,
                              shadow_computation=0):

    for current_angle_alpha in sun_angle_alpha_range:
        for current_angle_beta in sun_angle_beta_range:
            generate_AMS_data(vane_id, vaneAngleProblem, current_angle_alpha, current_angle_beta, alpha_1_range,
                              alpha_2_range, optical_model_str, savefig, savedat, shadow_computation)

"""
    Functions used in the vane controllers
"""
def fit_2d_ellipse(x, y):
    """
    https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/
    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.
    """
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.
    """
    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

@jit(nopython=True, cache=True)
def cart_to_pol(coeffs):
    """
    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.
    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        print(a, b, c, d, f, g)
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi

def fourierSumFunction(x, *bargs, order=4):
    """

    :param x: input array of length 2.
    :param b: coefficients array of length 1 + 2 * (order-1) + 2 * (order-1) + 4 * (len(combs) - 2 * (order-1)))
    :param order=4: order of the Fourier polynomial.
    :return:
    """
    comb_range = int(order)
    lst = list(range(comb_range))
    combinations = []
    for it in itertools.product(lst, repeat=2):
        combinations.append(list(it))
    combinations = combinations[1:]

    alpha_s = x[:, 0]
    beta_s = x[:, 1]

    b_ind = 0
    res = 1 * bargs[b_ind]; b_ind += 1
    for c in combinations:
        if (c[0] == 0):
            res += (np.sin(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
            res += (np.cos(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
        elif (c[1] == 0):
            res += (np.sin(alpha_s) ** c[0]) * bargs[b_ind]; b_ind += 1
            res += (np.cos(alpha_s) ** c[0]) * bargs[b_ind]; b_ind += 1
        else:
            res += (np.sin(alpha_s) ** c[0]) * (np.sin(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
            res += (np.cos(alpha_s) ** c[0]) * (np.sin(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
            res += (np.sin(alpha_s) ** c[0]) * (np.cos(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
            res += (np.cos(alpha_s) ** c[0]) * (np.cos(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
    return res

def fourierSeriesFunction(x, *bargs, order_n=4, order_m=4):
    alpha_s = x[:, 0]
    beta_s = x[:, 1]
    b_ind = 0
    res = bargs[b_ind]; b_ind += 1
    for i in range(1, order_n + 1):
        for j in range(1, order_m + 1):
            res += bargs[b_ind] * np.sin(i * alpha_s) * np.sin(j * beta_s); b_ind += 1
            res += bargs[b_ind] * np.sin(i * alpha_s) * np.cos(j * beta_s); b_ind += 1
            res += bargs[b_ind] * np.cos(i * alpha_s) * np.sin(j * beta_s); b_ind += 1
            res += bargs[b_ind] * np.cos(i * alpha_s) * np.cos(j * beta_s); b_ind += 1
    return res

def combinedFourierFitFunction(x, *bargs, order=4, order_n=4, order_m=4):
    coefficients_matrix = np.ones((len(x[:, 0]), len(bargs)))
    for i, b in enumerate(bargs):
        coefficients_matrix[:, i] *= b

    res_array, terms_list = combinedFourierFitDesignMatrix(x, *bargs, order=order, order_n=order_n, order_m=order_m)
    res_array = res_array * coefficients_matrix

    return np.sum(res_array, axis=1), np.sum(abs(res_array), axis=0)/np.shape(res_array)[0], terms_list, res_array

def combinedFourierFitDesignMatrix(x, *bargs, order=4, order_n=4, order_m=4):
    comb_range = int(order)
    lst = list(range(comb_range))
    combinations = []
    for it in itertools.product(lst, repeat=2):
        combinations.append(list(it))
    combinations = combinations[1:]

    alpha_s = x[:, 0]
    beta_s = x[:, 1]

    res_array = np.zeros((len(alpha_s), len(bargs)))
    terms_list = []
    b_ind = 0
    res_array[:, b_ind] = np.ones_like(alpha_s); b_ind += 1
    terms_list.append('1')
    for c in combinations:
        if (c[0] == 0):     # Note, this favourises sines, not sure of the implication of that
            res_array[:, b_ind] = (np.sin(beta_s) ** c[1]); b_ind += 1
            terms_list.append(f'(s_beta_s ** {c[1]})')
            if (c[1]<2):
                res_array[:, b_ind] = (np.cos(beta_s) ** c[1]); b_ind += 1
                terms_list.append(f'(c_beta_s ** {c[1]})')

        elif (c[1] == 0):   # Note, this favourises sines, not sure of the implication of that
            res_array[:, b_ind] = (np.sin(alpha_s) ** c[0]); b_ind += 1
            terms_list.append(f'(s_alpha_s ** {c[0]})')
            if (c[0] < 2):
                res_array[:, b_ind] = (np.cos(alpha_s) ** c[0]); b_ind += 1
                terms_list.append(f'(c_alpha_s ** {c[0]})')

        else:
            res_array[:, b_ind] = (np.sin(alpha_s) ** c[0]) * (np.sin(beta_s) ** c[1]); b_ind += 1
            terms_list.append(f'(s_alpha_s ** {c[0]}) * (s_beta_s ** {c[1]})')
            res_array[:, b_ind] = (np.cos(alpha_s) ** c[0]) * (np.sin(beta_s) ** c[1]); b_ind += 1
            terms_list.append(f'(c_alpha_s ** {c[0]}) * (s_beta_s ** {c[1]})')
            res_array[:, b_ind] = (np.sin(alpha_s) ** c[0]) * (np.cos(beta_s) ** c[1]); b_ind += 1
            terms_list.append(f'(s_alpha_s ** {c[0]}) * (c_beta_s ** {c[1]})')
            res_array[:, b_ind] = (np.cos(alpha_s) ** c[0]) * (np.cos(beta_s) ** c[1]); b_ind += 1
            terms_list.append(f'(c_alpha_s ** {c[0]}) * (c_beta_s ** {c[1]})')

    if (order_n > 1 or order_m>1):
        for i in range(1, order_n + 1):
            for j in range(1, order_m + 1):
                if (i==1 and j==1):
                    continue
                res_array[:, b_ind] = np.sin(i * alpha_s) * np.sin(j * beta_s); b_ind += 1
                terms_list.append(f'(s_harmonics_alpha[{i}] * s_harmonics_beta[{j}])')
                res_array[:, b_ind] = np.sin(i * alpha_s) * np.cos(j * beta_s); b_ind += 1
                terms_list.append(f'(s_harmonics_alpha[{i}] * c_harmonics_beta[{j}])')
                res_array[:, b_ind] = np.cos(i * alpha_s) * np.sin(j * beta_s); b_ind += 1
                terms_list.append(f'(c_harmonics_alpha[{i}] * s_harmonics_beta[{j}])')
                res_array[:, b_ind] = np.cos(i * alpha_s) * np.cos(j * beta_s); b_ind += 1
                terms_list.append(f'(c_harmonics_alpha[{i}] * c_harmonics_beta[{j}])')
    return res_array, terms_list

def buildEllipseCoefficientFunctions(filename, number_of_terms=-1):
    with open(filename, 'r') as f:
        lines = f.readlines()
        functions = []

        if (number_of_terms != -1 and number_of_terms>0):
            lines = lines[:number_of_terms+1]

        for line in lines:
            parts = line.strip().split(',')
            coeff = float(parts[0])
            expr = parts[2].strip()

            # Create a function for each line
            def make_function(coeff, expr):
                def fcn(alpha_s, beta_s):
                    c_alpha_s = np.cos(alpha_s)
                    s_alpha_s = np.sin(alpha_s)
                    c_beta_s = np.cos(beta_s)
                    s_beta_s = np.sin(beta_s)

                    s_harmonics_alpha = [np.sin(j * alpha_s) for j in range(16)]
                    c_harmonics_alpha = [np.cos(j * alpha_s) for j in range(16)]
                    s_harmonics_beta = [np.sin(j * beta_s) for j in range(16)]
                    c_harmonics_beta = [np.cos(j * beta_s) for j in range(16)]
                    return coeff * eval(expr)
                return fcn

            functions.append(make_function(coeff, expr))
    return functions

def ellipseCoefficientFunction(alpha_s, beta_s, built_functions):
    res = np.zeros_like(alpha_s)
    for func in built_functions:
        res += func(alpha_s, beta_s)
    return res

def rotated_ellipse_coefficients_wrt_vane_1(R_VB, base_coeffs):
    theta = np.arctan2(R_VB[1, 0], R_VB[0, 0])  # Check that this is correct in this context
    stheta, ctheta = np.sin(theta), np.cos(theta)

    (A, B, C, D, E, F) = base_coeffs
    A_Tx = A * 1
    B_Tx = B * stheta
    C_Tx = C * stheta ** 2
    D_Tx = D * stheta
    E_Tx = E * stheta ** 2
    F_Tx = F * stheta ** 2

    A_Ty = A * 1
    B_Ty = B * ctheta
    C_Ty = C * ctheta ** 2
    D_Ty = D * ctheta
    E_Ty = E * ctheta ** 2
    F_Ty = F * ctheta ** 2
    return [(A_Tx, B_Tx, C_Tx, D_Tx, E_Tx, F_Tx), (A_Ty, B_Ty, C_Ty, D_Ty, E_Ty, F_Ty)]

def ellipse_stretching(scaling_x, scaling_y, base_coeffs):
    (A, B, C, D, E, F) = base_coeffs
    A *= scaling_y ** 2
    B *= scaling_x * scaling_y
    C *= scaling_x ** 2
    D *= (scaling_y ** 2) * scaling_x
    E *= (scaling_x ** 2) * scaling_y
    F *= (scaling_x ** 2) * (scaling_y ** 2)
    return (A, B, C, D, E, F)


def get_ellipse_bb(x, y, ap, bp, angle_deg):
    """
    Compute tight ellipse bounding box.

    see https://stackoverflow.com/questions/87734/how-do-you-calculate-the-axis-aligned-bounding-box-of-an-ellipse#88020
    :param x:
    :param y:
    :param ap:
    :param bp:
    :param angle_deg:
    :return:
    """
    major = 2 * ap
    minor = 2 * bp
    if (angle_deg == 0): angle_deg = 1e-8  # slight fudge to avoid division by zero
    t = np.arctan(-minor / 2 * np.tan(np.radians(angle_deg)) / (major / 2))
    [min_x, max_x] = sorted([x + major / 2 * np.cos(t) * np.cos(np.radians(angle_deg)) -
                             minor / 2 * np.sin(t) * np.sin(np.radians(angle_deg)) for t in (t + np.pi, t)])
    t = np.arctan(minor / 2 * 1. / np.tan(np.radians(angle_deg)) / (major / 2))
    [min_y, max_y] = sorted([y + minor / 2 * np.sin(t) * np.cos(np.radians(angle_deg)) +
                             major / 2 * np.cos(t) * np.sin(np.radians(angle_deg)) for t in (t + np.pi, t)])
    return min_x, min_y, max_x, max_y

def sigmoid_transition(current_time, new_value, previous_time_update, previous_value, shift_time_parameter=1, scaling_parameter=10):
    """
     Calculates the value of a sigmoid transition between a previous value and a new value over time.

     This function models the transition from `previous_value` to `new_value` using a sigmoid function,
     which allows for smooth and gradual change. The sigmoid curve is adjusted by the specified time
     and scaling parameters.

     Parameters:
     current_time: float
        The current time at which the transition value is being calculated.
     new_value: float
        The target value to transition towards.
     previous_time_update: float
        The time at which the previous value was last updated.
     previous_value: float
        The value at `previous_time_update`.
     shift_time_parameter: float (optional)
        A parameter to shift the transition curve along the time axis. Default is 1.
     scaling_parameter: float (optional)
        A parameter to control the steepness of the sigmoid curve. Higher values result in a steeper transition.
        Default is 10.

     Returns:
        float: The transitioned value at `current_time`, smoothly approaching `new_value` from `previous_value`.
    """
    previous_time_update = shift_time_parameter + previous_time_update
    value_change = new_value-previous_value
    exp_term = np.exp(-scaling_parameter * (current_time-previous_time_update))
    res = (value_change/(1+exp_term)) + previous_value
    return res

