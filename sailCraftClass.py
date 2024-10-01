#from constants import sail_mass, sail_I, sail_nominal_CoM
import numpy as np
from MiscFunctions import compute_panel_geometrical_properties



class sail_craft:
    """
    Class defining an arbitary sail
    """
    def __init__(self,
                 sail_name,
                 num_wings,
                 num_vanes,
                 initial_wings_coordinates_body_frame,
                 initial_vanes_coordinates_body_frame,
                 initial_wings_optical_properties,
                 initial_vanes_optical_properties,
                 initial_inertia_tensor_body_frame,
                 sail_mass_without_ACS,
                 spacecraft_mass_without_sail,
                 sail_CoM_without_ACS,
                 sail_material_areal_density,
                 vane_material_areal_density,
                 attitude_control_object):
        """
        Constructor for the sailcraft class.

        Parameters:
        sail_name : str
            The name of the sail.
        num_wings : int
            The number of wings of the sail (e.g., the number of sail quadrants).
        num_vanes : int
            The number of vanes used for attitude control.
        initial_wings_coordinates_body_frame : array-like [m]
            A list of num_panels long list of num_points x 3 arrays representing the coordinates
            of the wings in the body frame, where points {p_i}, 0<=i<num_points, form a counterclockwise polygon.
        initial_vanes_coordinates_body_frame : array-like [m]
            A list of num_vanes long list of num_points x 3 arrays representing the coordinates
            of the vanes in the body frame, where points {p_i}, 0<=i<num_points, form a counterclockwise polygon.
        initial_wings_optical_properties : list [-]
            A num_wings long list of 1 x 10 arrays representing the surface properties of the wings.
        initial_vanes_optical_properties : list [-]
            A num_vanes long list of 1 x 10 arrays representing the surface properties of the vanes.
        initial_inertia_tensor_body_frame : numpy.ndarray [kg/m^2]
            A 3x3 tensor representing the initial inertia tensor in the body frame.
        sail_mass_without_ACS : float [kg]
            The mass of the sail without the attitude control system (ACS).
        spacecraft_mass_without_sail : float [kg]
            The mass of the spacecraft without the sail.
        sail_CoM_without_ACS : numpy.ndarray [m]
            A 3D vector representing the center of mass coordinates of the sail without the ACS, in the body-fixed frame.
        sail_material_areal_density : float [kg/m^2]
            The areal density of the sail material.
        vane_material_areal_density : float [kg/m^2]
            The areal density of the vane material.
        attitude_control_object : object
            An object representing the attitude control system (ACS).
        """
        # Link to other classes
        self.bodies = None
        self.attitude_control_system = attitude_control_object                          # The ACS class object to obtain all control mechanisms
        self.current_time = None

        # Non-varying sail characteristics
        self.sail_name = sail_name
        self.sail_num_wings = num_wings
        self.sail_num_vanes = num_vanes
        self.sail_mass_without_attitude_control_system = sail_mass_without_ACS
        self.spacecraft_mass_without_sail = spacecraft_mass_without_sail
        self.attitude_control_system_mass = attitude_control_object.get_attitude_system_mass()
        self.sail_mass = sail_mass_without_ACS + self.attitude_control_system_mass   # Without mass-based attitude control system,
        self.sail_center_of_mass_body_fixed_position_without_ACS = sail_CoM_without_ACS
        self.desired_sail_body_frame_inertial_rotational_velocity = None
        self.sail_material_areal_density = sail_material_areal_density
        self.vane_material_areal_density = vane_material_areal_density

        # Time-varying variables
        ## Panels
        self.sail_wings_coordinates = initial_wings_coordinates_body_frame            # List of num_points x 3 arrays - assuming points {p_i}, 0<=i<n, forms a counterclockwise polygon,
        self.sail_wings_optical_properties = initial_wings_optical_properties         # num_panels x 10 array of panel surface properties
        self.sail_wings_areas = np.zeros(num_wings)                                   # List of panel areas
        self.sail_wings_centroids = [None] * num_wings                                # List of panel centroids
        self.sail_wings_surface_normals = [None] * num_wings                          # List of panel surface normal

        ## Vanes
        self.sail_vanes_coordinates = initial_vanes_coordinates_body_frame              # List of num_points x 3 arrays - assuming points {p_i}, 0<=i<n, forms a counterclockwise polygon,
        self.sail_vanes_optical_properties = initial_vanes_optical_properties           # num_panels x 10 array of panel surface properties
        self.sail_vanes_areas = np.zeros(num_vanes)                                     # List of panel areas
        self.sail_vanes_centroids = [None] * num_vanes                                  # List of panel centroids
        self.sail_vanes_surface_normals = [None] * num_vanes                            # List of panel surface normal

        ## Gimball and moving masses systems
        self.sail_attitude_control_system_center_of_mass = sail_mass_without_ACS        # Mass of the ACS as an independent system, intialised at the sail CoM without ACS - TODO: check that this makes sense, and if the CoM is updated fast enough for this to be "allowed"
        self.moving_masses_positions_dict = None

        ## General spacecraft
        self.sail_center_of_mass_position = sail_CoM_without_ACS                        # 3D vector - initialised to value without ACS
        self.sail_inertia_tensor = initial_inertia_tensor_body_frame                    # 3x3 tensor

        # Initialising some properties
        self.compute_reflective_panels_properties(panel_id_list=list(range(self.sail_num_wings)), vanes_id_list=list(range(self.sail_num_vanes)))  # Compute the individual panel properties

        # Total vane mass
        vane_mass = 0
        for i in range(self.sail_num_vanes):
            vane_mass += self.sail_vanes_areas[i] * self.sail_material_areal_density
        self.sail_vanes_total_mass = vane_mass

        # Bookkeeping variables
        self.current_body_position = np.array([None, None, None])

    def compute_reflective_panels_properties(self, panel_id_list, vanes_id_list):  # should enable the possibility of only recomputing a specific panel
        # For each panel and vane, obtain coordinates of panels and compute the panel area
        for j, id_list in enumerate([panel_id_list, vanes_id_list]):
            if (j==0):
                coordinates_list_pointer = self.sail_wings_coordinates
                centroids_list_pointer = self.sail_wings_centroids
                areas_list_pointer = self.sail_wings_areas
                surface_normals_list_pointer = self.sail_wings_surface_normals
            else:
                coordinates_list_pointer = self.sail_vanes_coordinates
                centroids_list_pointer = self.sail_vanes_centroids
                areas_list_pointer = self.sail_vanes_areas
                surface_normals_list_pointer = self.sail_vanes_surface_normals

            for p_id in id_list:
                current_panel_centroid, current_panel_area, current_panel_surface_normal = compute_panel_geometrical_properties(coordinates_list_pointer[p_id])

                # Assign to correct lists
                centroids_list_pointer[p_id] = current_panel_centroid
                areas_list_pointer[p_id] = current_panel_area
                surface_normals_list_pointer[p_id] = current_panel_surface_normal
        return 0

    def compute_sail_center_of_mass(self, ACS_center_of_mass):
        # Contribution of vanes is already taking into account in the ACS component

        # Recompute everything to be sure
        self.compute_reflective_panels_properties(list(range(self.sail_num_wings)), list(range(self.sail_num_vanes)))
        summation = np.array([0, 0, 0], dtype="float64")

        # Fixed bus and booms
        summation += self.spacecraft_mass_without_sail * self.sail_center_of_mass_body_fixed_position_without_ACS

        # Compute contribution of panels
        for i in range(self.sail_num_wings):
            summation += self.sail_wings_centroids[i] * (self.sail_wings_areas[i] * self.sail_material_areal_density)

        summation += self.attitude_control_system_mass * ACS_center_of_mass
        return summation/self.sail_mass

    def compute_sail_inertia_tensor(self):  # TODO: handle both moving masses and moving panels
        return self.sail_inertia_tensor

    # Control the sail craft
    def sail_attitude_control_system(self):
        # Calls the ACS, updating the spacecraft properties before sending it to the propagation
        # Call all ACS controllers here to change the spacecraft state and control the orbit
        if (self.bodies != None):
            body_position = self.bodies.get(self.sail_name).position
            if (self.current_body_position[0] != None):
                difference_current_position_last_update = body_position - self.current_body_position
        else:
            body_position = np.array([None, None, None])

        if ((self.current_body_position != body_position).all() or (self.current_body_position == None).all()):  # Second condition for initialisation
            panels_coordinates, panels_optical_properties, ACS_CoM, moving_masses_positions = self.attitude_control_system.attitude_control(self.bodies, self.desired_sail_body_frame_inertial_rotational_velocity, self.current_time)    # Find way to include the current state and the desired one
            self.sail_wings_coordinates = panels_coordinates["wings"] if (len(panels_coordinates["wings"]) != 0) else self.sail_wings_coordinates
            self.sail_vanes_coordinates = panels_coordinates["vanes"] if (len(panels_coordinates["vanes"]) != 0) else self.sail_vanes_coordinates
            self.sail_wings_optical_properties = panels_optical_properties["wings"] if (len(panels_optical_properties["wings"]) != 0) else self.sail_wings_optical_properties
            self.sail_vanes_optical_properties = panels_optical_properties["vanes"] if (len(panels_optical_properties["vanes"]) != 0) else self.sail_vanes_optical_properties

            # Recompute the necessary spacecraft properties based on the new configuration
            self.compute_reflective_panels_properties(list(range(self.sail_num_wings)), list(range(self.sail_num_vanes)))
            self.sail_center_of_mass_position = self.compute_sail_center_of_mass(ACS_CoM)
            self.moving_masses_positions_dict = moving_masses_positions
            if (self.bodies != None): self.current_body_position = body_position
        return 0

    # Pass body object to the class
    def setBodies(self, bodies):
        # Sets the bodies necessary to call the controller
        # This should be called right after the creation of the bodies, and the class is not fully defined until then
        self.bodies = bodies
        return 0

    # Pass the desired state to the class
    def set_desired_sail_body_frame_inertial_rotational_velocity(self, desired_state):
        self.desired_sail_body_frame_inertial_rotational_velocity = desired_state
        return 0

    # Get rigid body properties
    def get_sail_mass(self, t):
        self.current_time = t
        self.sail_attitude_control_system()
        return self.sail_mass

    def get_sail_center_of_mass(self, t):
        return self.sail_center_of_mass_position

    def get_sail_inertia_tensor(self, t):
        return self.sail_inertia_tensor

    # Get moving masses positions
    def get_sail_moving_masses_positions(self, t):
        return self.moving_masses_positions_dict

    def get_number_of_wings(self):
        """
        Get the number of wings of the sailcraft.
        :return: int, the number of wings of the sailcraft.
        """
        return self.sail_num_wings

    def get_number_of_vanes(self):
        """
        Get the number of vanes of the sailcraft.
        :return: int, the number of vanes of the sailcraft attitude control system.
        """
        return self.sail_num_vanes

    # Get specific panel properties
    def get_ith_panel_surface_normal(self, panel_id, panel_type=""):
        """
        Get the ith panel surface normal in the body-fixed reference frame.
        :param panel_id: int, position in initial user-specified panel coordinates list.
        :param panel_type: str, "Vane" or "Wing", defining the type of panel considered.
        :return:  (numpy.ndarray): A 1D array of shape (3,) representing
            the x, y, and z components of the panel's surface normal.
        """
        self.sail_attitude_control_system()
        if (panel_type == "Vane"):
            return self.sail_vanes_surface_normals[panel_id]
        else:
            return self.sail_wings_surface_normals[panel_id]

    def get_ith_panel_area(self, panel_id, panel_type=""):
        """
        Get the ith panel area.
        :param panel_id: int, position in initial user-specified panel coordinates list.
        :param panel_type: str, "Vane" or "Wing", defining the type of panel considered.
        :return: The panel area.
        """
        self.sail_attitude_control_system()
        if (panel_type == "Vane"):
            return self.sail_vanes_areas[panel_id]
        else:
            return self.sail_wings_areas[panel_id]

    def get_ith_panel_centroid(self, panel_id, panel_type=""):
        """
        Get the ith panel centroid coordinates in the body-fixed reference frame.
        :param panel_id: int, position in initial user-specified panel coordinates list.
        :param panel_type: str, "Vane" or "Wing", defining the type of panel considered.
        :return: The panel centroid coordinates in the body-fixed reference frame.
        """
        self.sail_attitude_control_system()
        if (panel_type == "Vane"):
            return self.sail_vanes_centroids[panel_id]
        else:
            return self.sail_wings_centroids[panel_id]

    def get_ith_panel_optical_properties(self, panel_id, panel_type=""):
        """
        Get the ith panel optical properties.
        :param panel_id: int, position in initial user-specified panel coordinates list.
        :param panel_type: str, "Vane" or "Wing", defining the type of panel considered.
        :return: The panel optical properties
        """
        self.sail_attitude_control_system()
        if (panel_type == "Vane"):
            return self.sail_vanes_optical_properties[panel_id]
        else:
            return self.sail_wings_optical_properties[panel_id]

    def get_ith_panel_coordinates(self, panel_id, panel_type=""):
        """
        Get the ith panel body-fixed coordinates.
        :param panel_id: int, position in initial user-specified panel coordinates list.
        :param panel_type: str, "Vane" or "Wing", defining the type of panel considered.
        :return: The body-fixed panel coordinates.
        """
        self.sail_attitude_control_system()
        if (panel_type == "Vane"):
            return self.sail_vanes_coordinates[panel_id]
        else:
            return self.sail_wings_coordinates[panel_id]

