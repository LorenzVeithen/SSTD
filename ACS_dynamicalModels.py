"""
Functions describing the attitude control systems dynamics.
"""
from scipy.spatial.transform import Rotation as R
import numpy as np
from MiscFunctions import all_equal, closest_point_on_a_segment_to_a_third_point, compute_panel_geometrical_properties

def vane_dynamical_model(rotation_x_deg,
                         rotation_y_deg,
                         number_of_vanes,
                         vane_reference_frame_origin_list,
                         vane_panels_coordinates_list,
                         vane_reference_frame_rotation_matrix_list):
    """
        Perform dynamic modeling of vanes based on given rotations in the vane reference frame.

        Parameters:
        ----------
        rotation_x_deg: list of floats
        Rotation angles in degrees around the x-axis of each vane, in the same order as in the
        vane_panels_coordinates_list.

        rotation_y_deg: list of floats
        Rotation angles in degrees around the y-axis of each vane, in the same order as in the
        vane_panels_coordinates_list.

        number_of_vanes: int
        Number of vanes to process.

        vane_reference_frame_origin_list: list of (3, ) numpy arrays
        List of origin vectors for each vane in the body-fixed reference frame.

        vane_panels_coordinates_list: list of numpy arrays
        List of panel coordinates for each vane in the body-fixed reference frame.

        vane_reference_frame_rotation_matrix_list: list of (3, 3) numpy arrays
        List of rotation matrices giving the rotation from the vane reference frame to the body-fixed frame (R_BV).

        Returns:
        new_vane_coordinates: list of numpy arrays
        List of numpy arrays containing the new coordinates of each vane after rotation, in the global reference frame.
    """
    # New version which should be twice as fast as the previous one
    # Precompute rotation matrices for all vanes using SciPy's Rotation
    Rx_list = R.from_euler('x', rotation_x_deg, degrees=True).as_matrix()
    Ry_list = R.from_euler('y', rotation_y_deg, degrees=True).as_matrix()

    new_vane_coordinates = []
    for i in range(number_of_vanes):  # For each vane
        current_vane_origin = vane_reference_frame_origin_list[i]
        current_vane_coordinates = vane_panels_coordinates_list[i]
        current_vane_frame_rotation_matrix = vane_reference_frame_rotation_matrix_list[i]

        # Combine the rotation matrices
        vane_rotation_matrix = np.dot(Ry_list[i], Rx_list[i])

        # Transform coordinates to the vane-centered reference frame
        current_vane_coordinates_vane_reference_frame = (
            np.dot(current_vane_coordinates - current_vane_origin, np.linalg.inv(current_vane_frame_rotation_matrix).T)
        )

        # Apply the rotations
        current_vane_coordinates_rotated_vane_reference_frame = np.dot(
            current_vane_coordinates_vane_reference_frame, vane_rotation_matrix.T
        )

        # Convert back to the body fixed reference frame
        current_vane_coordinates_rotated_body_fixed_reference_frame = (
            np.dot(current_vane_coordinates_rotated_vane_reference_frame, current_vane_frame_rotation_matrix.T)
            + current_vane_origin
        )

        new_vane_coordinates.append(current_vane_coordinates_rotated_body_fixed_reference_frame)

    return new_vane_coordinates