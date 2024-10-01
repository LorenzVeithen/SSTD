from truncatedEllipseCoefficientsFunctions import ellipse_truncated_coefficients_function_shadow_FALSE_double_ideal_optical_model, ellipse_truncated_coefficients_function_shadow_TRUE_double_ideal_optical_model
from truncatedEllipseCoefficientsFunctions import ellipse_truncated_coefficients_function_shadow_FALSE_single_ideal_optical_model, ellipse_truncated_coefficients_function_shadow_TRUE_single_ideal_optical_model
from truncatedEllipseCoefficientsFunctions import ellipse_truncated_coefficients_function_shadow_FALSE_ACS3_optical_model, ellipse_truncated_coefficients_function_shadow_TRUE_ACS3_optical_model
from time import time
from vaneControllerMethods import cart_to_pol
from constants import wings_coordinates_list
from MiscFunctions import compute_panel_geometrical_properties

print("double ideal model")
funcs = ellipse_truncated_coefficients_function_shadow_TRUE_double_ideal_optical_model()

l = funcs[0](0, 1)
t0 = time()
l = funcs[1](0, 1)
l = funcs[2](0, 1)
l = funcs[3](0, 1)
l = funcs[4](0, 1)
l = funcs[5](0, 1)
print(time()-t0)

funcs = ellipse_truncated_coefficients_function_shadow_FALSE_double_ideal_optical_model()

l = funcs[0](0, 1)
t0 = time()
l = funcs[1](0, 1)
l = funcs[2](0, 1)
l = funcs[3](0, 1)
l = funcs[4](0, 1)
l = funcs[5](0, 1)
print(time()-t0)

print("single ideal model")
funcs = ellipse_truncated_coefficients_function_shadow_FALSE_single_ideal_optical_model()

l = funcs[0](0, 1)
t0 = time()
l = funcs[1](0, 1)
l = funcs[2](0, 1)
l = funcs[3](0, 1)
l = funcs[4](0, 1)
l = funcs[5](0, 1)
print(time()-t0)

funcs = ellipse_truncated_coefficients_function_shadow_TRUE_single_ideal_optical_model()

l = funcs[0](0, 1)
t0 = time()
l = funcs[1](0, 1)
l = funcs[2](0, 1)
l = funcs[3](0, 1)
l = funcs[4](0, 1)
l = funcs[5](0, 1)
print(time()-t0)

print("ACS3")
funcs = ellipse_truncated_coefficients_function_shadow_FALSE_ACS3_optical_model()

l = funcs[0](0, 1)
t0 = time()
l = funcs[1](0, 1)
l = funcs[2](0, 1)
l = funcs[3](0, 1)
l = funcs[4](0, 1)
l = funcs[5](0, 1)
print(time()-t0)

funcs = ellipse_truncated_coefficients_function_shadow_TRUE_ACS3_optical_model()

l = funcs[0](0, 1)
t0 = time()
l = funcs[1](0, 1)
l = funcs[2](0, 1)
l = funcs[3](0, 1)
l = funcs[4](0, 1)
l = funcs[5](0, 1)
print(time()-t0)

cart_to_pol((1, 0, 1, 1, 1, 1))
t0 = time()
cart_to_pol((1, 0, 1, 1, 1, 1))
print(time() - t0)

compute_panel_geometrical_properties(wings_coordinates_list[0])
t0 = time()
compute_panel_geometrical_properties(wings_coordinates_list[0])
print(time() - t0)