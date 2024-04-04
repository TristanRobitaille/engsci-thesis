"""
Measure the accuracy of the exponential approximation: e^x = 2^(x/ln(2))
"""
import sys
sys.path.append("../../")
import math
from utilities import *
from FixedPoint import FXnum

NUM_POINTS = 1000
MIN_ARG = -3
MAX_ARG = 3
MAX_ORDER = 10

float_vs_float_rel_errors = []
fix_vs_float_rel_errors = []

def find_max_error(num_taylor_terms):
    global float_vs_float_rel_errors, fix_vs_float_rel_errors
    # Maximum relative error and corresponding input for approximation as float
    max_error_float_vs_float_rel = 0
    max_error_float_vs_float_input = 0

    # Maximum relative error and corresponding input for approximation as fixed-point
    max_error_fix_vs_float_rel = 0
    max_error_fix_vs_float_input = 0

    # Loops though NUM_POINTS points between min and max
    for i in range(NUM_POINTS):
        x = MIN_ARG + (MAX_ARG - MIN_ARG) * i / NUM_POINTS
        x_fix = FXnum(x, num_Q_comp)
        ref_float = math.exp(x) # Calculate the reference value of e^x (as a float)
        ref_fix = x_fix.exp() # Calculate the reference value of e^x (as a fixed-point)

        # Approximation as fixed-point
        x_for_approx_fix = x_fix / FXnum(math.log(2, math.e), num_Q_comp) # Calculate the argument for the approximation
        x_int_fix = math.floor(x_for_approx_fix)
        x_frac_fix = x_for_approx_fix - x_int_fix # Get fractional part of x_for_approx
        if (x_int_fix < 0): int_part_fix = 1 / (1 << abs(x_int_fix)) # Calculate the integer part of the approximation
        elif (x_int_fix > 0): int_part_fix = 1 << x_int_fix # Calculate the integer part of the approximation
        else: int_part_fix = 1.0

        float_part_fix = 1.0
        for i in range(1, num_taylor_terms+1): # Calculate the float part of the approximation
            float_part_fix += x_frac_fix**i * FXnum((math.log(2,math.e)**i/math.factorial(i)), num_Q_comp)

        approx_fix = int_part_fix * float_part_fix # Calculate the approximation

        # Approximation as float (non-fixed-point)
        x_for_approx_float = float(x) / math.log(2, math.e)
        x_int_float = math.floor(x_for_approx_float)
        x_frac_float = x_for_approx_float - x_int_float
        if (x_int_float < 0): int_part_float = 1 / (1 << abs(x_int_float))
        elif (x_int_float > 0): int_part_float = 1 << x_int_float
        else: int_part_float = 1.0

        float_part_float = 1.0
        for i in range(1, num_taylor_terms+1): # Calculate the float part of the approximation
            float_part_float += x_frac_float**i * (math.log(2,math.e)**i/math.factorial(i))

        approx_float = int_part_float * float_part_float # Calculate the approximation

        # Calculate the error wrt reference float
        error_fix_vs_float = abs(ref_float - float(approx_fix)) / ref_float
        if (error_fix_vs_float > max_error_fix_vs_float_rel): 
            max_error_fix_vs_float_rel = error_fix_vs_float
            max_error_fix_vs_float_input = x

        error_float_vs_float = abs(ref_float - float(approx_float)) / ref_float
        if (error_float_vs_float > max_error_float_vs_float_rel): 
            max_error_float_vs_float_rel = error_float_vs_float
            max_error_float_vs_float_input = x

    print(f"Max relative error of float approximation vs. float reference: {100*max_error_float_vs_float_rel:.6f}% (x = {max_error_float_vs_float_input:.6f})")
    print(f"Max relative error of fixed-point approximation vs. float reference: {100*max_error_fix_vs_float_rel:.6f}% (x = {max_error_fix_vs_float_input:.6f})")
    float_vs_float_rel_errors.append(max_error_float_vs_float_rel)
    fix_vs_float_rel_errors.append(max_error_fix_vs_float_rel)

def main():
    for i in range(MAX_ORDER):
        print(f"Order of Taylor expansion: {i}")
        find_max_error(i)

    print('float_vs_float_rel_errors:', float_vs_float_rel_errors)
    print('fix_vs_float_rel_errors:', fix_vs_float_rel_errors)

if __name__ == "__main__":
    main()
