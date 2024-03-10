"""
Measure the accuracy of the exponential approximation: e^x = 2^(x/ln(2))
"""
import sys
sys.path.append("..")
import math
from asic.rtl.utilities import *
from FixedPoint import FXnum

NUM_POINTS = 1000
MIN_ARG = -3
MAX_ARG = 3

max_error_float_rel = 0
max_error_float_input = 0
max_error_fix_rel = 0
max_error_fix_input = 0
taylor_multipliers = [1, math.log(2,math.e), math.log(2,math.e)**2/2, math.log(2,math.e)**3/6]

# Print Taylor series multipliers in fixed point format
print(f"1/ln(2): {num_Q(1/math.log(2,math.e)).toBinaryString(logBase=1, twosComp=True)}")
for i in range(len(taylor_multipliers)):
    print(f"ln(2)^{i}/{i}!: {num_Q(taylor_multipliers[i]).toBinaryString(logBase=1, twosComp=True)}")

# Loops though NUM_POINTS points between min and max
for i in range(NUM_POINTS):
    x = MIN_ARG + (MAX_ARG - MIN_ARG) * i / NUM_POINTS
    x_fix = FXnum(x, num_Q).exp()
    real_float = math.exp(x) # Calculate the real value of e^x (as a float)
    real_fix = x_fix.exp() # Calculate the real value of e^x (as a fixed-point)

    x_for_approx = x_fix / FXnum(math.log(2, math.e), num_Q) # Calculate the argument for the approximation
    x_int = math.floor(x_for_approx)
    x_frac = x_for_approx - x_int # Get fractional part of x_for_approx
    if (x_int < 0): int_part = 1 / (1 << abs(x_int)) # Calculate the integer part of the approximation
    elif (x_int > 0): int_part = 1 << x_int # Calculate the integer part of the approximation
    else: int_part = 1.0
    float_part = 1 + x_frac * taylor_multipliers[1] + x_frac**2 * taylor_multipliers[2] +  + x_frac**3 * taylor_multipliers[3] # Calculate the float part of the approximation

    approx = int_part * float_part # Calculate the approximation
    
    # Calculate the error wrt real float
    error_float = abs(real_float - float(approx)) / real_float
    if (error_float > max_error_float_rel): 
        max_error_float_rel = error_float
        max_error_float_input = x

    error_fix = abs(real_fix - approx) / real_fix
    if (error_fix > max_error_fix_input): 
        max_error_fix_rel = error_fix
        max_error_fix_input = x

print(f"Max relative error wrt. float {100*max_error_float_rel:.4f}% (x = {max_error_float_input:.6f})")
print(f"Max relative error wrt. fixed-point {100*float(max_error_fix_rel):.4f}% (x = {float(max_error_fix_input):.6f})")