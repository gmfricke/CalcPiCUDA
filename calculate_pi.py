import time
import sys
import numpy as np


# Here we calculate pi by assessing the area under the curve, using the CPU
# Then we check the value against what is calculated in numpy
# source: https://github.com/UNM-CARC/QuickBytes/blob/master/workshop_slides/CS_Math_471.pdf

def Pi(num_steps):
    step = 1.0 / num_steps
    sum = 0

    for i in range(num_steps):
        x = (i+0.5) * step
        sum = sum + 4.0 / (1.0 + x * x)

    pi = step * sum


    return pi

if len(sys.argv)!=2:
    print("Usage: ", sys.argv[0], "<number of steps>")
    sys.exit(1)

num_steps = int(sys.argv[1],10)

# Call the function to calculate pi
start = time.time() #Start
pi = Pi(num_steps)
print(type(pi))
end = time.time() # End

# Print estimation, the difference from the numpy val and how long it took.
print("Pi = %.20f, (Diff=%.20f) (calculated in %f secs with %d steps)" %(pi, pi-np.pi, end-start, num_steps))

sys.exit(0)
