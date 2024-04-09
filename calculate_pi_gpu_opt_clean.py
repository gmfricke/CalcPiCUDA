import time
import sys
import numpy as np
import numba.cuda as cuda
from numba import vectorize
import math

# Here we calculate pi by assessing the area under the curve using the gpu (float32 precision)
# Then we check the value against what is calculated in numpy

#@cuda.reduce
#def sum_reduce(a, b):
#    return a + b

@cuda.jit
def Pi(step_vec, sum_vec, step_size):

    tx =  cuda.threadIdx.x # Unique thread ID within 1 block
    ty = cuda.blockIdx.x # Unique block ID
    block_size = cuda.blockDim.x # Number of threads per block
    grid_size = cuda.gridDim.x # Size of the grid
    
    # We define the grid size:
    startX = cuda.grid(1) # Starting point on the Grid
    gridX = cuda.gridDim.x * cuda.blockDim.x # stride in x
    stride = cuda.gridsize(1)
    
    # Or we could have written the following
    #startX = tx + ty * block_size
    #stride = block_size * grid_size

    for i in range(startX,  step_vec.shape[0], stride):
        step_vec[i] = (step_vec[i]+0.5) * step_size[0]
        sum_vec[i] +=  4.0 / (1.0 + step_vec[i] * step_vec[i])
    
    
    

if len(sys.argv)!=2:
    print("Usage: ", sys.argv[0], "<number of steps>")
    sys.exit(1)

step_vec = np.arange(int(sys.argv[1],10), dtype=np.float32)
sum_vec = np.zeros(int(sys.argv[1],10), dtype=np.float32)
pi = np.empty(1).astype(np.float32)
N = int(sys.argv[1],10)
step_size = np.ones(1, dtype=np.float32) / N



# We convert the variables to device
step_device = cuda.to_device(step_vec)
sum_device = cuda.to_device(sum_vec)
pi_device = cuda.to_device(pi)
step_size_device = cuda.to_device(step_size)


# We decide to use 128blocks, each containing 128 threads
blocks_per_grid = 128     
# With grid size of 32 I get 
#"Warning: Grid size 32 will likely result in 
#GPU under-utilization due to low occupancy."
thread_per_block = 128  # was 128

# Call the function to calculate pi
start = time.time() #Start
Pi[blocks_per_grid, thread_per_block](step_device, sum_device, step_size_device)

# I copy the devices to host and calculate the final pi version
sum_vec = sum_device.copy_to_host()
pi = step_size * sum_vec.sum()
end = time.time() # End

# Print estimation, the difference from the numpy val and how long it took.
print("Pi = %.20f, (Diff=%.20f) (calculated in %f secs with %d steps)" %(pi, pi-np.pi, end-start, step_vec.shape[0]))

sys.exit(0)
