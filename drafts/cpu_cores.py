import time
import numpy as np
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor

# @njit(parallcpyel=True)
def cpu_task_1(array):
    for i in range(len(array)):
        array[i] *= 2

array_1 = np.ones(1000000, dtype=np.float32)
array_2 = np.ones(1000000, dtype=np.float32)
array_3 = np.ones(1000000, dtype=np.float32)
array_4 = np.ones(1000000, dtype=np.float32)
array_5 = np.ones(1000000, dtype=np.float32)

start_time = time.time()
cpu_task_1(array_1)
cpu_task_1(array_2)
cpu_task_1(array_3)
cpu_task_1(array_4)
cpu_task_1(array_5)
cpu_task_1(array_1)
cpu_task_1(array_2)
cpu_task_1(array_3)
cpu_task_1(array_4)
cpu_task_1(array_5)
end_time = time.time()

print(f"CPU tasks completed in {end_time - start_time:.2f} seconds")