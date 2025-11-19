import time
import numpy as np
from numba import cuda
from concurrent.futures import ThreadPoolExecutor

def cpu_task_1(array):
    for i in range(len(array)):
        array[i] *= 2

# Function to run on GPU 1
@cuda.jit
def gpu_task_1(array):
    idx = cuda.grid(1)
    if idx < array.size:
        array[idx] *= 2

# Function to run on GPU 2
@cuda.jit
def gpu_task_2(array):
    idx = cuda.grid(1)
    if idx < array.size:
        array[idx] += 5

# Wrapper function to execute GPU tasks
def run_gpu_task(func, array):
    threads_per_block = 256
    blocks_per_grid = (array.size + (threads_per_block - 1)) // threads_per_block
    func[blocks_per_grid, threads_per_block](array)

def run_parallel_tasks():
    # Create large arrays for GPU processing
    array_1 = np.ones(1000000, dtype=np.float32)
    array_2 = np.ones(1000000, dtype=np.float32)
    array_3 = np.ones(1000000, dtype=np.float32)
    array_4 = np.ones(1000000, dtype=np.float32)
    array_5 = np.ones(1000000, dtype=np.float32)

    # Run the GPU tasks in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_gpu_task, gpu_task_1, array_1),
            executor.submit(run_gpu_task, gpu_task_1, array_2),
            executor.submit(run_gpu_task, gpu_task_1, array_3),
            executor.submit(run_gpu_task, gpu_task_1, array_4),
            executor.submit(run_gpu_task, gpu_task_1, array_5),
            executor.submit(run_gpu_task, gpu_task_1, array_1),
            executor.submit(run_gpu_task, gpu_task_1, array_2),
            executor.submit(run_gpu_task, gpu_task_1, array_3),
            executor.submit(run_gpu_task, gpu_task_1, array_4),
            executor.submit(run_gpu_task, gpu_task_1, array_5)
        ]
        # Wait for both tasks to finish
        for future in futures:
            future.result()

    print("GPU Tasks Completed")

# Measure execution time
start_time = time.time()
run_parallel_tasks()
end_time = time.time()

print(f"Parallel GPU tasks completed in {end_time - start_time:.2f} seconds")

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