import threading, multiprocessing
import time
import numpy as np
import concurrent.futures

# 进程数等于CPU核心数
NUM_PROCESS = multiprocessing.cpu_count()

def timer(func):
    """decorator: print the cost time of run function"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'{func.__name__} took {end_time - start_time:.6f} seconds to run')
        return result
    return wrapper

@timer
def matrix_mult(A, B):
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrix dimensions do not match.")
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i][j] += A[i][k] * B[k][j]
    return C

@timer
def matrix_mult_threaded(A, B, num_threads):
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrix dimensions do not match.")
    C = np.zeros((A.shape[0], B.shape[1]))
    def multiply(i_start, i_end):
        for i in range(i_start, i_end):
            for j in range(B.shape[1]):
                for k in range(A.shape[1]):
                    C[i][j] += A[i][k] * B[k][j]
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        chunk_size = A.shape[0] // num_threads
        for i in range(num_threads):
            i_start = i * chunk_size
            i_end = (i+1) * chunk_size if i != num_threads-1 else A.shape[0]
            futures.append(executor.submit(multiply, i_start, i_end))
        concurrent.futures.wait(futures)
    return C

@timer
def matrix_mult_multiprocess(A, B, num_processes):
    """matrix_mult_multiprocess 是主进程，
    并通过 multiprocessing.Pool 进程池的方式批量创建子进程。"""
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrix dimensions do not match.")
    C = np.zeros((A.shape[0], B.shape[1]))
    
    def multiply(i_start, i_end):
        for i in range(i_start, i_end):
            for j in range(B.shape[1]):
                for k in range(A.shape[1]):
                    C[i][j] += A[i][k] * B[k][j]
    
    # Pool 类创建一个进程池，大小默认为 CPU 的核数。
    p = multiprocessing.Pool()
    for i in range(num_processes):
        i_start = i * (A.shape[0] // num_processes)
        i_end = (i+1) * (A.shape[0] // num_processes) if i != num_processes-1 else A.shape[0]
        p.apply_async(multiply, args=(i_start, i_end))
   
    # print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    return C

if __name__ == "__main__":
    # 测试
    A = np.random.rand(200, 200)
    B = np.random.rand(200, 200)
    C = matrix_mult(A, B)
    C = matrix_mult_threaded(A, B, NUM_PROCESS)
    C = matrix_mult_multiprocess(A, B, NUM_PROCESS)
