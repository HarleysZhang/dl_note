# coding: utf-8
# Author: honggao.zhang + chatgpt
# Date: 2023-03-07
# Description: 下载网页图片的 I/O 型密集任务，单线程、多线程和多进程的性能对比

import time, os
import threading, multiprocessing
import queue
import requests

# https://picsum.photos/ 网站只需在我们的网址后添加您想要的图片尺寸（宽度和高度），就会得到一张随机图片。
# 图片的URL列表
IMAGE_URLS = [
    "https://picsum.photos/id/1/200/300",
    "https://picsum.photos/id/2/200/300",
    "https://picsum.photos/id/3/200/300",
    "https://picsum.photos/id/4/200/300",
    "https://picsum.photos/id/5/200/300",
    "https://picsum.photos/id/6/1024/1024",
    "https://picsum.photos/id/7/1024/1024",
    "https://picsum.photos/id/8/1024/1024",
    "https://picsum.photos/id/9/1024/1024",
    "https://picsum.photos/id/10/1024/1024",
]

# 存储图片的目录
SAVE_DIR = "images/download_images"
# 线程数
THREAD_POOL_SIZE = multiprocessing.cpu_count()
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

# 创建存储图片的目录
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# 下载图片的函数
def download_image(url, save_path):
    response = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(response.content)
        
# 单线程下载网页图片
@timer
def single_thread_download():
    dir_name  = SAVE_DIR + "_sig_thread"
    create_dir(dir_name)
    for i, url in enumerate(IMAGE_URLS):
        save_path = os.path.join(dir_name, f"image{i+1}.jpg")
        download_image(url, save_path)

@timer
def multi_thread_download():
    """线程数不固定取决于输入图片列表大小
    """
    dir_name  = SAVE_DIR + "_muti_thread"
    create_dir(dir_name)
    threads = []
    for i, url in enumerate(IMAGE_URLS):
        save_path = os.path.join(dir_name, f"image{i+1}.jpg")
        t = threading.Thread(target=download_image, args=(url, save_path))
        threads.append(t)
        t.start()
    
    # 等待所有线程执行完毕
    for t in threads:
        t.join() 

def thread_worker(work_queue, dir_name):
    while not work_queue.empty():
        try:
            url = work_queue.get(block=False) # 非阻塞取数据
        except queue.Empty:
            break
        else:
            save_path = os.path.join(dir_name, "image{}.jpg".format(url.split('/')[-3]))
            download_image(url, save_path)
            # 表示前面排队的任务已经被完成，被队列的消费者线程使用
            work_queue.task_done()

@timer     
def thread_pool_download():
    # queue 模块实现了多生产者、多消费者队列。适用于消息必须安全地在多线程间交换的多线程编程中。
    work_queue = queue.Queue()
    dir_name  = SAVE_DIR + "_muti_thread_pool"
    create_dir(dir_name)

    for i, url in enumerate(IMAGE_URLS):
        work_queue.put(url)
    
    threads = [threading.Thread(target = thread_worker, args=(work_queue, dir_name)) 
               for _ in range(THREAD_POOL_SIZE)]
    
    # 启动所有线程
    for thread in threads:
        thread.start()
    # 阻塞至队列中所有的元素都被接收和处理完毕
    work_queue.join()
    while threads:
        threads.pop().join()
        
def process_worker(url):
    save_path = os.path.join(SAVE_DIR + "_process_pool", f"image{url.split('/')[-3]}.jpg")
    download_image(url, save_path)
    
@timer
def process_pool_download():
    create_dir(SAVE_DIR + "_process_pool")

    # Pool 类创建一个进程池，大小默认为 CPU 的核数。
    with multiprocessing.Pool() as pool:
        # 自动将一个可迭代对象（如列表、元组等）中的所有元素分配给多个进程或线程，
        # 以进行并行计算，然后返回结果列表。它会使进程阻塞直到结果返回。
        _ = pool.map(process_worker, IMAGE_URLS)

if __name__ == "__main__":

    single_thread_download()
    multi_thread_download()
    thread_pool_download()
    process_pool_download()