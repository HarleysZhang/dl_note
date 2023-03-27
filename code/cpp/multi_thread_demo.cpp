#include <stdio.h>
#include <iostream>
#include <thread>
#include <mutex>

using namespace std;

// 线程函数
void worker() {
    std::cout << "Hello from worker thread " << std::endl;
}

// 线程函数对象，但是声明为只调用一次
void simple_do_once(std::once_flag* flag)
{
    std::call_once(*flag, [](){ std::cout << "Simple example: called once\n"; });
}

int main() {
    std::once_flag flag1;
    std::thread worker_thread(worker);
    worker_thread.join();

    std::thread thread_once(simple_do_once, &flag1);
    if (thread_once.joinable())
        thread_once.join();
    return 0;
}