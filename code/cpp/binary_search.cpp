#include <stdio.h>
#include <iostream>
#include <vector>

using namespace std;

int binary_search(std::vector<int> arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    while (left <= right) {
        auto mid = (left + right) / 2;

        if (arr[mid] == target)
            return mid;
        else if (arr[mid] < target) 
            right = mid - 1;
        else
            left = left + 1;
    }
    return -1;
}

int main() {
    std::vector<int> arr({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    int target = 5;
    auto index = binary_search(arr, target);
    std::cout << "index: " << index << std::endl;
}