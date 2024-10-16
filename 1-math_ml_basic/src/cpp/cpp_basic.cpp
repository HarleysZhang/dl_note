#include <iostream>
#include <vector>
#include <string>

class MyClass {
public:
    MyClass() { std::cout << "Default constructor" << std::endl; }
    MyClass(const MyClass& other) { std::cout << "Copy constructor" << std::endl; }
    MyClass(MyClass&& other) { std::cout << "Move constructor" << std::endl; }

};

int main() {
    std::vector<MyClass> v1;

    // 添加对象
    v1.push_back(MyClass());
    v1.push_back(MyClass());

    std::cout << "Before move: " << std::endl;

    // 打印容器中的对象数量
    std::cout << "Size: " << v1.size() << std::endl;

    // 移动容器中的对象
    std::vector<MyClass> v2(std::move(v1));

    std::cout << "After move: " << std::endl;

    // 打印容器中的对象数量
    std::cout << "v1 size: " << v1.size() << std::endl; // v1已被移动，不再包含任何元素
    std::cout << "v2 size: " << v2.size() << std::endl; // v2包含了v1中的元素

    return 0;
}
