#include <iostream>

class Complex
{
public:
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}

    // 重载 + 运算符，用于把两个 Complex 对象相加
    // 返回类型 函数名(参数列表) const;
    // 用于声明这个函数是一个常量成员函数，不能修改成员变量
    Complex operator+(const Complex& other) const
    {
        return Complex(real + other.real, imag + other.imag);
    }

    // 重载 - 运算符，用于把两个 Complex 对象相减
    Complex operator-(const Complex& other) const
    {
        return Complex(real - other.real, imag - other.imag);
    }
    
    // 定义为友元函数，并重载 << 运算符
    friend std::ostream& operator<<(std::ostream& os, const Complex& c)
    {
        os << "(" << c.real << "+" << c.imag << "i)";
        return os;
    }
private:
    double real, imag;
};

int main()
{
    Complex c1(1, 2), c2(3, 4);
    Complex c3 = c1 + c2;
    Complex c4 = c1 - c2;
    std::cout << "c1 is " << c1 << std::endl;
    std::cout << "c2 is " << c2 << std::endl;
    std::cout << "c1 + c2 result is " << c3 << std::endl;
    return 0;
}
