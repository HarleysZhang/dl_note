# 优点1: 生成器可以用于生成无限序列，而列表生成式只能用于有限序列。
def fibonacci():
    prev, curr = 0, 1
    while True:
        yield curr
        prev, curr = curr, prev + curr

for i, fib in enumerate(fibonacci()):
    print("fib(%d) = %d" % (i, fib))
    if i > 10:
        break
        
# 优点2:处理大型数据集时，生成器可以一次生成一个元素，而不是在内存中创建整个列表，
# 这可以大大减少内存使用。下面是一个使用生成器来读取大型文本文件的伪代码：
"""
def read_large_file(file_path):
    with open(file_path) as f:
        while True:
            data = f.readline() # 每次读取文件下一行内容
            if not data:
                break
            yield data

for line in read_large_file("large_file.txt"):
    print(line)
    # process_data(line)
"""

import time

def timer(func):
    """decorator: print the cost time of run function"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'{func.__name__} took {end_time - start_time:.6f} seconds to run')
        return result
    return wrapper

def runTime(func):
    """decorator: print the cost time of run function"""
    def wapper(arg, *args, **kwargs):
        start = time.time()
        res = func(arg, *args, **kwargs)
        end = time.time()
        print("="*80)
        print("function name: %s" %func.__name__)
        print("run time: %.4fs" %(end - start))
        print("="*80)
        return res
    return wapper

@timer
def fib(n):
    result_list = []
    prev, curr = 0, 1
    while n > 0:
        result_list.append(curr)
        prev, curr = curr, prev + curr
        n -= 1
    return result_list
    
result = fib(300)
# print(result)

import functools

def cache(func):
    cached_results = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key in cached_results:
            return cached_results[key]
        else:
            result = func(*args, **kwargs)
            cached_results[key] = result
            return result

    return wrapper

@cache
def fibonacci(n):
    if n < 2:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
print(fibonacci(20))

##################### 1，Python 多态 #########################
import math

class Shape:
    def area(self):
        pass

    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.__name__ = 'Rectangle'
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        self.__name__ = 'Circle'
        self.radius = radius

    def area(self):
        return math.pi * self.radius * self.radius

    def perimeter(self):
        return 2 * math.pi * self.radius

def calculate(shape):
    print("The class name is type", shape.__name__)
    print(f"Area: {shape.area():.2f}", shape.area())
    print(f"Perimeter: {shape.perimeter():.2f}", shape.perimeter())

rectangle = Rectangle(5, 10)
circle = Circle(5)

shapes = [rectangle, circle]

for shape in shapes:
    calculate(shape)
    
"""
The class name is type Rectangle
Area: 50.00 50
Perimeter: 30.00 30
The class name is type Circle
Area: 78.54 78.53981633974483
"""

##################### 2，Python @property #########################
class Person:
    def __init__(self, age):
        self._age = age
        self._age_group = self._calculate_age_group()

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, value):
        if value < 0:
            raise ValueError("Age cannot be negative")
        elif value > 150:
            raise ValueError("Age is too large")
        else:
            self._age = value
            self._age_group = self._calculate_age_group()

    @property
    def age_group(self):
        return self._age_group

    def _calculate_age_group(self):
        if self._age < 18:
            return "Age range: under 18"
        elif self._age < 65:
            return "Age range: 18-64"
        else:
            return "Age range: 65 or over"
        
person = Person(30)
print(person.age,",", person.age_group)  # Output: 30, Age range: 18-64

#@property 和 @age.setter 装饰器使得 age 属性看起来像一个普通的属性，但实际上它是一个方法。
person.age = 70
print(person.age,",",person.age_group)  # Output: 70,Age range: 65 or over

# person.age = -10  # Raises ValueError

##################### 3，面向对象编程-多重继承 #########################
class A:
    def method1(self):
        print("A method1")

class B:
    def method1(self):
        print("B method1")
    def method2(self):
        print("B method2")
        
class C(A, B):
    def method3(self):
        # super() 函数会自动查找方法调用的下一个继承类，并调用该类中的同名方法。
        super().method1()
        super().method2()
        print("C method3")

c = C()
c.method3() 
"""
A method1
B method2
C method3
"""

##################### 4，面向对象编程-定制类 #########################