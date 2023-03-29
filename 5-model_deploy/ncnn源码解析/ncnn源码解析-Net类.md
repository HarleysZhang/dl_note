## 一，网络类 Net

`net.cpp/net.h` 是模型结构定义和模型推理类所在文件，主要包括以下类：

- `Net`: 网络类，主要提供网络结构和网络权重文件加载/解析接口: `load_param` 和 `load_model`。
- `Extractor`: 模型推理之行类，主要对外接口是网络输入函数 `input` 和网络推理函数 `extract`。
- `NetPrivate`：Net 类的私有成员都定义这个单独的类中，比如 blobs、layers、input_blob_indexes、output_blob_indexes、custom_layer_registry 和 `local_blob_allocator` 等。
- `ExtractorPrivate`: Extractor 类的私有成员都定义这个单独的类中。

将 `ncnn::Net` 类的私有成员封装成了一个类 `NetPrivate`，ncnn 框架中很多类都有类似操作，比如 `Pipeline`、`ParamDict`、`Extractor`、`PoolAllocator`、`DataReaderFromMemory` 等类。

### 1.1，Net 类解析

`Net` 的部分定义如下（省略了部分代码）:

```cpp
class Net
{
public:
    // empty init
    Net();
    // clear and destroy
    virtual ~Net();

public:
    // option can be changed before loading
    Option opt;
    int register_custom_layer(int index, layer_creator_func creator, 		      layer_destroyer_func destroyer = 0, void* userdata = 0);
    // 实际的模型结构和权重文件加载函数
    int load_param_bin(const DataReader& dr);
    int load_model(const DataReader& dr);
    
    // load network structure from binary param file
  	int load_param(const char* protopath);
    // load network weight data from model file return 0 if success
    int load_model(const char* modelpath);

private:
    NetPrivate* const d; // 类
};
```

`NetPrivate` 主要成员变量是：

```cpp
// Blob 用于记录 featuremap 张量数据
std::vector<Blob> blobs;
std::vector<Layer*> layers;

std::vector<int> input_blob_indexes;
std::vector<int> output_blob_indexes;
```

## 二，参数字典类 paramdict

Net::load_param() 函数中用到 ParamDict 类的代码有以下三处，

```cpp
Mat shape_hints = pd.get(30, Mat());
layer->featmask = pd.get(31, 0);
int lr = layer->load_param(pd);
```

ParamDict 类中是通过 ParamDictPrivate 类保存私有成员变量，ParamDict 类定义如下所示:

```cpp
class NCNN_EXPORT ParamDict
{
public:
    // empty
    ParamDict();

    virtual ~ParamDict();

    // copy
    ParamDict(const ParamDict&);

    // assign
    ParamDict& operator=(const ParamDict&);

    // get type
    int type(int id) const;

    // get int
    int get(int id, int def) const;
    // get float
    float get(int id, float def) const;
    // get array
    Mat get(int id, const Mat& def) const;

    // set int
    void set(int id, int i);
    // set float
    void set(int id, float f);
    // set array
    void set(int id, const Mat& v);

protected:
    friend class Net;

    void clear();

    int load_param(const DataReader& dr);
    int load_param_bin(const DataReader& dr);

private:
    ParamDictPrivate* const d;
};
```

ParamDict 类的主要成员函数 load_param 解析如下:

```cpp
int ParamDict::load_param(const DataReader& dr)
{
    clear();

    //     0=100 1=1.250000 -23303=5,0.1,0.2,0.4,0.8,1.0

    // parse each key=value pair
    int id = 0;
    while (dr.scan("%d=", &id) == 1)
    {
        // 是否为数组类型
        bool is_array = id <= -23300;
        if (is_array)
        {
            id = -id - 23300;
        }
        // id 是否超过最大参数数
        if (id >= NCNN_MAX_PARAM_COUNT)
        {
            NCNN_LOGE("id < NCNN_MAX_PARAM_COUNT failed (id=%d, NCNN_MAX_PARAM_COUNT=%d)", id, NCNN_MAX_PARAM_COUNT);
            return -1;
        }
        // 如果是数组类型，执行以下解析操作
        if (is_array)
        {
            int len = 0;
            int nscan = dr.scan("%d", &len); // 解析数组长度
            if (nscan != 1)
            {
                NCNN_LOGE("ParamDict read array length failed");
                return -1;
            }

            d->params[id].v.create(len); // 创建数组
            // 遍历数组元素并解析
            for (int j = 0; j < len; j++)
            {   
                // 解析数组元素
                char vstr[16];
                nscan = dr.scan(",%15[^,\n ]", vstr);
                if (nscan != 1)
                {
                    NCNN_LOGE("ParamDict read array element failed");
                    return -1;
                }
                
                // 是否为浮点数，看解析的字符串中是否存在'.'或'e'
                // 小数点计数法和科学计数法
                bool is_float = vstr_is_float(vstr);

                // 转换为相应类型
                if (is_float)
                {
                    float* ptr = d->params[id].v;
                    ptr[j] = vstr_to_float(vstr);
                }
                else
                {
                  	// vstr赋值给params[id].v[j]
                    int* ptr = d->params[id].v;
                    nscan = sscanf(vstr, "%d", &ptr[j]);
                    if (nscan != 1)
                    {
                        NCNN_LOGE("ParamDict parse array element failed");
                        return -1;
                    }
                }
                // 设置参数类型
                d->params[id].type = is_float ? 6 : 5;
            }
        }
        // 如果不是数组类型，则解析单个值，步骤和 if 内部语句快一样
        else
        {
            char vstr[16];
            int nscan = dr.scan("%15s", vstr);
            if (nscan != 1)
            {
                NCNN_LOGE("ParamDict read value failed");
                return -1;
            }

            bool is_float = vstr_is_float(vstr);

            if (is_float)
            {
                d->params[id].f = vstr_to_float(vstr);
            }
            else
            {
                nscan = sscanf(vstr, "%d", &d->params[id].i);
                if (nscan != 1)
                {
                    NCNN_LOGE("ParamDict parse value failed");
                    return -1;
                }
            }

            d->params[id].type = is_float ? 3 : 2;
        }
    }

    return 0;
}
#endif // NCNN_STRING
```

dr.scan 对应的函数是 DataReaderFromMemory::scan，其定义如下所示:

```cpp
#if NCNN_STRING // 判断是否定义了 NCNN_STRING 宏
int DataReaderFromMemory::scan(const char* format, void* p) const
{
    // 获取给定格式字符串的长度
    size_t fmtlen = strlen(format);

    // 在原格式字符串后添加 '%n'，用于返回已经读取的字符数
    char* format_with_n = new char[fmtlen + 4];
    sprintf(format_with_n, "%s%%n", format);

    int nconsumed = 0; // 记录已经读取的字符数
    int nscan = sscanf((const char*)d->mem, format_with_n, p, &nconsumed); // 读取数据
    d->mem += nconsumed; // 更新指针，指向未读取的内存

    delete[] format_with_n; // 释放动态分配的内存

    return nconsumed > 0 ? nscan : 0; // 返回已经读取的字符数或者 0
}
#endif // NCNN_STRING

```

DataReaderFromMemory::scan 函数主要实现了**从内存中读取数据并按照给定的格式进行解析**。该函数首先获取给定格式字符串的长度，并在该字符串后添加 `%n`，用于返回已经读取的字符数。然后通过 `sscanf` 函数读取数据并更新指向未读取的内存的指针，最后返回已经读取的字符数或者 0。

值得注意的是，该函数代码依赖于 `NCNN_STRING` 宏，只有在定义了该宏时才会编译。

ParamDictPrivate 类定义如下所示:

```cpp
#define NCNN_MAX_PARAM_COUNT 32
class ParamDictPrivate
{
public:
    struct
    {
        // 0 = null
        // 1 = int/float
        // 2 = int
        // 3 = float
        // 4 = array of int/float
        // 5 = array of int
        // 6 = array of float
        int type;
        union
        {
            int i;
            float f;
        };
        Mat v;
    } params[NCNN_MAX_PARAM_COUNT];
};
```

NCNN_MAX_PARAM_COUNT 被宏定义为 32，表示 params 是一个大小为 32 的结构体数组，即模型参数文件每一行中特定参数数量不能超过 32。类中结构体的作用是存储参数值，可以根据 `type` 的值来确定参数类型。
