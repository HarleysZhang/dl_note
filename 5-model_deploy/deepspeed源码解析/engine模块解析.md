
## InferenceEngine(nn.Module) 类解析

### _apply_injection_policy 函数

该函数接受一个 `config` 参数和一个可选的 `client_module` 参数。它首先根据 `config` 中的 `checkpoint` 目录加载检查点，然后调用`generic_injection` 函数将注入策略应用到模型中。接下来，它检查 self.module 是否是 torch.nn.Module 类型的对象，如果是，则调用replace_transformer_layer 函数替换模型中的 Transformer 层。

## SDLoaderFactory 类

类包含了两个静态方法：
- `get_sd_loader_json`: 从 JSON 文件中获取 SDLoader 或者 checkpoints.json 文件中的数据。
- `get_sd_loader`:  根据指定的 SD 类型获取相应的 SDLoader 对象

```python
class SDLoaderFactory:

    @staticmethod
    def get_sd_loader_json(json_file, checkpoint_engine):
        """从 JSON 文件中获取 SDLoader 或者 checkpoints.json 文件中的数据。
        参数：
            - json_file: checkpoints.json 文件路径或包含 SDLoader 配置信息的字典
            - checkpoint_engine: TorchCheckpointEngine() 对象
        返回：
            - 如果 SD 类型为 'bloom' 或 'ds_model', 则返回原始的 SDLoader 配置数据，即 checkpoints.json 文件中的数据
            - 否则, 返回通过SDLoaderFactory.get_sd_loader()方法获取的SDLoader对象
        """
        
        if isinstance(json_file, str):
            with open(json_file) as f:
                # json.load 函数根据 JSON 数据的结构，json.load() 函数会将其解析为适当的 Python 数据类型，并返回对应的对象。
                data = json.load(f)
        else:
            assert isinstance(json_file, dict)
            data = json_file
        sd_type = data['type']
        ckpt_list = data['checkpoints']
        version = data['version']
        ckpt_type = data.get('parallelization', 'pp')
        mp_size = data.get('mp_size', 0)
        if sd_type.lower() in ['bloom', 'ds_model']:
            return data
        return SDLoaderFactory.get_sd_loader(ckpt_list, checkpoint_engine, sd_type, version)

    @staticmethod
    def get_sd_loader(ckpt_list, checkpoint_engine, sd_type='Megatron', version=None):
        if sd_type == 'Megatron':
            return MegatronSDLoader(ckpt_list, version, checkpoint_engine)
        else:
            assert False, '{} checkpoint type is not supported'.format(sd_type)
```

## get_accelerator() 和 set_rng_state() 函数

- `get_accelerator` 功能: 返回获取到的 DeepSpeed 加速器对象。
- `set_rng_state` 功能: 该函数的作用是设置 CUDA 随机数生成器（RNG）的状态。