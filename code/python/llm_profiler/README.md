# llm_profiler
llm theoretical performance analysis tools and support params, flops, memory and latency analysis.

## 主要功能

- 支持张量并行、`pipeline` 并行推理模式。
- 支持 `A100`、`V100`、`T4` 等硬件以及主流 decoder-only 的自回归模型，可自行在配置文件中增加。
- 支持分析性能瓶颈，不同 `layer` 是 `memory bound` 还是 `compute bound`，以及 `kv_cache` 的性能瓶颈。
- 支持输出每层和整个模型的参数量、计算量，内存和 `latency`。
- 推理时支持预填充和解码阶段分别计算内存和 latency、以及理论支持的最大 `batch_size` 等等。
- 支持设置计算效率、内存读取效率（不同推理框架可能不一样，这个设置好后，可推测输出实际值）。
- 推理性能理论分析结果的格式化输出。

## 如何使用

使用方法，直接调用 `llm_profiler/llm_profiler.py` 文件中函数 `llm_profile()` 函数并输入相关参数即可。

```python
def llm_profile(model_name="llama-13b",
                gpu_name: str = "v100-sxm-32gb",
                bytes_per_param: int = BYTES_FP16,
                batch_size_per_gpu: int = 1,
                seq_len: int = 522,
                generate_len=1526,
                ds_zero: int = 0,
                dp_size: int = 1,
                tp_size: int = 1,
                pp_size: int = 1,
                sp_size: int = 1,
                use_kv_cache: bool = True,
                layernorm_dtype_bytes: int = BYTES_FP16,
                kv_cache_dtype_bytes: int = BYTES_FP16,
                flops_efficiency: float = FLOPS_EFFICIENCY,
                hbm_memory_efficiency: float = HBM_MEMORY_EFFICIENCY,
                intra_node_memory_efficiency=INTRA_NODE_MEMORY_EFFICIENCY,
                inter_node_memory_efficiency=INTER_NODE_MEMORY_EFFICIENCY,
                mode: str = "inference",
            ) -> dict:

    """format print dicts of the total floating-point operations, MACs, parameters and latency of a llm.

    Args:
        model_name (str, optional): model name to query the pre-defined `model_configs.json`. Defaults to "llama-13b".
        gpu_name (str, optional): gpu name to query the pre-defined `model_configs.json`. Defaults to "v100-sxm2-32gb".
        batch_size_per_gpu (int, optional): _description_. Defaults to 1.
        seq_len (int, optional): batch size per GPU.. Defaults to 522.
        generate_len (int, optional): The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt. Defaults to 1526.
        ds_zero (int, optional): which DeepSpeed ZeRO stage to use.. Defaults to 0.
        dp_size (int, optional): data parallelism size. Defaults to 1.
        tp_size (int, optional): tensor parallelism size. Defaults to 1.
        pp_size (int, optional): pipeline parallelism size. Defaults to 1.
        sp_size (int, optional): sequence parallelism size. Defaults to 1.
        use_kv_cache (bool, optional): Whether or not the model should use the past last key/values attentions (if applicable to the model) to
            speed up decoding. Defaults to True.
        layernorm_dtype_bytes (int, optional): number of bytes in the data type for the layernorm activations.. Defaults to BYTES_FP16.
        kv_cache_dtype_bytes (int, optional): number of bytes in the data type for the kv_cache. Defaults to None.
        flops_efficiency (float, optional): flops efficiency, ranging from 0 to 1. Defaults to None.
        hbm_memory_efficiency (float, optional): GPU HBM memory efficiency, ranging from 0 to 1. Defaults to HBM_MEMORY_EFFICIENCY.
        intra_node_memory_efficiency (_type_, optional): intra-node memory efficiency, ranging from 0 to 1.. Defaults to INTRA_NODE_MEMORY_EFFICIENCY.
        inter_node_memory_efficiency (_type_, optional): inter-node memory efficiency, ranging from 0 to 1.. Defaults to INTER_NODE_MEMORY_EFFICIENCY.
        mode (str, optional): model training or inference. Defaults to "inference".

    Returns:
        None: format print some summary dictionary of the inference analysis
    """
```

`llama-13b` 模型，tp_size = 8 和 batch_size = 103，FT 框架实际输出 latency 是 41.85 ms，理论输出 22.0 ms。输出示例信息如下所示：

```bash
-------------------------- LLM main infer config --------------------------
{   'inference_config': {   'model_name': 'llama-13b',
                            'batch_size_per_gpu': 103,
                            'seq_len': 522,
                            'tp_size': 8,
                            'pp_size': 1,
                            'generate_len': 1526,
                            'use_kv_cache': True},
    'gpu_config': {   'name': 'v100-sxm-32gb',
                      'memory_GPU_in_GB': '32 GB',
                      'gpu_hbm_bandwidth': '810.0 GB/s',
                      'gpu_intra_node_bandwidth': '270.0 GB/s',
                      'gpu_TFLOPS': '78.39999999999999 TFLOPS'}}

---------------------------- LLM Params analysis ----------------------------
{   'params_per_layer': '314.57 M',
    'params_attn': '104.86 M',
    'params_mlp': '209.72 M',
    'params_layernorm': '0'}
{'params_model': '12.75 G'}

---------------------------- LLM Flops analysis -----------------------------
{   'flops_fwd_per_layer': '34.4 T',
    'flops_attn': '11.85 T',
    'flops_mlp': '22.55 T',
    'flops_layernorm': '0'}
{'flops_model': '1393.68 T'}

---------------------------- LLM Memory analysis -----------------------------
{   'weight_memory_per_gpu': '3.19 GB',
    'prefill_activation_memory_batch_size_1': '26.73 MB',
    'prefill_max_batch_size_per_gpu': 1078,
    'prefill_activation_memory_per_gpu': '2.75 GB'}
{   'weight_memory_per_gpu': '3.19 GB',
    'decode_activation_memory_per_gpu': '5.27 MB',
    'kv_cache_memory_per_gpu': '21.62 GB',
    'decode_memory_total': '24.81 GB',
    'decode_max_batch_size_per_gpu': 137}

-------------------------- LLM infer performance analysis --------------------------
{   'model_params': '12.75 G',
    'model_flops': '1393.68 T',
    'prefill_first_token_latency': '2.51 s',
    'decode_per_token_latency': '22.0 ms',
    'kv_cache_latency': '16.76 ms',
    'total_infer_latency': '36.08 s'}

-------------------------- LLM detailed's latency analysis --------------------------
{   'prefill_latency_fwd_per_layer': {   'latency_per_layer': '61.99 ms',
                                         'latency_attn': '18.89 ms',
                                         'latency_mlp': '35.96 ms',
                                         'latency_layernorm': '0.0 us',
                                         'latency_tp_comm': '7.14 ms'},
    'prefill_latency_fwd_attn': '755.76 ms',
    'prefill_latency_fwd_mlp': '1.44 s',
    'prefill_latency_fwd_layernorm': '0.0 us',
    'prefill_latency_fwd_tp_comm': '285.48 ms',
    'prefill_latency_fwd_input_embedding': '3.97 ms',
    'prefill_latency_fwd_output_embedding_loss': '28.09 ms',
    'prefill_latency': '2.51 s'}
{   'decode_latency_fwd_per_layer': {   'latency_per_layer': '119.32 us',
                                        'latency_attn': '34.44 us',
                                        'latency_mlp': '68.88 us',
                                        'latency_layernorm': '0.0 us',
                                        'latency_tp_comm': '16.0 us'},
    'decode_latency_fwd_attn': '1.38 ms',
    'decode_latency_fwd_mlp': '2.76 ms',
    'decode_latency_fwd_layernorm': '0.0 us',
    'decode_latency_fwd_tp_comm': '640.0 us',
    'decode_latency_fwd_input_embedding': '412.54 us',
    'decode_latency_fwd_output_embedding_loss': '53.81 us',
    'kv_cache_avg_latency': '16.76 ms',
    'kv_cache_peak_latency': '26.69 ms',
    'decode_avg_latency': '22.0 ms',
    'decode_peak_latency': '31.93 ms'}
```

## TODO
- 支持训练模型理论性能分析
- 支持 零推理模式等理论性能分析

## 参考链接

- [llm_analysis](https://github.com/cli99/llm-analysis)
- [Transformer Inference Arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/)