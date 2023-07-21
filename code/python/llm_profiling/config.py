# -*- coding  : utf-8 -*-
# Description : gpu, model, Parallelism, data, train and inference config definition

import math, json
from constants import *
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering, unique

@unique
class ActivationRecomputation(Enum):
    NONE = 0
    """No activation recomputation; requires the most amount of memory."""

    SELECTIVE = 1
    """Selectively checkpoints and recomputes only parts of each transformer
    layer that take up a considerable amount of memory but are not
    computationally expensive to recompute, i.e. Q K V matrix multiplies, 
    QK^T matrix multiply, softmax, softmax dropout, and attention over V."""

    FULL = 2
    """Full activation recomputation stores the input to EVERY transformer
    layer, which is sharded across the tensor parallel group, thus requiring an
    extra all-gather (ignored for now) per layer and add communication
    overhead; requires the lease amount of memory; requires an extra forward
    pass."""
    
@total_ordering
class DSZeRO(Enum):
    NONE = 0
    """No DeepSPeed ZeRO; requires the most amount of memory."""

    STAGE_1 = 1
    """ZeRO stage 1 shards the optimizer states across the data parallel
    group."""

    STAGE_2 = 2
    """ZeRO stage 2 shards the optimizer states and gradients across the data
    parallel group."""

    STAGE_3 = 3
    """ZeRO stage 3 shards the optimizer states, gradients, and model weights
    across the data parallel group."""

    def __lt__(self, other):
        # 炫技写法
        if other.__class__ is self.__class__:
            return self.value < other.value # Enum 枚举类自动赋值
        return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, DSZeRO):
            return self.value == other.value
        return NotImplemented

@dataclass
class GPUEfficiencyConfig:
    flops_efficiency: float = 1.0
    hbm_memory_efficiency: float = 1.0
    intra_node_memory_efficiency: float = 1.0
    inter_node_memory_efficiency: float = 1.0
    
@dataclass
class InferenceConfig:
    """Inference configuration dataclass."""
    batch_size_per_gpu: int = None      # batch size
    seq_len: int = None         # input sequence length
    generate_len: int = None    # number of tokens to generate
    use_kv_cache: bool = True   # whether to use key/value cache
    bytes_per_param: int = BYTES_FP16  # model weight bytes
    layernorm_dtype_bytes: int = BYTES_FP16  # layernorm data type bytes
    kv_cache_dtype_bytes: int = BYTES_FP16   # key/value cache data type bytes
    def __post_init__(self):
        if self.context_len is None:
            self.context_len = self.seq_len + self.generate_len
        
@dataclass
class ParallelismConfig:
    """dataclass module provides a decorator and functions for automatically adding generated special methods 
    such as __init__() and __repr__() to user-defined classes
    """
    tp_size: int = 1  # tensor parallelism size, Megatron-LM tensor parallelism implementation
    pp_size: int = 1  # pipeline parallelism size, Megatron-LM pipeline parallelism implementation
    dp_size: int = 1  # data parallelism size, DeepSpeed Zero parallelism implementation
    sp_size: int = 1  # sequence parallelism size, Megatron-LM sequence parallelism implementation

@dataclass
class ModelConfig:
    name: str        # model config's key name
    num_layers: int  # number of transformer layers (blocks)
    n_head: int      # number of attention heads
    hidden_dim: int  # hidden dimension
    vocab_size: int  # vocabulary size
    max_seq_len: int = None   # max sequence length
    ffn_embed_dim: int = None # hidden dimension of FFN, default to 4 * hidden_dim
    model_type: str = None    # model type as tagged on Hugging Face (e.g., gpt2, opt, llama.)
    
    def __post_init__(self):
        if self.ffn_embed_dim is None:
            self.ffn_embed_dim = self.hidden_dim * 4
        
@dataclass
class GPUConfig:
    # 1, gpu 型号和显存大小
    name: str  # GPU config name
    memory_GPU_in_GB: float  # memory per GPU in GB
    
    # 2, gpu 显存带宽、节点内带宽、节点间带宽
    hbm_bandwidth_in_GB_per_sec: float  # GPU HBM bandwidth in GB/s
    intra_node_bandwidth_in_GB_per_sec: float  # intra node GPU bandwidth in GB/s.(PCIE/NVLINK)
    inter_node_bandwidth_in_GB_per_sec: float = 200  # inter node bandwidth in GB/s, assuming Mellanox 200Gbps HDR Infiniband
    
    # 3, 不同精度的 Tensor core 的计算性能
    peak_fp32_TFLOPS: float = None  # peak Tensor TFLOPS for FP32
    peak_fp16_TFLOPS: float         # peak Tensor TFLOPS for FP16
    peak_int8_TFLOPS: float = None  # peak Tensor TFLOPS for INT8
    peak_int4_TFLOPS: float = None  # peak Tensor TFLOPS for INT4
    
    intra_node_min_message_latency: float  # minimum intra node message latency in seconds

    def __post_init__(self):
        """object creation of DataClass starts with __init__() (constructor-calling) and 
        ends with __post__init__() (post-init processing).
        """
        if self.peak_fp32_TFLOPS is None:
            self.peak_fp32_TFLOPS =  math.ceil(self.peak_fp16_TFLOPS / 2)
        if self.peak_int8_TFLOPS is None:
            self.peak_int8_TFLOPS = 2 * self.peak_fp16_TFLOPS
        if self.peak_int4_TFLOPS is None:
            self.peak_int4_TFLOPS = 4 * self.peak_fp16_TFLOPS
            
class LLMConfigs(object):
    def __init__(self, gpu_config: GPUConfig = GPUConfig(),
                 model_config: ModelConfig = ModelConfig(),
                 parallelism_config: ParallelismConfig = ParallelismConfig(),
                 inference_config: InferenceConfig = InferenceConfig(),
                 gpu_efficiency_config: GPUEfficiencyConfig = GPUEfficiencyConfig()
                ) -> None:
        self.model_config = model_config
        self.gpu_config = gpu_config
        self.parallelism_config = parallelism_config
        self.inference_config = inference_config # 用户自行指定配置
        self.gpu_efficiency_config = gpu_efficiency_config # 用户自行指定配置
      
def get_model_and_gpu_config_by_name(model_name="llama-13b", gpu_name="v100-pcie-32gb") -> dict:
    """Read model and gpu configs from a json file."""
    config_files = ["configs/model_configs.json", "configs/gpu_configs.json"]
    model_config, gpu_config = {}, {}
    
    for config_filename in config_files:
        with open(config_filename, "r") as f:
            config_json = json.load(f)
            
            if "model" in config_filename:
                assert model_name in config_json, f"model name {model_name} not found in {config_filename}"
                config_dict = config_json[model_name]
                model_config = ModelConfig(**config_dict)
            
            elif "gpu" in config_filename:
                assert gpu_name in config_json, f"gpu name {gpu_name} not found in {config_filename}"
                config_dict = config_json[gpu_name]
                gpu_config = GPUConfig(**config_dict)
            else:
                assert False, f"unknown config type when reading: {type}"
            
    return model_config, gpu_config

def get_TFLOPS_per_gpu(gpu_config: GPUConfig, data_type="fp16", flops_efficiency=1.0) -> float:
    """Get the expected TFLOPS per GPU for the specified data type
    configuration/GPU (adjusted by flops_efficiency)

    Returns:
        float: TFLOPS per GPU and unit is T.
    """
    if data_type == "int8":
        gemm_TFOPS = gpu_config.peak_int8_TFLOPS
    elif data_type == "fp16":
        gemm_TFOPS = gpu_config.peak_fp16_TFLOPS
    else:
        print("weight_bits and activation_bits must be 8, or 16!")
    
    return gemm_TFOPS * flops_efficiency

def get_gpu_hbm_bandwidth(gpu_config: GPUConfig, hbm_memory_efficiency=1.0) -> float:
    return (
        gpu_config.hbm_bandwidth_in_GB_per_sec * hbm_memory_efficiency
    )
    
def get_intra_node_bandwidth(gpu_config: GPUConfig, intra_node_memory_efficiency=1.0) -> float:
    return (
        gpu_config.intra_node_bandwidth_in_GB_per_sec * intra_node_memory_efficiency
    )

def get_inter_node_bandwidth(gpu_config: GPUConfig, inter_node_memory_efficiency=1.0) -> float:
    return (
        gpu_config.inter_node_bandwidth_in_GB_per_sec * inter_node_memory_efficiency
    )