# -*- coding  : utf-8 -*-
# author      : honggao.zhang
# Create      : 2023-7-19
# Version     : 0.1.0
# Description : transformer model(llm) profiling tools, can be used to profile the model's flops, memory, and latency.
# Reference   : https://github.com/cli99/llm-analysis

import json
import logging
from pprint import pformat
import pprint

from config import *
from utils import *

logger = logging.getLogger()
    
class CountCausalLMParams(object):
    def __init__(self, model_config: ModelConfig) -> None:
        self.h = model_config.hidden_dim
        self.l = model_config.num_layers
        self.V = model_config.vocab_size
        
        self.model_config = model_config

    def count_params_embedding(self, shared_embedding: bool = True) -> int:
        """Get the number of parameters in the embedding layer. params_te = vocab_size * d_model
        Args:
            shared_embedding (bool, optional):  whether the output embedding \
                shares weights with the input embedding. Defaults to True.

        Returns: 
            int: the number of parameters in the embedding layer
        """
        num_params_input_embedding = self.V * self.h
        num_params_output_embedding = self.V * self.h if not shared_embedding else 0
        
        return num_params_input_embedding + num_params_output_embedding
    
    def count_params_per_layer_attn(self) -> int:
        """Get the number of parameters per layer in the attention module 
        which include 4 linear layer: query/key/value projection and output matrices.
        params_attn(mha) = params_q + params_k + params_v + params_o = 4 * d_model**2
        
        Returns:
            int: the number of parameters per layer in the attention module(mha)
        """
        return 4 * self.h ** 2
    
    def count_params_per_layer_mlp(self) -> int:
        """Get the number of parameters in the MLP linear layers, including the
        intermediate and output matrices.
        params_mlp = prams_fc1 + params_fc2 = d_model * 4_d_model + 4_d_model * d_model = 8 * d_model**2
        
        Returns:
            int: the number of parameters in the two MLP linear layers
        """
        
        return 8 * self.h ** 2
    
    def count_params_per_layer_ln(self) -> int:
        """Get the number of parameters per layer in the two layer normalization module.
        params_ln = 4 * d_model
        
        Returns:
            int: the number of parameters per layer in the two layer normalization module
        """
        return 4 * self.h
    
    def count_params_per_layer(self, ln_ignore=True) -> tuple:
        """Get the number of params per layer in the transformer decoder blocks,
        mainly including the attention and MLP layers
        
        params_per_layer = params_attn + params_mlp + params_ln 
                         = 4d_model^2 + 8d_model^2 + 2*4d_model = 12d_model^2 + 8d_model
        
        Return:
            int: the number of params per layer in the transformer decoder blocks
        """
        params_per_layer_attn = self.count_params_per_layer_attn()
        params_per_layer_mlp = self.count_params_per_layer_mlp()
        params_per_layer_ln = 0 if ln_ignore else 2 * self.count_params_per_layer_ln()
        
        params_per_layer = (
            params_per_layer_attn
            + params_per_layer_mlp
            + params_per_layer_ln
        )
                
        dict_params_per_layer = {
            "params_per_layer": params_per_layer,
            "params_attn": params_per_layer_attn,
            "params_mlp": params_per_layer_mlp,
            "params_layernorm": params_per_layer_ln,
        }
        
        return params_per_layer, dict_params_per_layer
                
    def count_params_model(self) -> int:
        """Get the total number of parameters in the model including all layers and token embedding layer.
        params_model = params_embedding + params_per_layer * num_layers 
                    = V * d_model + 12 * d_model**2 * num_layers
        Returns:
            int: the total number of parameters in the model
        """
        params_per_layer, dict_params_per_layer = self.count_params_per_layer()
        
        return (params_per_layer * self.l
                + self.count_params_embedding()
        )
        
    def __call__(self, hidden_dim, num_layers, vocab_size) -> int:
        
        return (vocab_size * hidden_dim 
                + 12 * hidden_dim ** 2 * num_layers
            )


class CountCausalLMFlops(object):
    """The count is model-specific and does not depend on the parallelism strategy.
       And ignore layer normalization and other element-wise operations."""
    def __init__(self, model_config: ModelConfig, batch_size: int, seq_len: int, simp_count=False) -> None:
        self.h = model_config.hidden_dim
        self.l = model_config.num_layers
        self.V = model_config.vocab_size
        
        self.b = batch_size
        self.s = seq_len
        
        if not simp_count:
            llm_params = CountCausalLMParams(model_config)
            self.model_flops = llm_params(self.h, self.l, self.V) * 2
        
    def count_flops_fwd_per_layer_attn(self, batch_size: int, seq_len: int) -> int:
        """Get the number of floating point operations (flops) for the forward
        pass of the attention module in a transformer layer, given the batch
        size and sequence length. 

        mainly including four linear calculations: query/key/value projection and output 
        matrices multiplication、self-attention internal operation, and element-wise operations are ignored.
        
        flops_attn = flops_q + flops_k + flops_v + flops_output + flops_self_attention
              = 4(bsh^2) + 2(2bs^2h)
        Args:
            batch_size (int): batch size
            seq_len (int): sequence length

        Returns:
            int: flops for the forward pass of the attention module in a transformer layer
        """
        return (
            8 * batch_size * seq_len * self.h ** 2 
            + 4 * batch_size * seq_len ** 2 * self.h
        )
        
    def count_flops_fwd_per_layer_mlp(self, batch_size: int, seq_len: int) -> int:
        """Count two flops of matrices multiplication(two linear layers in the MLP module.)
        
        flops_mlp = flops_fc1 + flops_fc2 = 2bs(4h^2) + 2bs(4h^2) = 16bsh^2
        """
        return 16 * batch_size * seq_len * self.h ** 2
        
    def count_flops_fwd_per_layer(self, batch_size: int, seq_len: int, ln_ignore=True) -> tuple:
        flops_fwd_per_layer_attn = self.count_flops_fwd_per_layer_attn(batch_size, seq_len)
        flops_fwd_per_layer_mlp = self.count_flops_fwd_per_layer_mlp(batch_size, seq_len)
        flops_fwd_per_layer_ln = 0
        
        flops_fwd_per_layer = (
            flops_fwd_per_layer_attn
            + flops_fwd_per_layer_mlp
            + flops_fwd_per_layer_ln
        )
        
        dict_flops_fwd_per_layer = {
            "flops_fwd_per_layer": flops_fwd_per_layer,
            "flops_attn": flops_fwd_per_layer_attn,
            "flops_mlp": flops_fwd_per_layer_mlp,
            "flops_layernorm": flops_fwd_per_layer_ln,
        }
                
        return flops_fwd_per_layer, dict_flops_fwd_per_layer
      
    def count_flops_logits_layer(self,) -> int:
        """flops of output token logits layer"""
        return 2 * self.b * self.s * self.h * self.V
    
    def count_flops_fwd_model(self, batch_size: int, seq_len: int) -> int:
        """Count flops of the forward pass of the transformer model, given the batch size and sequence length."""
        num_flops_fwd_model = (
            self.count_flops_fwd_per_layer(batch_size, seq_len)[0] * self.l
            + self.count_flops_logits_layer()
        )
        
        # validate
        assert within_range(
            num_flops_fwd_model,
            (
                24 * self.b * self.s * self.l * self.h**2
                * (1 + self.s / (6 * self.h) + self.V / (12 * self.l * self.h))
            ),
            TOLERANCE,
        )
        
        return num_flops_fwd_model
    
    def count_flops_bwd_model(self, batch_size: int, seq_len: int) -> int:
        """Get the number of floating point operations (flops) for the backward
        pass of the entire transformer model, given the batch size and sequence"""
        return 2 * self.count_flops_fwd_model(batch_size, seq_len)

   
class CountCausalLMMemory(object):
    """Count memory of the model and layers."""
    def __init__(self, llm_configs: LLMConfigs) -> None:
        self.model_config = llm_configs.model_config
        self.h = self.model_config.hidden_dim
        self.l = self.model_config.num_layers
        self.V = self.model_config.vocab_size
        
        self.b = llm_configs.inference_config.batch_size_per_gpu
        self.s = llm_configs.inference_config.seq_len
        self.o = llm_configs.inference_config.generate_len

        self.bytes_per_param = llm_configs.inference_config.bytes_per_param
        
        self.tp_size = llm_configs.parallelism_config.tp_size
        self.pp_size = llm_configs.parallelism_config.pp_size
        self.num_layers_per_gpu = int(self.l / self.pp_size)
        
        self.gpu_memory_in_GB = llm_configs.gpu_config.memory_GPU_in_GB * 10**9  # 单位 GB
        
        self.llm_params = CountCausalLMParams(self.model_config)
            
    def count_memory_weights(self, embedding_dtype_bytes: int = BYTES_FP16):
        """Get the memory of the model weights"""
        params_per_layer, dict_params_per_layer = self.llm_params.count_params_per_layer()
        params_embedding = self.llm_params.count_params_embedding()
        
        memory_weight_per_layer = (
            (params_per_layer / self.tp_size) * self.bytes_per_param
        )
        memory_weight_per_gpu = memory_weight_per_layer *  self.num_layers_per_gpu
        
        memory_embedding = (params_embedding / self.tp_size) * embedding_dtype_bytes
        memory_weight_per_gpu = memory_weight_per_gpu + memory_embedding
        
        return memory_weight_per_gpu
    
    def count_memory_activation_per_layer_attn(
        self,
        batch_size: int,
        seq_len: int,
        is_inference: bool = True,
        activation_recomputation: ActivationRecomputation = ActivationRecomputation.FULL
    ) -> float:
        """Count the memory (in bytes) required  to store the activations of the
        attention in a transformer layer, given the batch size, sequence length, 
        whether it is inference or training, the activation recomputation strategy, 
        and the activation data type.
        """
        if activation_recomputation == ActivationRecomputation.FULL:
            return (batch_size * seq_len * self.h / self.tp_size) * self.bytes_per_param 
    
    def count_memory_activation_per_layer_mlp(
        self,
        is_inference: bool = True,
        activation_recomputation: ActivationRecomputation = ActivationRecomputation.FULL,
    ) -> float:
        """ The `mlp` activations include the input to the two linear layers."""
        if activation_recomputation == ActivationRecomputation.FULL:
            return 0
        
        return 0
    def count_memory_activation_per_layer_layernorm(
        self,
        is_inference: bool = True,
        activation_recomputation: ActivationRecomputation = ActivationRecomputation.FULL,
        layernorm_dtype_bytes: int = BYTES_FP16
    ) -> float:
        if activation_recomputation == ActivationRecomputation.FULL:
            return 0
        return 0
    
    def count_memory_activation_per_layer(
        self,
        batch_size: int,
        seq_len: int,
        is_inference: bool = True,
        activation_recomputation: ActivationRecomputation = ActivationRecomputation.FULL,
        layernorm_dtype_bytes: int = BYTES_FP16
    ) -> float:
        
        if activation_recomputation == ActivationRecomputation.FULL:
            return (
                (batch_size * seq_len * self.h / self.tp_size) * self.bytes_per_param
            )
        return 0
          
    def count_memory_kv_cache_per_layer(
        self,
        batch_size: int,
        seq_len: int,
        generate_len: int,
        kv_cache_dtype_bytes: int = BYTES_FP16,
    ) -> float:
        """Get the memory (in bytes) required to store the key and value cache
        for a transformer layer in inference, given the batch size, sequence
        length, activation data type, and tensor parallelism size.
        
        memory_kv_cache = 4blh(s+o) unit is byte
        Args:
            batch_size (int): batch size
            context_len (int): seq_len + generate_len
            
        Returns:
            float: the memory (in bytes) required  to store the key and value cache for a transformer layer in inference
        """
        
        return (
            (2 * batch_size * (seq_len + generate_len) * self.h) / self.tp_size
        ) * kv_cache_dtype_bytes
    
    def count_memory_per_gpu(
        self, 
        batch_size: int,
        seq_len: int,
        generate_len: int,
        is_inference: bool = True,
        use_kv_cache: bool = True,
        activation_recomputation: ActivationRecomputation = ActivationRecomputation.FULL,
        layernorm_dtype_bytes: int = BYTES_FP16,
        kv_cache_dtype_bytes: int = BYTES_FP16
    ) -> tuple:
        
        # 1, prefill stage count memory and max_batch_size
        
        weight_memory_per_gpu = self.count_memory_weights() # count model weights memory
        memory_left = self.gpu_memory_in_GB - weight_memory_per_gpu

        prefill_activation_memory_batch_size_1 = ( # count model activations and kv cache memory of prefill stage
            self.count_memory_activation_per_layer(
                1, seq_len, is_inference, ActivationRecomputation.FULL, layernorm_dtype_bytes
            )
            * self.num_layers_per_gpu
        )
        
        prefill_max_batch_size_per_gpu = int(
            memory_left / prefill_activation_memory_batch_size_1
        )

        prefill_activation_memory_per_gpu = (
            self.count_memory_activation_per_layer(
                batch_size, seq_len, is_inference, ActivationRecomputation.FULL, layernorm_dtype_bytes
            )
            * self.num_layers_per_gpu
        )
        
        assert memory_left > prefill_activation_memory_per_gpu, (
            f"weight_memory_per_gpu {num_to_string(weight_memory_per_gpu)}, activation memory {num_to_string(prefill_activation_memory_per_gpu)} is too large can't fit in GPU memory! memory_left is {num_to_string(memory_left)}!"
        )
        
        # 2, decode stage count memory and max_batch_size
        if use_kv_cache:
            kv_cache_memory_batch_size_1 = (
                self.count_memory_kv_cache_per_layer(
                    1,
                    seq_len + generate_len,
                    kv_cache_dtype_bytes
                )
                * self.num_layers_per_gpu
            )
            
            kv_cache_memory_per_gpu = (
                self.count_memory_kv_cache_per_layer(
                    batch_size,
                    seq_len + generate_len,
                    kv_cache_dtype_bytes
                )
                * self.num_layers_per_gpu
            )
            
            decode_activation_memory_batch_size_1 = (
                # seq_len 1 is used for decoding
                self.count_memory_activation_per_layer(
                    1, 1, is_inference, ActivationRecomputation.FULL, layernorm_dtype_bytes
                )
                * self.num_layers_per_gpu
            )
            
            decode_activation_memory_per_gpu = (
                # seq_len 1 is used for decoding
                self.count_memory_activation_per_layer(
                    batch_size, 1, is_inference, ActivationRecomputation.FULL, layernorm_dtype_bytes
                )
                * self.num_layers_per_gpu
            )
            
            decode_max_batch_size_per_gpu = int(
                memory_left / (decode_activation_memory_batch_size_1 + kv_cache_memory_batch_size_1) 
            )
            
            assert batch_size <= decode_max_batch_size_per_gpu, (
                f"batch_size_per_gpu {batch_size} is too large to fit"
                " in GPU memory, decode_max_batch_size_per_gpu:"
                f" {decode_max_batch_size_per_gpu}"
            )
            
            assert memory_left > (
                kv_cache_memory_per_gpu + decode_activation_memory_per_gpu
            ), ("kv_cache and activation memory with batch_size_per_gpu ="
                f" {batch_size} is too large to fit in GPU memory"
            )
        else:
            # 上下文长度不再是新生成的那个 token，而是 seq_len + generate_len
            decode_activation_memory_batch_size_1 = (
                self.count_memory_activation_per_layer(
                    1, seq_len + generate_len, True, ActivationRecomputation.FULL, layernorm_dtype_bytes
                )
                * self.num_layers_per_gpu
            )
            decode_max_batch_size_per_gpu = int(
                memory_left / decode_activation_memory_batch_size_1
            )
            assert batch_size <= decode_max_batch_size_per_gpu, (
                f"batch_size {batch_size} is too large to fit"
                " in GPU memory, decode_max_batch_size_per_gpu:"
                f" {decode_max_batch_size_per_gpu}"
            )
            
            decode_activation_memory_per_gpu = (
                self.count_memory_activation_per_layer(
                    batch_size, seq_len + generate_len, True, ActivationRecomputation.FULL, layernorm_dtype_bytes
                )
                * self.num_layers_per_gpu
            )
            kv_cache_memory_per_gpu = 0
        
        decode_memory_total = (weight_memory_per_gpu + decode_activation_memory_per_gpu + kv_cache_memory_per_gpu)
        
        # memory summary
        memory_prefill_summary_dict = {
            "weight_memory_per_gpu": weight_memory_per_gpu,
            "prefill_activation_memory_batch_size_1": prefill_activation_memory_batch_size_1,
            "prefill_max_batch_size_per_gpu": prefill_max_batch_size_per_gpu,
            "prefill_activation_memory_per_gpu": prefill_activation_memory_per_gpu, 
        }
        
        memory_decode_summary_dict = {
            "weight_memory_per_gpu": weight_memory_per_gpu,
            "decode_activation_memory_per_gpu": decode_activation_memory_per_gpu,
            "kv_cache_memory_per_gpu": kv_cache_memory_per_gpu,
            "decode_memory_total": decode_memory_total,
            "decode_max_batch_size_per_gpu": decode_max_batch_size_per_gpu,
        }
        
        return memory_prefill_summary_dict, memory_decode_summary_dict
    
    
class CountCausalLMLatency(object):
    """Count latency by roof-line performance model."""
    def __init__(self, llm_configs: LLMConfigs, data_type="fp16") -> None:
        self.model_config = llm_configs.model_config
        self.gpu_config = llm_configs.gpu_config
        self.inference_config = llm_configs.inference_config
        self.parallelism_config = llm_configs.parallelism_config
        
        self.h = self.model_config.hidden_dim
        self.l = self.model_config.num_layers
        self.V = self.model_config.vocab_size
        
        self.b = llm_configs.inference_config.batch_size_per_gpu
        self.s = llm_configs.inference_config.seq_len
        self.o = llm_configs.inference_config.generate_len
        self.bytes_per_param = llm_configs.inference_config.bytes_per_param
        
        self.tp_size = self.parallelism_config.tp_size
        self.pp_size = self.parallelism_config.pp_size
        self.num_layers_per_gpu = int(self.l / self.parallelism_config.pp_size)
        
        self.gpu_hbm_bandwidth = get_gpu_hbm_bandwidth(self.gpu_config) * 10**9 # 单位 GB/s
        self.gpu_intra_node_bandwidth = get_intra_node_bandwidth(self.gpu_config) * 10**9       # 互连带宽，单位 GB/s
        self.gpu_TFLOPS = get_TFLOPS_per_gpu(self.gpu_config) * 10**12           # 单位 TFLOPS
        
        self.gpu_memory_in_GB = llm_configs.gpu_config.memory_GPU_in_GB * 10**9  # 单位 GB
        
        self.llm_params = CountCausalLMParams(self.model_config)
        self.llm_memory = CountCausalLMMemory(llm_configs)
        self.llm_flops = CountCausalLMFlops(self.model_config, self.b, self.o)
        
    def common_count_latency_for_ops(
        self, 
        batch_size: int, 
        seq_len: int, 
        is_inference=True,
        activation_recomputation: ActivationRecomputation = ActivationRecomputation.FULL,
        ops_type: str="attn"
    ) -> float:
        """Count the latency for the forward layer or model, assuming the compute and memory operations are perfectly overlapped.

        Args:
            flops (float): flops of the forward layer or model
            memory (float): r/w memory(bytes) of the forward layer or model
            tp_size (float): tensor parallelism size
            gpu_TFLOPS (float): GPU TFLOPS in T(10^12)FLOPS
            gpu_hbm_bandwidth (float): GPU HBM bandwidth in GB/s(10^9)

        Returns:
            float: the latency in seconds for the forward pass
        """
        
        if ops_type=="attn":
            
            flops = self.llm_flops.count_flops_fwd_per_layer_attn(batch_size, seq_len)
            weight_memory = self.llm_params.count_params_per_layer_attn() * self.bytes_per_param
            activation_memory = self.llm_memory.count_memory_activation_per_layer_attn(
                                batch_size, seq_len, is_inference, activation_recomputation
            )
        elif ops_type=="mlp":
            flops = self.llm_flops.count_flops_fwd_per_layer_mlp(batch_size, seq_len)
            weight_memory = self.llm_params.count_params_per_layer_mlp() * self.bytes_per_param
            activation_memory = self.llm_memory.count_memory_activation_per_layer_mlp(is_inference, activation_recomputation)
        elif ops_type=="layernorm":
            activation_memory = self.llm_memory.count_memory_activation_per_layer_layernorm(
                                is_inference, activation_recomputation) # activation_memory
            weight_memory = 0   # layernorm has no matrix weight, only vector weight, is ignored
            flops = 0   # layernorm is not compute bound, flops is very small
        else:    
            print("error! unsupported ops_type")
        
        activation_memory = 0
        
        memory = weight_memory + activation_memory
        
        compute_latency = flops / (self.tp_size * self.gpu_TFLOPS) # 单位秒
        memory_latency = memory / (self.tp_size * self.gpu_hbm_bandwidth)
        
        if memory_latency > compute_latency:
            print(f"memory_latency {latency_to_string(memory_latency)} > compute_latency {latency_to_string(compute_latency)}, this {ops_type} layer is memory bound!")
        else:
            print(f"memory_latency {latency_to_string(memory_latency)} <= compute_latency {latency_to_string(compute_latency)}, this {ops_type} layer is compute bound!")
            
        return max(compute_latency, memory_latency)
    
    def count_latency_fwd_per_layer_tp_comm(self, batch_size: int, seq_len: int) -> float:
        """Count the latency of a single allreduce communication across the
        tensor parallel group in the forward pass of a transformer layer.
        The latency is the max of the latency for the allreduce and the minimum 
        message latency through intra-node connect.
        """
        
        if self.tp_size == 1:
            return 0
        
        # \phi is communication data, if tp_size is large enough num_data_per_all_reduce can be 2bsh
        num_data_per_all_reduce = (
            2 * batch_size * seq_len * self.h * 
            (self.tp_size - 1) / (self.tp_size)
        )

        latency_per_all_reduce = (
            num_data_per_all_reduce * self.bytes_per_param
            / (self.gpu_intra_node_bandwidth)
        )
        
        # intra_node_min_message_latency: 节点内连接的最小消息延迟
        return max(
            latency_per_all_reduce,
            self.gpu_config.intra_node_min_message_latency,
        )
        
    def count_latency_fwd_per_layer(
        self, 
        batch_size: int, 
        seq_len: int, 
        is_inference: bool=True,
        activation_recomputation: ActivationRecomputation = ActivationRecomputation.FULL,
        layernorm_dtype_bytes: int = BYTES_FP16
    ) -> tuple:
        latency_fwd_per_layer_attn = self.common_count_latency_for_ops(batch_size, seq_len, is_inference, activation_recomputation, ops_type="attn")
        latency_fwd_per_layer_mlp = self.common_count_latency_for_ops(batch_size, seq_len, is_inference, activation_recomputation, ops_type="mlp")
        latency_fwd_per_layer_layernorm = self.common_count_latency_for_ops(batch_size, seq_len, is_inference, activation_recomputation, "layernorm")
        
        latency_fwd_per_layer_tp_comm = self.count_latency_fwd_per_layer_tp_comm(batch_size, seq_len)
    
        latency_per_layer = (
            latency_fwd_per_layer_attn
            + latency_fwd_per_layer_mlp
            + 2 * latency_fwd_per_layer_layernorm   # 2 个 layernorm 层
            + 2 * latency_fwd_per_layer_tp_comm     # 一次 AllReduce 产生的通讯量为 2bsh
        )
    
        dict_latency_per_layer = {
            "latency_per_layer": (latency_per_layer),
            "latency_attn": (latency_fwd_per_layer_attn),
            "latency_mlp": (latency_fwd_per_layer_mlp),
            "latency_layernorm": (2 * latency_fwd_per_layer_layernorm),
            "latency_tp_comm": (2 * latency_fwd_per_layer_tp_comm),
        }

        return latency_per_layer, dict_latency_per_layer
    
    def count_latency_fwd_input_embedding(
        self, batch_size: int, seq_len: int
    ) -> float:
        """Get the latency for the forward pass of the input embedding layer,
        given the batch size, sequence length, and data type of the embedding
        weight.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length
            dtype_bytes (int, optional): number of bytes in the data type for the embedding weight. Defaults to BYTES_FP32.

        Returns:
            float: the latency in seconds for the forward pass of the input embedding layer
        """
        memory_latency = (
            self.model_config.vocab_size
            * self.model_config.hidden_dim
            * self.bytes_per_param
            / (self.gpu_hbm_bandwidth)
        )
        comm_latency = self.count_latency_fwd_per_layer_tp_comm(
            batch_size, seq_len
        )
        return memory_latency + comm_latency
    
    def count_latency_fwd_output_embedding_loss(
        self, batch_size: int, seq_len: int
    ) -> float:
        """Get the latency for the forward pass of the output embedding layer (computing the logits). The operation is compute bound. With tensor parallelism size > 1, an allgather communicates `batch_size * seq_len` elements, which is ignored here. Refer to https://arxiv.org/abs/1909.08053 for more details.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length

        Returns:
            float: the latency in seconds for the forward pass of the output embedding layer
        """
        
        compute_latency = (
            2 * batch_size * seq_len  * self.h * self.V
            / self.tp_size
            / self.gpu_TFLOPS
        )
        
        return compute_latency
    
    def count_latency_kv_cache(
        self, 
        batch_size: int, 
        seq_len: int, 
        generate_len: int,
        use_kv_cache: bool = True,
        kv_cache_dtype_bytes: int = BYTES_FP16
    ) -> tuple:
        """Get the latency for the forward pass of the key and value cache in a transformer layer, given the batch size, sequence length, and whether the key and value cache is used.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length
            generate_len (int): number of tokens to generate
            use_kv_cache (bool, optional): whether the key and value cache is used. Defaults to True.

        Returns:
            float: the latency in seconds for the forward pass of the key and value cache in a transformer layer
        """
        if not use_kv_cache:
            return 0
        kv_cache_memory_list_per_gpu, kv_cache_latency_list = [], []
        
        for context_len in range(seq_len, seq_len + generate_len + 1):
            kv_cache_memory_per_gpu = (
                self.llm_memory.count_memory_kv_cache_per_layer(
                    batch_size,
                    context_len,
                    kv_cache_dtype_bytes
                ) * self.num_layers_per_gpu
            )
            
            kv_cache_latency = (
                kv_cache_memory_per_gpu / self.gpu_hbm_bandwidth
            )

            kv_cache_memory_list_per_gpu.append(kv_cache_memory_per_gpu)
            kv_cache_latency_list.append(kv_cache_latency)
        
        kv_cache_avg_latency = average(kv_cache_latency_list)
        kv_cache_peak_latency = max(kv_cache_latency_list)
        
        return kv_cache_avg_latency, kv_cache_peak_latency
        
    def count_latency_fwd_model(
        self,
        batch_size: int,
        seq_len: int,
        is_inference: bool = True,
        activation_recomputation: ActivationRecomputation = ActivationRecomputation.FULL,
        layernorm_dtype_bytes: int = BYTES_FP32,
        breakdown_prefix: str = "",
    ) -> tuple:
        latency_fwd_per_layer, breakdown_per_layer = self.count_latency_fwd_per_layer(
            batch_size,
            seq_len,
            is_inference,
            activation_recomputation,
            layernorm_dtype_bytes,
        )
        num_layers_per_gpu = self.num_layers_per_gpu
        
        latency_fwd_all_layers = latency_fwd_per_layer * self.num_layers_per_gpu
        latency_fwd_input_embedding = self.count_latency_fwd_input_embedding(batch_size, seq_len)
        latency_fwd_output_embedding_loss = self.count_latency_fwd_output_embedding_loss(batch_size, seq_len)
        
        model_latency = (
            latency_fwd_all_layers
            + latency_fwd_input_embedding
            + latency_fwd_output_embedding_loss
        )
        
        model_latency_breakdown = {
            breakdown_prefix + "latency_fwd_per_layer": breakdown_per_layer,
            breakdown_prefix + "latency_fwd_attn": (breakdown_per_layer["latency_attn"] * num_layers_per_gpu),
            breakdown_prefix + "latency_fwd_mlp": (breakdown_per_layer["latency_mlp"] * num_layers_per_gpu),
            breakdown_prefix + "latency_fwd_layernorm": (breakdown_per_layer["latency_layernorm"] * num_layers_per_gpu),
            breakdown_prefix + "latency_fwd_tp_comm": (breakdown_per_layer["latency_tp_comm"] * num_layers_per_gpu),
            breakdown_prefix + "latency_fwd_input_embedding": (latency_fwd_input_embedding),
            breakdown_prefix + "latency_fwd_output_embedding_loss": (latency_fwd_output_embedding_loss),
        }
        
        return model_latency, model_latency_breakdown
    
    def count_latency_fwd(
        self,
        batch_size: int,
        seq_len: int,
        generate_len: int,
        use_kv_cache: bool = True,
        kv_cache_dtype_bytes: int = BYTES_FP16,
        is_inference: bool = True,
        activation_recomputation: ActivationRecomputation = ActivationRecomputation.FULL,
        layernorm_dtype_bytes: int = BYTES_FP32,
    ) -> tuple:
        # 1, 预填充阶段
        prefill_latency, prefill_latency_breakdown = self.count_latency_fwd_model(
            batch_size,
            seq_len,
            is_inference=is_inference,
            layernorm_dtype_bytes=layernorm_dtype_bytes,
            breakdown_prefix="prefill_",
        )
        
        prefill_latency_breakdown.update(
            {
                "prefill_latency": prefill_latency,
            }
        )
         
        # 2, 解码阶段
        kv_cache_avg_latency, kv_cache_peak_latency = self.count_latency_kv_cache(
            batch_size, 
            seq_len,
            generate_len,
            use_kv_cache,
            kv_cache_dtype_bytes
        ) 
        
        decode_model_latency, decode_latency_breakdown = self.count_latency_fwd_model(
            batch_size,
            1 if use_kv_cache else (seq_len + generate_len) * (2/3), # k、v cache 占 2/3，重新计算
            is_inference=is_inference,
            activation_recomputation=activation_recomputation,
            layernorm_dtype_bytes=layernorm_dtype_bytes,
            breakdown_prefix="decode_",
        )
        
        decode_avg_latency = decode_model_latency + kv_cache_avg_latency
        decode_peak_latency = decode_model_latency + kv_cache_peak_latency
        
        decode_latency_breakdown.update(
            {
                "kv_cache_avg_latency": (kv_cache_avg_latency),
                "kv_cache_peak_latency": (kv_cache_peak_latency),
                "decode_avg_latency": (decode_avg_latency),
                "decode_peak_latency": (decode_peak_latency)
            }
        )
        
        return prefill_latency_breakdown, decode_latency_breakdown

          
class LLMProfiler(object):
    """Measures the latency, memory, number of estimated floating-point operations and parameters of each module in a PyTorch model."""
    def __init__(self, llm_configs: LLMConfigs) -> None:
        self.model_config = llm_configs.model_config
        self.gpu_config = llm_configs.gpu_config
        self.inference_config = llm_configs.inference_config
        self.parallelism_config = llm_configs.parallelism_config
        self.gpu_efficiency_config = llm_configs.gpu_efficiency_config
        
        self.h = self.model_config.hidden_dim
        self.l = self.model_config.num_layers
        self.V = self.model_config.vocab_size
        
        self.b = llm_configs.inference_config.batch_size_per_gpu
        self.s = llm_configs.inference_config.seq_len
        self.o = llm_configs.inference_config.generate_len
        self.bytes_per_param = llm_configs.inference_config.bytes_per_param
        
        self.tp_size = self.parallelism_config.tp_size
        self.pp_size = self.parallelism_config.pp_size
        self.num_layers_per_gpu = int(self.l / self.parallelism_config.pp_size)
        
        self.gpu_hbm_bandwidth = get_gpu_hbm_bandwidth(self.gpu_config) * 10**9 # 单位 GB/s
        self.gpu_intra_node_bandwidth = get_intra_node_bandwidth(self.gpu_config) * 10**9       # 互连带宽，单位 GB/s
        self.gpu_TFLOPS = get_TFLOPS_per_gpu(self.gpu_config) * 10**12           # 单位 TFLOPS
        
        self.gpu_memory_in_GB = llm_configs.gpu_config.memory_GPU_in_GB * 10**9  # 单位 GB
        
        self.llm_params = CountCausalLMParams(self.model_config)
        self.llm_flops = CountCausalLMFlops(self.model_config, self.b, self.s)
        self.llm_memory = CountCausalLMMemory(llm_configs)
        self.llm_latency = CountCausalLMLatency(llm_configs)
    
    def infer_profile(
        self, 
        batch_size_per_gpu: int = 1, 
        seq_len: int = 522, 
        generate_len: int = 1526,
        use_kv_cache: bool = True,
        activation_recomputation: ActivationRecomputation = ActivationRecomputation.FULL,
        layernorm_dtype_bytes: int = 2,
        kv_cache_dtype_bytes: int = 2,
        flops_efficiency: float = None,
        hbm_memory_efficiency: float = HBM_MEMORY_EFFICIENCY,
        intra_node_memory_efficiency=INTRA_NODE_MEMORY_EFFICIENCY,
        inter_node_memory_efficiency=INTER_NODE_MEMORY_EFFICIENCY
    ) -> dict:
        """LLM inference analysis given the llm configs and inputs.

        Args:
            generate_len (int, optional): number of tokens to generate for generative models. Defaults to 100.
            use_kv_cache (bool, optional): whether to use kv_cache. Defaults to True.
            layernorm_dtype_bytes (int, optional): number of bytes in the data type for the layernorm activations. Defaults to BYTES_FP32. 
                Often has to be at least FP16 in inference to maintain model accuracy.

        Returns:
            dict: a summary dict of the training analysis
        """
        if self.model_config.max_seq_len is not None:
            assert(
                seq_len + generate_len <= self.model_config.max_seq_len
            ), f"seq_len {seq_len} exceeds the max_seq_len {self.model_config.max_seq_len}"
        
        if self.l % self.pp_size != 0:
            logger.warning(
                "Warning: the number of layers is not divisible by pp_size, please taking the floor!"
            )
        
        print("\n-------------------------- LLM main infer config --------------------------")
        infer_config_dict = {
            "inference_config":{
                "model_name": self.model_config.model_name,
                "batch_size_per_gpu": batch_size_per_gpu,
                "seq_len": seq_len,
                "tp_size": self.tp_size,
                "pp_size": self.pp_size,
                "generate_len": generate_len,
                "use_kv_cache": use_kv_cache,
            },
            "gpu_config": {
                "name": self.gpu_config.name,
                "memory_GPU_in_GB": f"{self.gpu_config.memory_GPU_in_GB} GB",
                "gpu_hbm_bandwidth": f"{get_gpu_hbm_bandwidth(self.gpu_config)} GB/s",
                "gpu_intra_node_bandwidth": f"{get_intra_node_bandwidth(self.gpu_config)} GB/s",
                "gpu_TFLOPS": f"{get_TFLOPS_per_gpu(self.gpu_config)} TFLOPS",
            }
        }
        pprint.pprint(infer_config_dict, indent=4, sort_dicts=False)
        
        print("\n---------------------------- LLM Params analysis ----------------------------")
        params_per_layer, dict_params_per_layer = self.llm_params.count_params_per_layer()
        num_params_model = self.llm_params.count_params_model()
        
        self.print_format_summary_dict(dict_params_per_layer, get_dict_depth(dict_params_per_layer))
        pprint.pprint({"params_model": num_to_string(num_params_model)}, indent=4, sort_dicts=False)
        
        print("\n---------------------------- LLM Flops analysis -----------------------------")
        flops_fwd_per_layer, dict_flops_fwd_per_layer = self.llm_flops.count_flops_fwd_per_layer(self.b, self.s)
        num_flops_fwd_model = self.llm_flops.count_flops_fwd_model(self.b, self.s)
        
        self.print_format_summary_dict(dict_flops_fwd_per_layer, get_dict_depth(dict_flops_fwd_per_layer))
        pprint.pprint({"flops_model": num_to_string(num_flops_fwd_model)}, indent=4, sort_dicts=False)
        
        print("\n---------------------------- LLM Memory analysis -----------------------------")
        memory_prefill_summary_dict, memory_decode_summary_dict = self.llm_memory.count_memory_per_gpu(
            batch_size_per_gpu,
            seq_len,
            generate_len,
            is_inference=True,
            use_kv_cache=use_kv_cache,
            activation_recomputation=activation_recomputation,
            layernorm_dtype_bytes=layernorm_dtype_bytes,
            kv_cache_dtype_bytes=kv_cache_dtype_bytes
        )
        self.print_format_summary_dict(memory_prefill_summary_dict, get_dict_depth(memory_prefill_summary_dict))
        self.print_format_summary_dict(memory_decode_summary_dict, get_dict_depth(memory_decode_summary_dict))
        
        print("\n-------------------------- LLM infer performance analysis --------------------------")
        prefill_latency_breakdown, decode_latency_breakdown = self.llm_latency.count_latency_fwd(
            batch_size_per_gpu,
            seq_len,
            generate_len,
            use_kv_cache=use_kv_cache,
            activation_recomputation=activation_recomputation,
            layernorm_dtype_bytes=layernorm_dtype_bytes,
            kv_cache_dtype_bytes=kv_cache_dtype_bytes
        )
        
        infer_result_dict = {
            "model_params": num_params_model,
            "model_flops": num_flops_fwd_model,
            "prefill_first_token_latency": prefill_latency_breakdown["prefill_latency"],
            "decode_per_token_latency": decode_latency_breakdown["decode_avg_latency"],
            "kv_cache_latency": decode_latency_breakdown["kv_cache_avg_latency"],
            "total_infer_latency": prefill_latency_breakdown["prefill_latency"] + decode_latency_breakdown["decode_avg_latency"] * generate_len,
        }
        
        self.print_format_summary_dict(infer_result_dict, get_dict_depth(infer_result_dict))
        
        print("\n-------------------------- LLM detailed's latency analysis --------------------------")
        
        # pprint.pprint([prefill_latency_breakdown, decode_latency_breakdown], indent=4, sort_dicts=False)
        
        # print("prefill_latency_breakdown depth is ", get_dict_depth(prefill_latency_breakdown), prefill_latency_breakdown)
        self.print_format_summary_dict(prefill_latency_breakdown, get_dict_depth(prefill_latency_breakdown))
        self.print_format_summary_dict(decode_latency_breakdown, get_dict_depth(decode_latency_breakdown))
            
    def print_format_summary_dict(self, summary_dict: dict, depth:int) -> str:
        for key, value in summary_dict.items():
            if "params" in key or "flops" in key:
                if not isinstance(value, dict):
                    summary_dict.update({key: num_to_string(value)})
                else:
                    self.print_format_summary_dict(value, get_dict_depth(value)-1) # 递归调用函数
            if "latency" in key:
                if not isinstance(value, dict):
                    summary_dict.update({key: latency_to_string(value)})
                else:
                    self.print_format_summary_dict(value, get_dict_depth(value)-1)
            if "memory" in key:
                if not isinstance(value, dict):
                    summary_dict.update({key: f"{num_to_string(value)}B"})
                else:
                    self.print_format_summary_dict(value, get_dict_depth(value)-1)
        if depth >= 1:
            pprint.pprint(summary_dict, indent=4, sort_dicts=False)
    
def llm_profile(model_name="llama-13b",
                gpu_name: str = "v100-sxm-32gb",
                bytes_per_param: int = BYTES_FP16,
                batch_size_per_gpu: int = 3,
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
    """Returns dict of the total floating-point operations, MACs, parameters and latency of a llm.

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
        dict: a summary dictionary of the inference analysis
    """
    model_config, gpu_config = get_model_and_gpu_config_by_name(model_name, gpu_name)
    
    parallelism_config = ParallelismConfig(tp_size=tp_size, pp_size=pp_size, 
                                        dp_size=dp_size, sp_size=sp_size
                                        )
    
    inference_config = InferenceConfig(batch_size_per_gpu=batch_size_per_gpu, seq_len=seq_len, 
                                       generate_len=generate_len, use_kv_cache=use_kv_cache,
                                       bytes_per_param=bytes_per_param,
                                       layernorm_dtype_bytes=layernorm_dtype_bytes,
                                       kv_cache_dtype_bytes=kv_cache_dtype_bytes
                                       )
    
    gpu_efficiency_config = GPUEfficiencyConfig(flops_efficiency=flops_efficiency,
                                                hbm_memory_efficiency=hbm_memory_efficiency,
                                                intra_node_memory_efficiency=intra_node_memory_efficiency,
                                                inter_node_memory_efficiency=inter_node_memory_efficiency
    )
    
    llm_configs = LLMConfigs(model_config=model_config, gpu_config=gpu_config,
                             parallelism_config=parallelism_config, inference_config=inference_config,
                             gpu_efficiency_config=gpu_efficiency_config
                            )

    profiler = LLMProfiler(llm_configs)
    
    profiler.infer_profile(batch_size_per_gpu=batch_size_per_gpu, seq_len=seq_len, 
                        generate_len=generate_len, use_kv_cache=use_kv_cache,
                        layernorm_dtype_bytes=layernorm_dtype_bytes,
                        flops_efficiency=flops_efficiency,
                        hbm_memory_efficiency=hbm_memory_efficiency)  
    
if __name__ == "__main__": 
    llm_profile()
    # fire.Fire(serialize=lambda x: json.dumps(x, indent=4))