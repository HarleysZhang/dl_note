import os

gpus = [1, 2, 4, 8]

def get_model_config(model="llama13b"):
    model_config = dict(
        llama13b = dict(h=5120, n_layers=40, vocab_size = 32000),
        llama33b = dict(h=6656, n_layers=60, vocab_size = 32000),
        llama65b = dict(h=8192, n_layers=80, vocab_size = 32000),
        )
    
    h = model_config[model]["h"]
    n = model_config[model]["n_layers"]
    V = model_config[model]["vocab_size"]
    return h, n, V

def get_CausalLM_params(model="llama13b"):
    h, n, V = get_model_config(model)
    
    total_params = (12 * h * h + 4 * h) * n + V * h
    simple_params = 12 * n * h * h
    
    formatted_num = '{:,.0f}'.format(simple_params)
    print(f"simple params: ", formatted_num)
    
    return simple_params, total_params

def get_CausalLM_flops(model="llama13b", batch_size = 1, seq_len = 1):
    b, s = batch_size, seq_len
    h, n, V = get_model_config(model)
    
    total_flops = (24 * b * s * h * h + 4 * b * s * s * h) * n + 2 * b * s * h * V
    simple_flops = 24 * b * s * h * h * n
    formatted_num = '{:,.0f}'.format(simple_flops)
    print(f"simple flops: ", formatted_num)
    
    return simple_flops, total_flops

def get_gpu_memory_consume(model="llama13b", batch_size = 1, seq_len = 1024, output_ids_len = 1024, ignore_memory_intermediate=True):
    b, s, o = batch_size, seq_len, output_ids_len
    h, n, V = get_model_config(model)
    
    simple_params, total_params = get_CausalLM_params(model)
    memory_model = total_params * 2
    memory_intermediate = 8 * b * s * h
    memory_kv_cache = 4 *b * n * h * (s+o)
    
    if ignore_memory_intermediate:
        memory_consume = memory_model + memory_kv_cache
    
    memory_consume_format = round(memory_consume / (1024**3), 2)  # 单位换算成 GB
    formatted_num = '{:,.0f}'.format(memory_consume_format)
    print(f"simple gpu memory consume: {formatted_num} GB")
    
    return memory_consume
    
def get_theory_latency(device="v100", model="llama13b", batch_size=1):
    if device == "v100":
        interconnect_bandwidth = 32 # gb/s
        gpu_memory_bandwidth = 900 # gb/s
        performance = 112 * pow(10, 3) # tflops
        batch_size_threshold = performance / gpu_memory_bandwidth # 124.4
    elif device == "t4":
        interconnect_bandwidth = 32 # gb/s
        gpu_memory_bandwidth = 300 # gb/s
        performance = 65 * pow(10, 3) # tflops
        batch_size_threshold = performance / gpu_memory_bandwidth 
        
    if model == "llama13b":
        model_size = 13 # gb
        n_layers = 40
        d_model = 5120
    elif model == "llama65b":
        model_size = 65 # gb
        n_layers = 80
        d_model = 8192
    
    for N in gpus:
        if batch_size < batch_size_threshold:
            compute_theory_latency = 2 * model_size / (N * gpu_memory_bandwidth)
            comms_theory_latency = 2 * 4 * n_layers * 8 # us
            comms_theory_latency_ms = comms_theory_latency / 1000
        else:
            compute_theory_latency = batch_size * 2 * model_size / (N * performance)
            comms_theory_latency = batch_size * 2 * 4 * n_layers * d_model / (interconnect_bandwidth * pow(10, 9))
            comms_theory_latency_ms = comms_theory_latency * 1000
        
        compute_theory_latency_ms = compute_theory_latency * 1000
        
        # print(f'{compute_theory_latency_ms: .2f} ms, {comms_theory_latency_ms: .2f} ms', end=". ")
        print("%s, gpu_num: %d, Latency of token: %.2f ms" % (model, N, compute_theory_latency_ms + comms_theory_latency_ms))
              
if __name__ == "__main__": 
    get_theory_latency()
    _ = get_CausalLM_params("llama13b")
    _ = get_CausalLM_flops("llama13b")
    _ = get_gpu_memory_consume("llama13b")