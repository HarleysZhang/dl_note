#########################################
#######     llm profiler    ############
#########################################

FLOPS_EFFICIENCY = 1.0  # FLOPS efficiency achieved by Megatron-LM is ~0.5 for LLM training
HBM_MEMORY_EFFICIENCY = 1  # GPU HBM memory efficiency
INTRA_NODE_MEMORY_EFFICIENCY = 1.0  # intra-node (nvlink) memory efficiency
INTER_NODE_MEMORY_EFFICIENCY = 1.0  # inter-node memory efficiency

NUM_GPUS_PER_NODE = 8  # number of GPUs per node

TOLERANCE = 0.01  # tolerance for floating point comparisons

BITS_PER_BYTE = 8  # number of bits in a byte

BITS_FP32 = 32  # number of bits in FP32 data type
BITS_FP16 = 16  # number of bits in FP16 data type
BITS_INT8 = 8  # number of bits in INT8 data type
BITS_INT4 = 4  # number of bits in INT4 data type

BYTES_FP32 = BITS_FP32 // BITS_PER_BYTE  # number of bytes in FP32 data type
BYTES_FP16 = BITS_FP16 // BITS_PER_BYTE  # number of bytes in FP16 data type
BYTES_INT8 = BITS_INT8 // BITS_PER_BYTE  # number of bytes in INT8 data type
BYTES_INT4 = BITS_INT4 // BITS_PER_BYTE  # number of bytes in INT4 data type

PRINT_LINE_WIDTH = 100

GPUS = [1, 2, 4, 8]