1，LLM 推理时间组成；

![infer time](../images/llm_infer/infer_time.png)

2，FlashLlamaQKVBias 继承自 FlashCausalLM。在 FlashLlamaQKVBiasForCausalLM 类初始化之前，先得加载模型权重和配置文件。

3，internlm2 模型报错，定位是在于模型权重解析出来的权重参数字典，和目前的模型结构（FlashLlamaQKVBiasForCausalLM）对不上。