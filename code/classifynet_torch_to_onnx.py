# -*- coding  : utf-8 -*-
# author      : honggao.zhang
# Create      : 2021-2-20
# Update      : 2021-3-12
# Version     : 0.1.0
# Description : 1, Classify net pytorch convert to onnx model template program.
#               2, Support alxnet、inceptionv3、resnet、shufflenetv2 and so on.

import sys, os,argparse
import os.path as osp
current_dir = osp.abspath(os.path.dirname(__file__))
import numpy as np

from torchsummary import summary
from thop import profile

import torch
from torch.autograd import Variable

# Gets the GPU if there is one, otherwise the cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu");print(device)
input_shape = [1, 3, 224, 224]
model_file_name = 'ResNet50_with_mask_sparsity_50.pth'  # 模型文件
base_name = osp.splitext(model_file_name)[0] 
current_dir = os.path.abspath(os.path.dirname(__file__))
weight_path = osp.join(current_dir, '../weights/', model_file_name); print(weight_path)

####################################################################################################################
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default = 'shufflenetv2_x1.0')
    parser.add_argument('--weight', '-w', default = None)
    parser.add_argument('--gpu', '-gpu', action = 'store_false')
    args = parser.parse_args()
    return args

###################################################################################################
def model_inference(net, input_data, weight_path):
    """Load model weight file and inference model."""
    # net = torch.load(model_path).to(device)
    if weight_path is not None:
        weight_path_name = osp.splitext(weight_path)[0]
        state_dict = torch.load(weight_path, map_location=lambda storage, loc: storage).to(device)
        # net.load_state_dict(state_dict, strict = False)
        print("Load model weight file %s success!" % weight_path_name)
    else:
        print("There is no model weights file!")
    net.eval()
    with torch.no_grad():     
        output = net(input_data)
        
        print("Model output type and shape is ", type(output), output.shape)  # torch.Size([1, 13, 48, 80])
        assert list(output.shape) == [1, 1000]
    return net

def torch_to_onnx(torch_model, input_data, onnx_file_path):
    torch.onnx.export(torch_model,
                    input_data,
                    onnx_file_path,
                    verbose=False,
                    opset_version=9,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],	    # 输入名
                    output_names=["output"],	# 输出名
                    )
####################################################################################################################
def define_net():
    pass

####################################################################################################################
def main(net_torch, net_name, input_shape):
    # 1, Define model structure
    net = net_torch
    # 2, Define model input
    input_data = Variable(torch.ones(input_shape)).to(device)
    # 3, Start model inference
    net = model_inference(net, input_data, None)
    # 4, Convert torch model to onnx model
    onnx_file_path = osp.abspath(osp.join(current_dir, '.../data/onnx_model/classifynet', net_name+".onnx"))  # onnx 模型权重文件路径定义
    torch_to_onnx(net, input_data, onnx_file_path)

    # 5, Analysis model FLOPs
    print(tuple(input_shape[1:]))
    summary(net, tuple(input_shape[1:]))
    print(input_data.shape)
    macs, params = profile(net, inputs=(input_data, ))
    print("Sum of model ops andparams is", (macs, params))
    print("Convert and analysis model success!")
 
###################################################################################################      
if __name__ == '__main__':  
    args = arg_parse()
    if args.model == 'alexnet':       # Alexnet example
        net_name = 'alexnet'
        from torchvision.models.alexnet import alexnet
        net_torch = alexnet(True).eval()
    elif args.model == 'resnet18':    # ResNet example
        net_name = 'resnet18'
        from torchvision.models.resnet import resnet18
        net_torch = resnet18(True).eval()
    elif args.model=='inception_v3':  # Inception_v3 example
        net_name = 'inception_v3'
        from torchvision.models.inception import inception_v3
        net_torch = inception_v3(True, transform_input=False).eval()
    elif args.model == 'vgg16':       # VGG19 example
        net_name = 'vgg16'
        from torchvision.models.vgg import vgg16
        net_torch = vgg16(True).eval()
    elif args.model == 'densenet121':
        net_name = 'densenet121'
        from torchvision.models.densenet import *
        net_torch = densenet121(True).eval()
    elif args.model == 'MobileNetV2':
        net_name = 'MobileNetV2'
        from torchvision.models.mobilenet import mobilenet_v2
        net_torch = mobilenet_v2(True).eval()
    elif args.model == 'shufflenetv2_x1.0':
        net_name = 'shufflenetv2_x1.0'
        from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0
        net_torch = shufflenet_v2_x1_0(False).eval()
    else:
        raise NotImplementedError()   
    
    if args.gpu:
        net_torch.to(device)
    
    main(net_torch, net_name, input_shape)



