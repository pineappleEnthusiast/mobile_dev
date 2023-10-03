import torch
import torchvision
import models
import torch_directml

import onnx

dml = torch_directml.device()

net = models.ConvNet().to(dml)
net.load_state_dict(torch.load("RaspPiModel"))


x = torch.randn(64, 3, 32, 32, requires_grad=True).to(dml)
out = net(x)

torch.onnx.export(net,x, "RaspPiONNX.onnx", export_params=True,
                  opset_version=10, do_constant_folding=True,
                  input_names = ['input'], output_names = ['output'])

