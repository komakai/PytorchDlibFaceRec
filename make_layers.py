from resnet_layers import *
import torch
from torch import nn
import numpy as np

def make_tensor_shaped(raw, dims, transposed = False):
    raw_array = np.copy(np.frombuffer(raw, dtype="float32").reshape(dims))
    adjusted_array = raw_array if not transposed else raw_array.transpose()
    return torch.from_numpy(adjusted_array)

def make_tensor(raw):
    return torch.from_numpy(np.copy(np.frombuffer(raw, dtype="float32")))

def make_loss_layer(deserializer):
    return LossLayer(deserializer)

def make_input_rgb_image_layer(deserializer):
    return InputRGBImageLayer(deserializer)

def make_conv(deserializer):
    deserializer.read_int()  # version
    (n, k, nr, nc) = deserializer.read_int4()
    size = n * k * nr * nc
    mark = deserializer.skip(size * 4)
    conv_num_filters = deserializer.read_int()
    conv_nr = deserializer.read_int()
    conv_nc = deserializer.read_int()
    conv_stride_y = deserializer.read_int()
    conv_stride_x = deserializer.read_int()
    conv_padding_y = deserializer.read_int()
    conv_padding_x = deserializer.read_int()
    deserializer.read_int()  # version
    tensor_dims = deserializer.read_int4()
    (out_channels, in_channels, nr, nc) = tensor_dims
    size = out_channels * in_channels * nr * nc * 4
    weights = make_tensor_shaped(deserializer.read_tensor(mark, size), tensor_dims)
    mark += size
    deserializer.read_int()  # version
    (n, k, nr, nc) = deserializer.read_int4()
    size = n * k * nr * nc * 4
    biases = make_tensor(deserializer.read_tensor(mark, size))
    learning_rate_multiplier, weight_decay_multiplier, bias_learning_rate_multiplier, bias_weight_decay_multiplier = deserializer.read_float4()
    deserializer.skip(33)
    ret = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(conv_nr, conv_nc),
                          stride=(conv_stride_x, conv_stride_y), padding=(conv_padding_x, conv_padding_y), bias=True)
    with torch.no_grad():
        ret.weight.copy_(weights)
        ret.bias.copy_(biases)
    return ret

def make_affine(deserializer):
    deserializer.read_int()  # version
    (n, k, nr, nc) = deserializer.read_int4()
    size = n * k * nr * nc
    mark = deserializer.skip(size * 4)
    deserializer.read_int()  # version
    (n, affine_channels, nr, nc) = deserializer.read_int4()
    size = n * affine_channels * nr * nc * 4
    gamma = make_tensor_shaped(deserializer.read_tensor(mark, size), (affine_channels, 1, 1, 1))
    mark += size
    deserializer.read_int()  # version
    (n, k, nr, nc) = deserializer.read_int4()
    size = n * k * nr * nc * 4
    beta = make_tensor(deserializer.read_tensor(mark, size))
    mode = deserializer.read_int()
    deserializer.skip(33)
    ret = nn.Conv2d(in_channels=affine_channels, out_channels=affine_channels, kernel_size=1, stride=1,
                            groups=affine_channels, bias=True)
    with torch.no_grad():
        ret.weight.copy_(gamma)
        ret.bias.copy_(beta)

    return ret

def make_relu(deserializer):
    deserializer.skip(33)
    return nn.ReLU()

def make_maxpool(deserializer):
    nr, nc, stride_y, stride_x, padding_y_, padding_x_ = deserializer.read_int6()
    deserializer.skip(33)
    return nn.MaxPool2d(kernel_size=(nr, nc), stride=(stride_x, stride_y), padding=(padding_x_, padding_y_))

def make_avgpool(deserializer):
    nr, nc, stride_y, stride_x, padding_y_, padding_x_ = deserializer.read_int6()
    deserializer.skip(33)
    return AvgPool(kernel_size=(nr, nc), stride=(stride_x, stride_y), padding=(padding_x_, padding_y_))

def make_fc(deserializer):
    num_outputs = deserializer.read_uint()
    num_inputs = deserializer.read_uint()
    deserializer.read_int()  # version
    (n, k, nr, nc) = deserializer.read_int4()
    size = n * k * nr * nc
    mark = deserializer.skip(size * 4)
    deserializer.read_int()  # version
    (n, k, nr, nc) = deserializer.read_int4()
    size = n * k * nr * nc * 4
    weights = make_tensor_shaped(deserializer.read_tensor(mark, size), (num_inputs, num_outputs), True)

    mark += size
    deserializer.read_int()  # version
    (n, k, nr, nc) = deserializer.read_int4()
    size = n * k * nr * nc * 4
    biases = make_tensor(deserializer.read_tensor(mark, size))
    bmode = deserializer.read_int()
    learning_rate_multiplier, weight_decay_multiplier, bias_learning_rate_multiplier, bias_weight_decay_multiplier = deserializer.read_float4()
    deserializer.skip(33)
    ret = Fc(in_features=num_inputs, out_features=num_outputs, bias=bmode == FC_HAS_BIAS)
    with torch.no_grad():
        ret.weight.copy_(weights)
        if bmode == FC_HAS_BIAS:
            ret.bias.copy_(biases)
    return ret

builderMap = {
    "loss_metric_2" : make_loss_layer,
    "input_rgb_image_sized": make_input_rgb_image_layer,
    "con_4": make_conv,
    "affine_": make_affine,
    "relu_": make_relu,
    "max_pool_2": make_maxpool,
    "avg_pool_2": make_avgpool,
    "fc_2": make_fc,
}
