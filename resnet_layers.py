import torch
from torch import nn

class LossLayer(object):
    def __init__(self, deserializer):
        super().__init__()
        self.margin = deserializer.read_float()
        self.dist_thresh = deserializer.read_float()

class InputRGBImageLayer(torch.nn.Module):
    def __init__(self, deserializer):
        super().__init__()
        self.avg_red = deserializer.read_float()
        self.avg_green = deserializer.read_float()
        self.avg_blue = deserializer.read_float()
        self.nr = deserializer.read_uint()
        self.nc = deserializer.read_uint()

    def forward(self, x):
        return torch.stack([(x[0] - self.avg_red) / 256.0, (x[1] - self.avg_green) / 256.0, (x[2] - self.avg_blue) / 256.0])

class Tag(torch.nn.Module):
    def __init__(self, tag_layer_cache, layer_id):
        super().__init__()
        self.tag_layer_cache = tag_layer_cache
        self.layer_id = layer_id

    def forward(self, x):
        self.tag_layer_cache[self.layer_id] = x
        return x

class Skip(torch.nn.Module):
    def __init__(self, tag_layer_cache, layer_id):
        super().__init__()
        self.tag_layer_cache = tag_layer_cache
        self.layer_id = layer_id

    def forward(self, x):
        ret = self.tag_layer_cache[self.layer_id]
        self.tag_layer_cache.pop(self.layer_id)
        return ret

class AddPrev(torch.nn.Module):
    def __init__(self, deserializer, tag_layer_cache, layer_id):
        super().__init__()
        self.tag_layer_cache = tag_layer_cache
        self.layer_id = layer_id
        deserializer.skip(33)

    def pad(self, t, padded_size):
        padding = []
        for i in range(len(padded_size)-1, -1, -1):
            padding.append(0)
            padding.append(padded_size[i] - t.size()[i])
        padding_tuple = tuple(padding)
        return torch.nn.functional.pad(t, padding_tuple)

    def forward(self, x):
        tagged_layer_output = self.tag_layer_cache[self.layer_id]
        if x.size() == tagged_layer_output.size():
            ret = x + tagged_layer_output
        else:
            out_size = torch.Size(list(map(lambda x,y: max(x,y), list(x.size()), list(tagged_layer_output.size()))))
            padded_x = x if x.size() == out_size else self.pad(x, out_size)
            padded_tagged_layer_output = tagged_layer_output if tagged_layer_output.size() == out_size else self.pad(tagged_layer_output, out_size)
            ret = padded_x + padded_tagged_layer_output
        self.tag_layer_cache.pop(self.layer_id)
        return ret

class AvgPool(nn.AvgPool2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.match_input_size = self.kernel_size == (0,0)

    def forward(self, x):
        if self.match_input_size:
            self.kernel_size = (x.size()[-2], x.size()[-1])
        return super().forward(x)

AFFINE_CONV_MODE = 0

class Affine(nn.Module):
    def __init__(self, gamma, beta):
        super().__init__()
        # shaped (C, 1, 1) to broadcast per-channel over both (C, H, W) and (N, C, H, W) inputs
        self.register_buffer("gamma", gamma.reshape(-1, 1, 1))
        self.register_buffer("beta", beta.reshape(-1, 1, 1))

    def forward(self, x):
        return x * self.gamma + self.beta

FC_HAS_BIAS = 0

class Fc(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        reshaped_input = x.reshape(int(x.numel()/self.in_features), self.in_features)
        return super().forward(reshaped_input)
