from resnet_deserializer import Deserializer
import make_layers
from make_layers import builderMap
from resnet_layers import InputRGBImageLayer, Tag, Skip, AddPrev
import torch
from torch import nn
import re

class FaceRecResNet(nn.Module):
    def __init__(self, model_file, layer_tag_file):
        super().__init__()
        with open(layer_tag_file, "r") as f:
            layer_tags = f.read().splitlines()

        file = open(model_file, "r+b")
        deserializer = Deserializer(file)

        loss_layer_version = deserializer.read_int()
        length = deserializer.read_int()
        layer_name = deserializer.readString(length)
        layer = builderMap[layer_name](deserializer)

        layer_version = deserializer.read_int()
        version_stack = [(layer_version, layer_tags.pop())]
        while 1 <= layer_version and layer_version <= 3:
            version_stack.append((layer_version, layer_tags.pop()))
            layer_version = deserializer.read_int()

        layer_name_len = layer_version

        self.tag_layer_cache = {}
        layers = []
        curr_layer = None
        layer_tag = ""
        while len(version_stack) > 0:
            prev_layer = curr_layer
            layer_name = deserializer.readString(layer_name_len)
            m_tag_skip = re.fullmatch('(tag|skip)(\\d)', layer_tag)
            while m_tag_skip:
                layer_id = int(m_tag_skip.group(2))
                if m_tag_skip.group(1) == 'tag':
                    layers.append(Tag(self.tag_layer_cache, layer_id))
                elif m_tag_skip.group(1) == 'skip':
                    layers.append(Skip(self.tag_layer_cache, layer_id))
                (version, layer_tag) = version_stack.pop()
                m_tag_skip = re.fullmatch('(tag|skip)(\\d)', layer_tag)
            m_add_prev = re.fullmatch('add_prev(\\d)', layer_tag)
            if m_add_prev:
                assert layer_name == 'add_prev_', "Unexpected add_prev_ layer"
                curr_layer = AddPrev(deserializer, self.tag_layer_cache, int(m_add_prev.group(1)))
            else:
                curr_layer = builderMap[layer_name](deserializer)
            layers.append(curr_layer)
            (version, layer_tag) = version_stack.pop()
            curr_layer.version = version
            if isinstance(prev_layer, InputRGBImageLayer) and prev_layer.version == 3:
                deserializer.skip(2)
            if len(version_stack) > 0:
                layer_name_len = deserializer.read_int()

        deserializer.close()
        file.close()

        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.stack.forward(x)
