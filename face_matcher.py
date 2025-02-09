import torch
from torch import nn
from torchvision.io import read_image

class FaceMatcher:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def distance(self, templateA, templateB):
        diff = templateB - templateA
        diff_squared = diff * diff
        sum = torch.sum(diff_squared)
        return sum.item()

    def is_match(self, faceA, faceB):
        imageA = read_image(faceA).to(self.device)
        imageB = read_image(faceB).to(self.device)
        outA = self.model(imageA).reshape([128])
        outB = self.model(imageB).reshape([128])
        dist = self.distance(outA, outB)
        return dist < 0.6
