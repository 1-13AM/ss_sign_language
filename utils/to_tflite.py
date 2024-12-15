import torch
import torchvision
import ai_edge_torch
from functools import partial
import torch.nn as nn

import sys

from models.VideoMAE_v2 import VideoMAEForClassification

import os
os.environ['PJRT_DEVICE'] = 'CPU'

model_ckpt = f'/workspace/pytorch-gpu/sign-language-codes/VideoMAE-2-ss-data-finetuned/model_epoch_33.pth'
model = VideoMAEForClassification(
                          num_classes = 83,
                          qkv_bias=True, 
                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          init_values=0.
                          )
model.load_state_dict(torch.load(model_ckpt)['model_state_dict'], strict=False)
model.eval()

sample_inputs = (torch.randn(1, 3, 16, 224, 224),)

edge_model = ai_edge_torch.convert(model, sample_inputs)
edge_model.export("videomae_v2.tflite")
