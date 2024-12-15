import torch
import torch.nn as nn
from functools import partial
import sys

from models.VideoMAE_v2 import VideoMAEForClassification

model_ckpt = f'/workspace/pytorch_gpu/sign_language_code/model_ckpts/VideoMAE-2-131-labels-more-data/model_epoch_5.pth'
model = VideoMAEForClassification(
                          num_classes = 129,
                          qkv_bias=True, 
                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          init_values=0.
                          )
model.load_state_dict(torch.load(model_ckpt)['model_state_dict'], strict=False)
model.eval()

batch_size = 4
inp = torch.randn(batch_size, 3, 16, 224, 224)

onnx_program = torch.onnx.export(
                                model, 
                                inp, 
                                "/workspace/pytorch_gpu/sign_language_code/model_ckpts/onnx_models/videomae_v2_129_labels_more_data_5.onnx", 
                                input_names=['input'], 
                                output_names=['output'], 
                                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                            )
                                                            