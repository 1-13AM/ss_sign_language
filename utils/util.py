from typing import Optional
import torchvision
import pathlib
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import OrderedDict
import torch.nn as nn
import random

def temporal_subsample(first_idx: int, last_idx: int, strategy: str = 'uniform', num_output_frames: int = 16) -> np.ndarray:
    
    assert last_idx - first_idx + 1 >= num_output_frames, "num_frames must at least be equal to num_output_frames"
    if strategy == 'uniform':
        return np.linspace(first_idx, last_idx, num_output_frames, endpoint=True).astype(int)
    
    elif strategy == 'random':
        sampled_frame_idx = chunk_and_sample(first_idx, last_idx, num_output_frames)
        return np.array(sampled_frame_idx)
    
def chunk_and_sample(a, b, n):
    """
    Divides the sequence of numbers from 0 to n into k chunks, samples one value from each chunk,
    and returns a list of sampled values.
    
    Args:
        n (int): The end value of the sequence (exclusive), generating sequence from 0 to n-1.
        k (int): The number of chunks to divide the sequence into.
    
    Returns:
        list: A list of sampled values, one from each chunk.
    """
    boundaries = np.linspace(a, b, n + 1, endpoint=True).astype(int)
    
    sampled_values = []
    
    for i in range(0, len(boundaries) - 1):
        chunk = np.arange(boundaries[i], boundaries[i+1])
        if len(chunk) > 0:
            sampled_value = np.random.choice(chunk)
            sampled_values.append(sampled_value)
    
    return sampled_values


class RescaleTransform(torch.nn.Module):
    def forward(self, img):
        img = img * 2 - 1
        return img


def create_video_transforms(mean: tuple[float], std: tuple[float], resize_to: int, crop_size:  Optional[tuple[int,...]]) -> torch.nn.Sequential:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std),
        ShortSideScale(resize_to),
        transforms.CenterCrop(crop_size)
    ])

def create_video_transforms_ss(mean: tuple[float], std: tuple[float], resize_to: int, crop_size:  Optional[tuple[int,...]]) -> torch.nn.Sequential:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std),
        # ShortSideScale(resize_to),
        # transforms.CenterCrop(crop_size)
    ])

def create_video_transforms_i3d(resize_to: int, crop_size:  Optional[tuple[int,...]]) -> torch.nn.Sequential:
    return transforms.Compose([
        transforms.ToTensor(),
        RescaleTransform(),
        ShortSideScale(resize_to),
        transforms.CenterCrop(crop_size)
    ])
    
def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example['pixel_values'].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example['label'] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def summary(model, input_size, save_file: str = None, batch_size=-1, device="cuda"):

    import torch
    import torch.nn as nn
    
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []


    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
            
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    
    # saved files
    for name, layer in summary.items():
        if isinstance(layer['nb_params'], torch.Tensor):
            layer['nb_params'] = layer['nb_params'].item()
    
    if save_file is not None:
        import json
        with open(save_file, "w") as f:
            json.dump(summary, f)

class ShortSideScale(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        h, w = img.shape[-2:]
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
        else:
            oh = self.size
            ow = int(self.size * w / h)
        # torch.nn.functional.interpolate takes batched images as input
        if len(img.shape) == 3:
            img = img[None,...]
        return torch.nn.functional.interpolate(img, size=(oh, ow), mode="bilinear", align_corners=False)[0,:]            

def kd_loss_func(student_features, teacher_features):
    
    import torch.nn.functional as F
    
    l2_norm_loss = 0.
    dot_prod_loss = 0.
    
    def lp_norm_loss(x, y, p=2):
        return (((x - y) ** p).sum()) ** (1/p)
    
    for student_feature, teacher_feature in zip(student_features.values(), teacher_features.values()):
        student_feature = F.normalize(student_feature, dim=-1)
        teacher_feature = F.normalize(teacher_feature, dim=-1)
        
        l2_norm_loss += lp_norm_loss(student_feature, teacher_feature, p=2) / (student_feature.shape[0] * student_feature.shape[1])
        dot_prod_loss += abs(1 - (student_feature * teacher_feature).sum(-1).mean())
    
    return l2_norm_loss, dot_prod_loss
    
def set_all_seeds(seed=42, deterministic=True):
    """
    Set all seeds to make results reproducible.
    
    Args:
        seed (int): Seed number, defaults to 42
        deterministic (bool): If True, ensures deterministic behavior in CUDA operations
                            Note that this may impact performance
    
    Note:
        Setting deterministic=True may significantly impact performance, but ensures
        complete reproducibility. If speed is crucial, you might want to set it to False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False