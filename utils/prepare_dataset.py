import os
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler
import random
import shutil
import glob
import sys

from .util import temporal_subsample

class CustomVideoDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.instances = []
        self.labels = []
        self.str_labels = []
        self.label_to_idx = {}
        
        # Load all labels and instances
        for label_idx, label_folder in enumerate(sorted(os.listdir(root_dir))):
            label_folder_path = os.path.join(root_dir, label_folder)
            if os.path.isdir(label_folder_path):
                self.label_to_idx[label_folder] = label_idx
                for instance in os.listdir(label_folder_path):
                    instance_path = os.path.join(label_folder_path, instance)
                    if os.path.isdir(instance_path):
                        self.instances.append(instance_path)
                        self.str_labels.append(label_folder)
                        self.labels.append(self.label_to_idx[label_folder])
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance_path = self.instances[idx]
        label = self.labels[idx]
        str_label = self.str_labels[idx]
        images = []
        
        # Load all jpg images for the instance
        for img_name in os.listdir(instance_path):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(instance_path, img_name)
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
        
        # Stack images into a tensor (e.g., [N, C, H, W] where N is number of images)
        images = torch.stack(images)
        
        # [N, C, H, W]
        return {"pixel_values": images, "label": label, "str_label": str_label, "video_path": instance_path}

class MergedDataset(Dataset):
    def __init__(self, datasets):
        """
        Args:
            datasets (list): A list of PyTorch datasets to merge.
        """
        self.datasets = datasets
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        self.cumulative_sizes = torch.cumsum(torch.tensor(self.dataset_sizes), dim=0)

        self.prepare_mapping()
        
    def prepare_mapping(self):
        # Create a global mapping of unique labels
        self.global_label_to_idx = {}
        current_idx = 0

        for dataset in self.datasets:
            for label in dataset.str_labels:
                if label not in self.global_label_to_idx:
                    self.global_label_to_idx[label] = current_idx
                    current_idx += 1

        # Map each dataset's labels to the global label index
        self.labels = []
        self.dataset_label_offsets = []

        for dataset in self.datasets:
            label_offset = []
            for label in dataset.str_labels:
                label_offset.append(self.global_label_to_idx[label])
            self.labels.extend(label_offset)
            self.dataset_label_offsets.append(label_offset)
            
    def __len__(self):
        return sum(self.dataset_sizes)
    
    def __getitem__(self, idx):
        # Determine which dataset and sample the index corresponds to
        dataset_idx = (self.cumulative_sizes > idx).nonzero(as_tuple=False)[0].item()
        if dataset_idx > 0:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        else:
            sample_idx = idx

        # Get the sample and adjust the label using the global label mapping
        sample = self.datasets[dataset_idx][sample_idx]
        images, str_label, video_path = sample['pixel_values'], sample['str_label'], sample['video_path']
        adjusted_label = self.global_label_to_idx[str_label]

        return {"pixel_values": images, "label": adjusted_label, "str_label": str_label, "video_path": video_path}
    
class FlexibleVideoDataset(Dataset):
    
    def __init__(self, root_dir, transform=None, num_output_frames=16, frame_sampling_strategy="random", first_idx: int = 0, last_idx_from_end: int = -1):
        self.root_dir = root_dir
        self.transform = transform
        self.instances = []
        self.labels = []
        self.str_labels = []
        self.label_to_idx = {}
        self.num_output_frames = num_output_frames
        self.first_idx = first_idx
        self.last_idx_from_end = last_idx_from_end
        self.frame_sampling_strategy = frame_sampling_strategy
        
        # Load all labels and instances
        for label_idx, label_folder in enumerate(sorted(os.listdir(root_dir))):
            label_folder_path = os.path.join(root_dir, label_folder)
            if os.path.isdir(label_folder_path):
                self.label_to_idx[label_folder] = label_idx
                for instance in os.listdir(label_folder_path):
                    instance_path = os.path.join(label_folder_path, instance)
                    if os.path.isdir(instance_path):
                        self.instances.append(instance_path)
                        self.str_labels.append(label_folder)
                        self.labels.append(self.label_to_idx[label_folder])

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance_path = self.instances[idx]
        label = self.labels[idx]
        str_label = self.str_labels[idx]
        images = []
        
        num_frames = len(glob.glob(instance_path + "/*.jpg"))
        
        last_idx = num_frames - 1 if self.last_idx_from_end == -1 else num_frames - self.last_idx_from_end - 1 
        # Load all jpg images for the instance
        frame_idx = temporal_subsample(first_idx=self.first_idx,
                                       last_idx=last_idx,
                                       strategy=self.frame_sampling_strategy,
                                       num_output_frames=self.num_output_frames)
        
        all_image_paths = os.listdir(instance_path)
        sampled_image_paths = [all_image_paths[i] for i in frame_idx]
        
        for img_name in sampled_image_paths:
            if img_name.endswith('.jpg'):
                img_path = os.path.join(instance_path, img_name)
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
        
        # Stack images into a tensor (e.g., [N, C, H, W] where N is number of images)
        images = torch.stack(images)
        
        # [N, C, H, W]
        return {"pixel_values": images, 
                "label": label,
                "str_label": str_label, 
                "video_path": instance_path, 
                "sampled_frame_idx": frame_idx}

class MergedFlexibleDataset(Dataset):
    def __init__(self, datasets, balanced=False):
        """
        Args:
            datasets (list): A list of PyTorch datasets to merge.
        """
        self.datasets = datasets
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        self.cumulative_sizes = torch.cumsum(torch.tensor(self.dataset_sizes), dim=0)
        self.weights, self.sampler = None, None
        
        if balanced:
            self.weights = []
            for dataset_size in self.dataset_sizes:
                self.weights.extend([self.cumulative_sizes[-1].item() / dataset_size] * dataset_size)
            self.sampler = WeightedRandomSampler(self.weights, num_samples=self.cumulative_sizes[-1].item(), replacement=True)

        self.prepare_mapping()
        
    # def prepare_mapping(self):
    #     # Create a global mapping of unique labels
    #     self.global_label_to_idx = {}
    #     current_idx = 0

    #     # Build the global label mapping
    #     for dataset in self.datasets:
    #         for label in dataset.labels:
    #             if label not in self.global_label_to_idx:
    #                 self.global_label_to_idx[label] = current_idx
    #                 current_idx += 1

    #     # Store labels using the global mapping
    #     self.labels = []
    #     for dataset in self.datasets:
    #         for label in dataset.labels:
    #             self.labels.append(self.global_label_to_idx[label])
    
    def prepare_mapping(self):
        # Create a global mapping of unique labels
        self.global_label_to_idx = {}
        current_idx = 0

        for dataset in self.datasets:
            for label in dataset.str_labels:
                if label not in self.global_label_to_idx:
                    self.global_label_to_idx[label] = current_idx
                    current_idx += 1

        # Map each dataset's labels to the global label index
        self.labels = []
        self.dataset_label_offsets = []

        for dataset in self.datasets:
            label_offset = []
            for label in dataset.str_labels:
                label_offset.append(self.global_label_to_idx[label])
            self.labels.extend(label_offset)
            self.dataset_label_offsets.append(label_offset)
    
    def __len__(self):
        return sum(self.dataset_sizes)
    
    def __getitem__(self, idx):
        # Determine which dataset and sample the index corresponds to
        dataset_idx = (self.cumulative_sizes > idx).nonzero(as_tuple=False)[0].item()
        if dataset_idx > 0:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        else:
            sample_idx = idx

        # Get the sample and adjust its label using the global label mapping
        dtps = self.datasets[dataset_idx][sample_idx]
        images = dtps['pixel_values']
        str_label = dtps['str_label']
        video_path = dtps['video_path']
        frame_idx = dtps['sampled_frame_idx']

        # Adjust the label using the global mapping
        adjusted_label = self.global_label_to_idx[str_label]

        return {
            "pixel_values": images,
            "label": adjusted_label,
            "str_label": str_label,
            "video_path": video_path,
            "sampled_frame_idx": frame_idx
        }

def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):
    # Ensure the output directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Iterate through each label folder
    for label in os.listdir(source_dir):
        label_path = os.path.join(source_dir, label)

        # Skip if not a directory
        if not os.path.isdir(label_path):
            continue
        
        # List all instance folders inside the label folder
        instances = [f for f in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, f))]
        random.shuffle(instances)

        # Split instances into train and validation sets
        split_index = int(len(instances) * split_ratio)
        train_instances = instances[:split_index]
        val_instances = instances[split_index:]

        # Copy instances to train directory
        for instance in train_instances:
            src_path = os.path.join(label_path, instance)
            dest_path = os.path.join(train_dir, label, instance)
            shutil.copytree(src_path, dest_path)

        # Copy instances to validation directory
        for instance in val_instances:
            src_path = os.path.join(label_path, instance)
            dest_path = os.path.join(val_dir, label, instance)
            shutil.copytree(src_path, dest_path)

        print(f"Copied {len(train_instances)} instances to train, {len(val_instances)} instances to validation for label: {label}")