import os
import shutil
from collections import defaultdict
import random
from typing import List, Dict, Tuple

def split_dataset(
    source_dir: str,
    output_dir: str,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> Dict[str, Dict[str, List[str]]]:
    """
    Split a hierarchical video dataset into train/validation/test sets while maintaining class balance.
    
    Args:
        source_dir: Path to the source directory containing class folders
        output_dir: Path to create the split dataset
        split_ratios: Tuple of (train, validation, test) ratios that sum to 1.0
    
    Returns:
        Dictionary containing the split information for each set
    """
    # Validate split ratios
    if sum(split_ratios) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Create output directories
    splits = ['train', 'validation', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Collect all class folders and their instance folders
    class_instances = defaultdict(list)
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            for instance in os.listdir(class_path):
                instance_path = os.path.join(class_path, instance)
                if os.path.isdir(instance_path):
                    class_instances[class_name].append(instance)
    

    # Calculate split sizes for each class
    split_counts = {}
    split_assignments = defaultdict(lambda: defaultdict(list))
    
    for class_name, instances in class_instances.items():
        n_instances = len(instances)
        split_counts[class_name] = {
            'train': int(n_instances * split_ratios[0]),
            'validation': int(n_instances * split_ratios[1]),
            'test': int(n_instances * split_ratios[2])
        }
        
        # Adjust for rounding errors
        total = sum(split_counts[class_name].values())
        if total < n_instances:
            split_counts[class_name]['train'] += n_instances - total
        
        # Randomly assign instances to splits
        shuffled_instances = instances.copy()
        random.shuffle(shuffled_instances)
        
        current_idx = 0
        for split, count in split_counts[class_name].items():
            split_instances = shuffled_instances[current_idx:current_idx + count]
            split_assignments[split][class_name].extend(split_instances)
            current_idx += count
    
    # Copy files to their new locations
    for split in splits:
        for class_name in class_instances.keys():
            # Create class directory in split
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            
            # Copy assigned instances
            for instance in split_assignments[split][class_name]:
                src_path = os.path.join(source_dir, class_name, instance)
                dst_path = os.path.join(split_class_dir, instance)
                shutil.copytree(src_path, dst_path)
    
    return split_assignments

def print_split_statistics(split_assignments: Dict[str, Dict[str, List[str]]]):
    """Print statistics about the dataset split."""
    print("\nDataset Split Statistics:")
    print("-" * 50)
    
    splits = ['train', 'validation', 'test']
    total_instances = defaultdict(int)
    
    for split in splits:
        print(f"\n{split.capitalize()} Set:")
        split_total = 0
        for class_name, instances in split_assignments[split].items():
            count = len(instances)
            split_total += count
            total_instances[class_name] += count
            print(f"  {class_name}: {count} instances")
        print(f"  Total: {split_total} instances")
    
    print("\nTotal instances per class:")
    for class_name, total in total_instances.items():
        print(f"  {class_name}: {total}")

if __name__ == "__main__":
    # Example usage
    source_directory = "/home/hinhnv/SignLanguage/data_cut_full_frames"
    output_directory = "/home/hinhnv/SignLanguage/data_83_labels_full_frames"

    # Split the dataset
    split_info = split_dataset(source_directory, output_directory)

    # Print statistics about the split
    print_split_statistics(split_info)