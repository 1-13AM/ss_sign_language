import os
import shutil
from pathlib import Path

def merge_datasets(source_dirs, output_dir):
    """
    Merge multiple dataset directories while preserving their train/test/val structure
    and adding prefixes to label folders to avoid conflicts.
    
    Args:
        source_dirs (list): List of paths to source dataset directories
        output_dir (str): Path to output directory
        dataset_prefixes (list, optional): List of prefixes for each dataset. 
                                         If None, uses 'dataset_1', 'dataset_2', etc.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create splits directories in output
    splits = ['train', 'test', 'val']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Process each source directory
    for source_dir in source_dirs:
        print(f"\nProcessing dataset from {source_dir}")
        
        for split in splits:
            split_dir = os.path.join(source_dir, split)
            if not os.path.exists(split_dir):
                print(f"Warning: Split directory {split_dir} not found, skipping...")
                continue
                
            # Process each label directory in the split
            for label in os.listdir(split_dir):
                label_dir = os.path.join(split_dir, label)
                if not os.path.isdir(label_dir):
                    continue
                    
                # Create new label name with prefix
                new_label_name = f"{label}"
                new_label_dir = os.path.join(output_dir, split, new_label_name)
                
                print(f"Copying {label} -> {new_label_name}")
                
                # Copy the entire label directory to the new location
                shutil.copytree(label_dir, new_label_dir, dirs_exist_ok=True)
    
    # Print summary
    print("\nMerge complete! Summary:")
    for split in splits:
        split_path = os.path.join(output_dir, split)
        if os.path.exists(split_path):
            num_labels = len([d for d in os.listdir(split_path) 
                            if os.path.isdir(os.path.join(split_path, d))])
            print(f"{split}: {num_labels} label folders")

if __name__ == "__main__":
    # Example usage
    source_dirs = [
        "/workspace/sign-language-data/data_83_labels_full_frames",
        "/workspace/sign-language-data/data_50_labels_full_frames"
    ]
    
    output_dir = "/workspace/sign-language-data/data_frames"
    
    merge_datasets(source_dirs, output_dir)