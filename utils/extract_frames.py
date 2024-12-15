import cv2
import os
import concurrent.futures
import tensorflow as tf
import argparse

def center_crop(img, dim):
    """Returns center cropped image.
    
    Args:
    img (numpy.ndarray): Image to be center cropped.
    dim (tuple): Dimensions (width, height) to be cropped.
    
    Returns:
    numpy.ndarray: The cropped image.
    """
    width, height = img.shape[1], img.shape[0]
    cropped_size = min(width, height)
    mid_x, mid_y = width // 2, height // 2
    cw2, ch2 = cropped_size // 2, cropped_size // 2
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img

def format_frames(frame, output_size):
    frame = center_crop(frame, (1600,1600))
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame.numpy()

def extract_frames_from_video(video_path, output_folder, num_frames=16):
    video_name = os.path.basename(video_path).split('.')[0]
    
    # Check if output folder exists and contains frames
    if os.path.exists(output_folder):
        existing_frames = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]
        if len(existing_frames) > 0:
            print(f"Skipping {video_path} - frames already extracted")
            return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    # Create output folder for the current video
    os.makedirs(output_folder, exist_ok=True)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = 1
    
    frame_count = 0
    extracted_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % step == 0:
            frame_file = os.path.join(output_folder, f"{video_name}_frame{extracted_count + 1}.jpg")
            frame = format_frames(frame, (224,224))
            cv2.imwrite(frame_file, frame)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Successfully extracted {extracted_count} frames from {video_path}")

def process_single_video(video_path, input_folder, output_base_folder, num_frames):
    relative_path = os.path.relpath(video_path, input_folder)
    video_output_folder = os.path.join(output_base_folder, os.path.splitext(relative_path)[0])
    extract_frames_from_video(video_path, video_output_folder, num_frames)

def process_videos_in_structure(input_folder, output_base_folder, num_frames=16):
    video_paths = []
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".mp4") or file.endswith(".avi"):
                video_paths.append(os.path.join(root, file))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_video, video_path, input_folder, output_base_folder, num_frames)
                   for video_path in video_paths]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocessing steps that extracts all frames from a video & center crops'
    )
    
    parser.add_argument(
        '--input_folder',
        type=str,
        required=True,
        help='Path to the video data directory'
    )
    
    parser.add_argument(
        '--output_base_folder',
        type=str,
        required=True,
        help='Path to the frame data directory'
    )
    
    args = parser.parse_args()
    
    process_videos_in_structure(args.input_folder, args.output_base_folder)