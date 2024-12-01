import os
import json
import requests
import hashlib
import zipfile
import shutil
import subprocess
from tqdm import tqdm
from pathlib import Path
import logging
import sys
from time import sleep
import csv
import argparse

# Configure logging
logging.basicConfig(
    filename='combined_download.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def download_file(url, dest_path, sha1sum=None):
    """
    Download a file from a URL to a destination path with optional SHA-1 checksum verification.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(dest_path, 'wb') as f, tqdm(
                    desc=f"Downloading {os.path.basename(dest_path)}",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            size = f.write(chunk)
                            bar.update(size)
            if sha1sum:
                if verify_sha1(dest_path, sha1sum):
                    logging.info(f"Downloaded and verified: {dest_path}")
                    return True
                else:
                    logging.warning(f"SHA-1 mismatch for {dest_path}. Attempt {attempt} of {MAX_RETRIES}.")
                    os.remove(dest_path)
            else:
                logging.info(f"Downloaded: {dest_path}")
                return True
        except Exception as e:
            logging.error(f"Error downloading {url}: {e}. Attempt {attempt} of {MAX_RETRIES}.")
            sleep(RETRY_DELAY)
    logging.error(f"Failed to download {url} after {MAX_RETRIES} attempts.")
    return False

def verify_sha1(file_path, expected_sha1):
    """
    Verify the SHA-1 checksum of a file.
    """
    sha1 = hashlib.sha1()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha1.update(chunk)
        file_sha1 = sha1.hexdigest()
        return file_sha1 == expected_sha1
    except Exception as e:
        logging.error(f"Error verifying SHA-1 for {file_path}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """
    Extract a ZIP file to a specified directory.
    """
    try:
        os.makedirs(extract_to, exist_ok=True)  # Ensure the extraction directory exists
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info(f"Extracted {zip_path} to {extract_to}")
        return True
    except zipfile.BadZipFile as e:
        logging.error(f"Bad ZIP file {zip_path}: {e}")
        return False
    except Exception as e:
        logging.error(f"Error extracting {zip_path}: {e}")
        return False

def extract_frames(video_path, frames_output_dir, fps=1, sequence_id=""):
    """
    Extract frames from a video at a specified frames per second (fps) using ffmpeg.

    Args:
        video_path (str): Path to the video file.
        frames_output_dir (str): Directory to save extracted frames.
        fps (int): Frames per second to extract.
        sequence_id (str): Identifier for the sequence to name frames appropriately.

    Returns:
        bool: True if frames were successfully extracted, False otherwise.
    """
    try:
        Path(frames_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Naming scheme: "sequenceID_frame####.jpg"
        output_pattern = os.path.join(frames_output_dir, f"{sequence_id}_frame%04d.jpg")
        
        # Construct ffmpeg command to extract frames at specified FPS
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps={fps}',
            '-q:v', '2',  # Set quality for JPEG
            output_pattern
        ]
        
        # Execute ffmpeg command
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
        
        # Verify that frames were extracted
        extracted_frames = list(Path(frames_output_dir).glob(f"{sequence_id}_frame*.jpg"))
        if not extracted_frames:
            raise ValueError(f"No frames were extracted from {video_path}.")
        
        logging.info(f"Extracted {len(extracted_frames)} frames from {video_path} to {frames_output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg error while extracting frames from {video_path}: {e}")
        return False
    except Exception as e:
        logging.error(f"Error extracting frames from {video_path}: {e}")
        return False

def parse_activity_label(sequence_id: str) -> str:
    """
    Extracts the activity label from the sequence ID.
    Example: "ADT_Apartment_release_clean_seq131_M1292" -> "Clean"
    Handles cases where 'release' is followed by 'skeleton' or 'multiskeleton'.
    """
    try:
        parts = sequence_id.split('_')
        release_idx = parts.index('release')
        activity_label = parts[release_idx + 1]
        if activity_label.lower() in ["skeleton", "multiskeleton", "multiuser"]:
            activity_label = parts[release_idx + 2]
        return activity_label.capitalize()  # e.g., "clean" -> "Clean"
    except (ValueError, IndexError):
        logging.warning(f"Could not parse activity label from sequence ID: {sequence_id}")
        return "Unknown"

def generate_composite_label(sequence_id, annotations_dir):
    """
    Generate a composite label based on activity and key objects.
    Example: "Clean with ChoppingBoard and Sponge"
    """
    try:
        instances_path = os.path.join(annotations_dir, 'instances.json')
        with open(instances_path, 'r') as f:
            instances = json.load(f)
        
        # Extract dynamic objects
        key_objects = [info['instance_name'] for info in instances.values() if info.get('motion_type') == 'dynamic']
        activity_label = parse_activity_label(sequence_id)
        if key_objects:
            composite_label = f"{activity_label} with " + ", ".join(key_objects)
        else:
            composite_label = activity_label
        return composite_label
    except Exception as e:
        logging.error(f"Error generating composite label for {sequence_id}: {e}")
        return "Unknown"

def process_sequence(sequence_id, sequence_data, dataset_dir, metadata_list, fps=1):
    """
    Process a single sequence: download video and ground truth, verify, extract frames and annotations,
    and organize them into the dataset.
    """
    activity_label = parse_activity_label(sequence_id).lower()  # e.g., "Clean" -> "clean"
    # Removed activity_dir to flatten the images directory
    frames_dir = os.path.join(dataset_dir, 'images')  # Directly use 'images' directory
    os.makedirs(frames_dir, exist_ok=True)

    # Define paths
    video_info = sequence_data.get('video_main_rgb', {})
    video_filename = video_info.get('filename')
    video_url = video_info.get('download_url')
    video_sha1 = video_info.get('sha1sum')

    groundtruth_info = sequence_data.get('main_groundtruth', {})
    groundtruth_filename = groundtruth_info.get('filename')
    groundtruth_url = groundtruth_info.get('download_url')
    groundtruth_sha1 = groundtruth_info.get('sha1sum')

    if not video_filename or not video_url:
        logging.warning(f"No video information for sequence {sequence_id}")
        return False

    # Define paths
    video_path = os.path.join(dataset_dir, 'videos', sequence_id, video_filename)
    groundtruth_path = os.path.join(dataset_dir, 'groundtruth', sequence_id, groundtruth_filename) if groundtruth_filename else None
    annotations_dir = os.path.join(dataset_dir, 'groundtruth', sequence_id, 'annotations') if groundtruth_filename else None

    # Ensure directories exist
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    if groundtruth_filename:
        os.makedirs(os.path.dirname(groundtruth_path), exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)  # Ensure annotations_dir exists

    # Download video
    if not os.path.exists(video_path):
        success = download_file(video_url, video_path, sha1sum=video_sha1)
        if not success:
            logging.error(f"Failed to download video for sequence {sequence_id}")
            return False
    else:
        logging.info(f"Video already exists: {video_path}")

    # Extract frames
    frame_prefix = sequence_id
    expected_frame_pattern = f"{frame_prefix}_frame*.jpg"
    if not any(Path(frames_dir).glob(expected_frame_pattern)):
        success = extract_frames(video_path, frames_dir, fps=fps, sequence_id=sequence_id)
        if not success:
            logging.error(f"Failed to extract frames for sequence {sequence_id}")
            return False
    else:
        logging.info(f"Frames already extracted for sequence {sequence_id}")

    # Download and extract ground truth if available
    if groundtruth_filename and groundtruth_url:
        if not os.path.exists(groundtruth_path):
            success = download_file(groundtruth_url, groundtruth_path, sha1sum=groundtruth_sha1)
            if not success:
                logging.error(f"Failed to download ground truth for sequence {sequence_id}")
                return False
        else:
            logging.info(f"Ground truth already exists: {groundtruth_path}")

        if annotations_dir and not os.listdir(annotations_dir):
            success = extract_zip(groundtruth_path, annotations_dir)
            if not success:
                logging.error(f"Failed to extract ground truth for sequence {sequence_id}")
                return False
        else:
            logging.info(f"Ground truth already extracted for sequence {sequence_id}")
    else:
        logging.warning(f"No ground truth information for sequence {sequence_id}")

    # Generate composite label based on annotations
    composite_label = generate_composite_label(sequence_id, annotations_dir) if annotations_dir else "Unknown"
    logging.info(f"Composite label for sequence {sequence_id}: {composite_label}")

    # Add to metadata list
    metadata_list.append({
        'sequence_id': sequence_id,
        'activity_label': composite_label,  # e.g., "Clean with ChoppingBoard, Sponge"
        'frames_path': 'images'  # Directly reference 'images' directory
    })

    return True

def convert_metadata_to_llava_json(metadata_csv_path, output_json_path, dataset_dir):
    """
    Convert metadata CSV to LLaVA's required JSON format with relative file paths.

    Args:
        metadata_csv_path (str): Path to the metadata CSV file.
        output_json_path (str): Path to save the output JSON file.
        dataset_dir (str): Parent directory of 'aria_dataset/'.
    """
    samples = []

    # Get the absolute path of the dataset directory
    dataset_dir_abs = os.path.abspath(dataset_dir)

    with open(metadata_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Converting metadata to JSON"):
            sequence_id = row['sequence_id']
            activity_label = row['activity_label']
            frames_path = row['frames_path']

            # Normalize frames_path to use forward slashes
            frames_path = frames_path.replace("\\", "/")

            # Construct the full path to the frames directory
            full_frames_path = os.path.join(dataset_dir_abs, frames_path).replace("\\", "/")

            # Check if the frames directory exists
            if not os.path.isdir(full_frames_path):
                print(f"Frames directory '{full_frames_path}' does not exist. Skipping sequence '{sequence_id}'.")
                logging.warning(f"Frames directory '{full_frames_path}' does not exist. Skipping sequence '{sequence_id}'.")
                continue

            # List all frame files in full_frames_path
            frame_files = sorted([
                f for f in os.listdir(full_frames_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])

            if not frame_files:
                print(f"No frame images found in '{full_frames_path}'. Skipping sequence '{sequence_id}'.")
                logging.warning(f"No frame images found in '{full_frames_path}'. Skipping sequence '{sequence_id}'.")
                continue

            for frame_file in frame_files:
                # Generate a unique frame ID
                frame_id = f"{sequence_id}_{os.path.splitext(frame_file)[0]}"  # e.g., "seq123_frame_0001"

                # Construct the relative path to the image **without** prepending 'aria_dataset/'
                image_relative_path = os.path.join(frames_path, frame_file).replace("\\", "/")

                # Verify if the image file exists
                image_full_path = os.path.join(full_frames_path, frame_file).replace("\\", "/")
                if not os.path.exists(image_full_path):
                    print(f"Image file '{image_full_path}' does not exist. Skipping frame '{frame_id}'.")
                    logging.warning(f"Image file '{image_full_path}' does not exist. Skipping frame '{frame_id}'.")
                    continue

                sample = {
                    "id": frame_id,
                    "image": image_relative_path,  # Use relative path as is
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nWhat main activity is taking place?"
                        },
                        {
                            "from": "gpt",
                            "value": activity_label
                        }
                    ]
                }

                samples.append(sample)

    # Save to JSON
    try:
        with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(samples, jsonfile, indent=2)
        logging.info(f"Converted {len(samples)} samples to '{output_json_path}'")
        print(f"Converted {len(samples)} samples to '{output_json_path}'")
    except IOError as e:
        logging.error(f"Error writing to file '{output_json_path}': {e}")
        print(f"Error: Failed to save JSON to '{output_json_path}'. Check logs for details.")

def main(json_path, dataset_dir, metadata_output_path, json_output_path, fps=1, max_download=None):
    """
    Main function to download and organize the ARIA dataset, then convert metadata to JSON.
    """
    metadata_list = []
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        sequences = data['sequences']
    except Exception as e:
        logging.error(f"Error loading JSON file {json_path}: {e}")
        sys.exit(1)

    sequence_items = list(sequences.items())

    total_sequences = len(sequence_items)
    if max_download is not None:
        if max_download <= 0:
            logging.error("The --max_download value must be a positive integer.")
            print("Error: The --max_download value must be a positive integer.")
            sys.exit(1)
        if max_download > total_sequences:
            logging.warning(f"Requested max_download ({max_download}) exceeds total sequences ({total_sequences}). Setting max_download to {total_sequences}.")
            print(f"Warning: Requested max_download ({max_download}) exceeds total sequences ({total_sequences}). Setting max_download to {total_sequences}.")
            max_download = total_sequences
        sequence_items = sequence_items[:max_download]
    else:
        max_download = total_sequences

    logging.info(f"Starting processing of {max_download} sequences out of {total_sequences} available.")

    for sequence_id, sequence_data in tqdm(sequence_items, desc="Processing sequences", unit="sequence"):
        success = process_sequence(sequence_id, sequence_data, dataset_dir, metadata_list, fps=fps)
        if success:
            logging.info(f"Successfully processed sequence {sequence_id}")
        else:
            logging.error(f"Failed to process sequence {sequence_id}")

    # Save metadata to CSV
    try:
        os.makedirs(os.path.dirname(metadata_output_path), exist_ok=True)
    except FileNotFoundError:
        # If metadata_output_path is in the current directory
        pass

    try:
        with open(metadata_output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['sequence_id', 'activity_label', 'frames_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in metadata_list:
                writer.writerow(entry)
        logging.info(f"Metadata saved to {metadata_output_path}")
        print(f"Metadata saved to {metadata_output_path}")
        print(f"Total sequences downloaded and processed: {len(metadata_list)}")
    except Exception as e:
        logging.error(f"Error saving metadata to CSV: {e}")
        print(f"Error: Failed to save metadata to {metadata_output_path}. Check logs for details.")
        sys.exit(1)

    # Convert metadata CSV to JSON
    convert_metadata_to_llava_json(metadata_output_path, json_output_path, dataset_dir)

def run(json_path, dataset_dir, metadata_output_path, json_output_path, fps, max_download):
    main(json_path, dataset_dir, metadata_output_path, json_output_path, fps, max_download)

def run_combined_script():
    parser = argparse.ArgumentParser(description="Download and organize ARIA dataset, then convert metadata to LLaVA JSON format.")
    parser.add_argument('--json_path', type=str, required=True, help="Path to ADT_download_urls.json")
    parser.add_argument('--dataset_dir', type=str, default='aria_dataset', help="Directory to store the dataset")
    parser.add_argument('--metadata_output_path', type=str, default='aria_dataset/metadata.csv', help="Path to save the metadata CSV")
    parser.add_argument('--json_output_path', type=str, default='aria_dataset/data_llava.json', help="Path to save the output JSON file for LLaVA")
    parser.add_argument('--fps', type=int, default=1, help="Frames per second to extract from videos")
    parser.add_argument('--max_download', type=int, default=None, help="Maximum number of sequences to download and process")

    args = parser.parse_args()

    main(
        json_path=args.json_path,
        dataset_dir=args.dataset_dir,
        metadata_output_path=args.metadata_output_path,
        json_output_path=args.json_output_path,
        fps=args.fps,
        max_download=args.max_download
    )

if __name__ == "__main__":
    run_combined_script()
