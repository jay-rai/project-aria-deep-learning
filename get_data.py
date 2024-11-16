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

# Configure logging
logging.basicConfig(
    filename='download_aria_dataset.log',
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
                with open(dest_path, 'wb') as f, tqdm(
                    desc=f"Downloading {os.path.basename(dest_path)}",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1024):
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

def extract_frames(video_path, frames_output_dir, num_frames=8):
    """
    Extract uniformly sampled frames from a video using ffmpeg.
    """
    try:
        Path(frames_output_dir).mkdir(parents=True, exist_ok=True)
        # Get video duration in seconds
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        duration = float(result.stdout)
        # Calculate timestamps for sampling
        timestamps = [duration * (i + 0.5) / num_frames for i in range(num_frames)]
        for idx, ts in enumerate(timestamps):
            frame_path = os.path.join(frames_output_dir, f"frame_{idx+1:04d}.jpg")
            subprocess.run([
                'ffmpeg', '-ss', str(ts), '-i', video_path,
                '-frames:v', '1', '-q:v', '2', frame_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info(f"Extracted {num_frames} frames from {video_path} to {frames_output_dir}")
        return True
    except Exception as e:
        logging.error(f"Error extracting frames from {video_path}: {e}")
        return False

def process_sequence(sequence_id, sequence_data, dataset_dir):
    """
    Process a single sequence: download video and ground truth, verify, extract frames.
    """
    video_filename = sequence_data['video_main_rgb']['filename']
    video_url = sequence_data['video_main_rgb']['download_url']
    video_sha1 = sequence_data['video_main_rgb'].get('sha1sum')
    
    groundtruth_filename = sequence_data['main_groundtruth']['filename']
    groundtruth_url = sequence_data['main_groundtruth']['download_url']
    groundtruth_sha1 = sequence_data['main_groundtruth'].get('sha1sum')
    
    # Paths
    video_dir = os.path.join(dataset_dir, 'video_main_rgb')
    annotations_dir = os.path.join(dataset_dir, 'annotations', sequence_id)
    frames_dir = os.path.join(dataset_dir, 'frames', sequence_id)
    
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    Path(annotations_dir).mkdir(parents=True, exist_ok=True)
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    
    video_path = os.path.join(video_dir, video_filename)
    groundtruth_path = os.path.join(dataset_dir, 'groundtruth_zips', sequence_id, groundtruth_filename)
    
    Path(os.path.dirname(groundtruth_path)).mkdir(parents=True, exist_ok=True)
    
    # Download video
    if not os.path.exists(video_path):
        success = download_file(video_url, video_path, sha1sum=video_sha1)
        if not success:
            logging.error(f"Failed to download video for sequence {sequence_id}")
            return False
    else:
        logging.info(f"Video already exists: {video_path}")
    
    # Download ground truth
    if not os.path.exists(groundtruth_path):
        success = download_file(groundtruth_url, groundtruth_path, sha1sum=groundtruth_sha1)
        if not success:
            logging.error(f"Failed to download ground truth for sequence {sequence_id}")
            return False
    else:
        logging.info(f"Ground truth already exists: {groundtruth_path}")
    
    # Extract ground truth
    if not os.listdir(annotations_dir):
        success = extract_zip(groundtruth_path, annotations_dir)
        if not success:
            logging.error(f"Failed to extract ground truth for sequence {sequence_id}")
            return False
    else:
        logging.info(f"Ground truth already extracted for sequence {sequence_id}")
    
    # Extract frames
    if not os.listdir(frames_dir):
        success = extract_frames(video_path, frames_dir, num_frames=8)
        if not success:
            logging.error(f"Failed to extract frames for sequence {sequence_id}")
            return False
    else:
        logging.info(f"Frames already extracted for sequence {sequence_id}")
    
    return True

def main(json_path, dataset_dir, max_download=10):
    """
    Main function to download and organize the ARIA dataset.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        sequences = data['sequences']
    except Exception as e:
        logging.error(f"Error loading JSON file {json_path}: {e}")
        sys.exit(1)
    
    groundtruth_download_dir = os.path.join(dataset_dir, 'groundtruth_zips')
    Path(groundtruth_download_dir).mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    for sequence_id, sequence_data in tqdm(sequences.items(), desc="Processing sequences"):
        if downloaded >= max_download:
            logging.info(f"Reached max_download limit: {max_download}")
            break
        success = process_sequence(sequence_id, sequence_data, dataset_dir)
        if success:
            downloaded += 1
            logging.info(f"Successfully processed sequence {sequence_id}")
        else:
            logging.error(f"Failed to process sequence {sequence_id}")
    
    logging.info(f"Completed processing. Total sequences downloaded: {downloaded}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and organize ARIA dataset for Video-LLaVa fine-tuning.")
    parser.add_argument('--json_path', type=str, required=True, help="Path to ADT_downloads_url.json")
    parser.add_argument('--dataset_dir', type=str, default='aria_dataset', help="Directory to store the dataset")
    parser.add_argument('--max_download', type=int, default=10, help="Maximum number of sequences to download")
    
    args = parser.parse_args()
    
    main(args.json_path, args.dataset_dir, args.max_download)
