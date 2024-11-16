import os
import json
import pandas as pd
import jsonlines
import logging

logging.basicConfig(
    filename='prepare_dataset.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def load_annotations(annotation_dir):
    """
    Load all annotation files for a given sequence.
    """
    annotations = {}
    try:
        # Load instances.json
        instances_path = os.path.join(annotation_dir, 'instances.json')
        if os.path.exists(instances_path):
            with open(instances_path, 'r') as f:
                annotations['instances'] = json.load(f)
            print(f"Loaded instances.json with {len(annotations['instances'])} entries.")
        else:
            print(f"Missing instances.json in {annotation_dir}")
            annotations['instances'] = {}
        
        # Load 2d_bounding_box.csv
        bbox_2d_path = os.path.join(annotation_dir, '2d_bounding_box.csv')
        if os.path.exists(bbox_2d_path):
            annotations['bbox_2d'] = pd.read_csv(bbox_2d_path)
            print(f"Loaded 2d_bounding_box.csv with {len(annotations['bbox_2d'])} entries.")
        else:
            print(f"Missing 2d_bounding_box.csv in {annotation_dir}")
            annotations['bbox_2d'] = pd.DataFrame()
        
        # Load ground truth action labels
        action_labels_path = os.path.join(annotation_dir, 'action_labels.json')
        if os.path.exists(action_labels_path):
            with open(action_labels_path, 'r') as f:
                annotations['action_labels'] = json.load(f)
            print(f"Loaded action_labels.json with {len(annotations['action_labels'])} entries.")
        else:
            print(f"Missing action_labels.json in {annotation_dir}")
            annotations['action_labels'] = {}
        
    except Exception as e:
        print(f"Error loading annotations from {annotation_dir}: {e}")
        logging.error(f"Error loading annotations from {annotation_dir}: {e}")
    
    return annotations

def list_dynamic_objects_with_annotations(annotation_dir):
    """
    List all dynamic objects and check their presence in 2d_bounding_box.csv
    
    Args:
        annotation_dir (str): Path to the annotation directory
    
    Returns:
        dict: Dictionary of dynamic object_uids and their names present in annotations.
    """
    instances_path = os.path.join(annotation_dir, 'instances.json')
    bbox_2d_path = os.path.join(annotation_dir, '2d_bounding_box.csv')
    
    if not os.path.exists(instances_path):
        print(f"instances.json not found in {annotation_dir}")
        return {}
    if not os.path.exists(bbox_2d_path):
        print(f"2d_bounding_box.csv not found in {annotation_dir}")
        return {}
    
    # Load instances.json
    with open(instances_path, 'r') as f:
        instances = json.load(f)
    # Filter dynamic objects
    dynamic_objects = {uid: info['instance_name'] for uid, info in instances.items() if info.get('motion_type', '').lower() == 'dynamic'}
    
    if not dynamic_objects:
        print("No dynamic objects found in instances.json.")
        return {}
    
    # Load 2d_bounding_box.csv
    bbox_2d = pd.read_csv(bbox_2d_path)
    # Filter bounding boxes for dynamic objects
    dynamic_bbox = bbox_2d[bbox_2d['object_uid'].astype(str).isin(dynamic_objects.keys())]
    
    # List dynamic objects present in annotations
    dynamic_present = dynamic_bbox['object_uid'].astype(str).unique()
    dynamic_present_info = {uid: dynamic_objects[uid] for uid in dynamic_present}
    
    print(f"\nDynamic Objects Present in Annotations ({len(dynamic_present_info)}):")
    for uid, name in dynamic_present_info.items():
        print(f" - UID: {uid}, Name: {name}")
    
    return dynamic_present_info

def generate_description(action_label, dynamic_objects):
    """
    Generate a descriptive annotation string based on action label and dynamic objects.
    
    Args:
        action_label (str): The high-level action label.
        dynamic_objects (dict): Dictionary of dynamic object_uids and their names.
        
    Returns:
        str: Combined description string.
    """
    if dynamic_objects:
        dynamic_desc = ", ".join([f"{name} is moving" for name in dynamic_objects.values()])
    else:
        dynamic_desc = "."
    description = f"Dynamic Objects: {dynamic_desc}\nDescribe the actions in this frame.\nASSISTANT: {action_label}."
    return description

def create_jsonl_entry(sequence_id, frames_dir, annotations, output_jsonl_path, annotation_dir, fps=30):
    """
    Create a single JSONL entry for a given sequence based on frame timestamps and action labels.
    
    Args:
        sequence_id (str): Identifier for the sequence.
        frames_dir (str): Directory containing the frame images.
        annotations (dict): Loaded annotations for the sequence.
        output_jsonl_path (str): Path to save the JSONL file.
        annotation_dir (str): Path to the annotation directory.
        fps (int): Frames per second.
        
    Returns:
        None
    """
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    # One of the llava models mentioned doing it across 8 frames (or more) for longer videos 
    if len(frames) != 8:
        print(f"Skipping {sequence_id}: Expected 8 frames, found {len(frames)}.")
        logging.warning(f"Skipping {sequence_id}: Expected 8 frames, found {len(frames)}.")
        return
    
    dynamic_objects = list_dynamic_objects_with_annotations(annotation_dir)
    
    action_labels = annotations.get('action_labels', {})
    action_label = action_labels.get(sequence_id, "No significant actions detected.")
    
    # Generate description
    description = generate_description(action_label, dynamic_objects)
    
    prompt = f"USER: <video>\n{description}"
    
    # Get absolute paths for frames
    frame_paths = [os.path.abspath(os.path.join(frames_dir, frame)) for frame in frames]
    
    entry = {
        "video": frame_paths,
        "text": prompt
    }
    
    try:
        with jsonlines.open(output_jsonl_path, mode='w') as writer:
            writer.write(entry)
        print(f"Written 1 entry to {output_jsonl_path}")
        logging.info(f"Written 1 entry to {output_jsonl_path}")
    except Exception as e:
        print(f"Error writing JSONL for {sequence_id}: {e}")
        logging.error(f"Error writing JSONL for {sequence_id}: {e}")

def prepare_dataset(aria_dataset_dir, output_dir, fps=30):
    """
    Prepare the JSONL dataset for Video-LLaVa fine-tuning.
    
    Args:
        aria_dataset_dir (str): Path to the ARIA dataset directory.
        output_dir (str): Path to the output directory for JSONL files.
        fps (int): Frames per second.
    """
    annotations_base = os.path.join(aria_dataset_dir, 'annotations')
    frames_base = os.path.join(aria_dataset_dir, 'frames')
    jsonl_output_dir = os.path.join(output_dir, 'jsonl')
    os.makedirs(jsonl_output_dir, exist_ok=True)
    
    for sequence_id in os.listdir(annotations_base):
        annotation_dir = os.path.join(annotations_base, sequence_id)
        frames_dir = os.path.join(frames_base, sequence_id)
        
        if not os.path.isdir(annotation_dir) or not os.path.isdir(frames_dir):
            print(f"Skipping {sequence_id}: Missing annotations or frames directory.")
            logging.warning(f"Skipping {sequence_id}: Missing annotations or frames directory.")
            continue
        
        frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        print(f"\nProcessing sequence '{sequence_id}' with {len(frames)} frames.")
        
        if len(frames) != 8:
            print(f"Skipping {sequence_id}: Expected 8 frames, found {len(frames)}.")
            logging.warning(f"Skipping {sequence_id}: Expected 8 frames, found {len(frames)}.")
            continue
        
        annotations = load_annotations(annotation_dir)
        dynamic_objects = list_dynamic_objects_with_annotations(annotation_dir)
        
        if not dynamic_objects:
            print(f"No dynamic objects found in sequence '{sequence_id}'. Skipping.")
            logging.info(f"No dynamic objects found in sequence '{sequence_id}'. Skipping.")
            continue
        
        output_jsonl_path = os.path.join(jsonl_output_dir, f"{sequence_id}.jsonl")
        create_jsonl_entry(sequence_id, frames_dir, annotations, output_jsonl_path, annotation_dir, fps)

        print(f"Created JSONL for '{sequence_id}' with 1 entry.")
        logging.info(f"Created JSONL for '{sequence_id}' with 1 entry.")

if __name__ == "__main__":
    aria_dataset_dir = 'aria_dataset/'
    output_dir = 'prepared_dataset/'
    prepare_dataset(aria_dataset_dir, output_dir)
