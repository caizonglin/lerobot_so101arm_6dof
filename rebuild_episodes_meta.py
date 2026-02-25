# rebuild_episodes_meta.py
import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import json
import numpy as np

from lerobot.datasets.utils import write_info

print("Starting episode metadata reconstruction script...")

try:
    # --- Define Paths ---
    from lerobot.utils.constants import HF_LEROBOT_HOME
    repo_id = "zonglin11/bi_so100_test_dataset"
    repo_path = HF_LEROBOT_HOME / repo_id
    
    data_file = repo_path / "data/chunk-000/file-000.parquet"
    info_file = repo_path / "meta/info.json"
    episodes_dir = repo_path / "meta/episodes/chunk-000"
    episodes_file = episodes_dir / "file-000.parquet"

    # --- 1. Load the existing raw data ---
    print(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)
    print("Data loaded successfully.")

    # --- 2. Rebuild the episode metadata ---
    episode_meta_list = []
    total_frames = 0
    
    unique_episode_indices = sorted(df['episode_index'].unique())
    print(f"Found {len(unique_episode_indices)} unique episodes: {unique_episode_indices}")

    for idx in unique_episode_indices:
        episode_df = df[df['episode_index'] == idx]
        episode_length = len(episode_df)
        
        # Create a metadata entry for this episode
        # We make simplified assumptions: everything is in chunk 0, file 0.
        # We also create dummy/inferred data for stats and tasks.
        # This is enough to make the dataset loadable.
        meta_entry = {
            'episode_index': idx,
            'tasks': ['Controlling a bimanual SO100 robot.'],  # Assuming default task
            'length': episode_length,
            'stats/action.abs_max': np.array([1.0] * 14, dtype=np.float32), # Dummy value
            'data/chunk_index': 0,
            'data/file_index': 0,
            'videos/observation.images.front_camera/chunk_index': 0,
            'videos/observation.images.front_camera/file_index': 0,
            'videos/observation.images.front_camera/from_timestamp': 0.0, # Dummy
            'videos/observation.images.front_camera/to_timestamp': episode_length / 30.0, # Dummy
            'videos/observation.images.left_wrist_camera/chunk_index': 0,
            'videos/observation.images.left_wrist_camera/file_index': 0,
            'videos/observation.images.left_wrist_camera/from_timestamp': 0.0, # Dummy
            'videos/observation.images.left_wrist_camera/to_timestamp': episode_length / 30.0, # Dummy
            'videos/observation.images.right_wrist_camera/chunk_index': 0,
            'videos/observation.images.right_wrist_camera/file_index': 0,
            'videos/observation.images.right_wrist_camera/from_timestamp': 0.0, # Dummy
            'videos/observation.images.right_wrist_camera/to_timestamp': episode_length / 30.0, # Dummy
            'dataset_from_index': total_frames,
            'dataset_to_index': total_frames + episode_length,
        }
        episode_meta_list.append(meta_entry)
        
        total_frames += episode_length

    print(f"Reconstructed metadata for {len(episode_meta_list)} episodes.")

    # --- 3. Write the new episodes parquet file ---
    episodes_df = pd.DataFrame(episode_meta_list)
    
    print(f"Writing episode metadata to {episodes_file}...")
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episodes_df.to_parquet(episodes_file)
    print("Episode metadata file written successfully.")

    # --- 4. Update info.json ---
    print(f"Updating {info_file}...")
    with open(info_file, 'r') as f:
        info_data = json.load(f)
    
    info_data['total_episodes'] = len(unique_episode_indices)
    info_data['total_frames'] = total_frames
    info_data['splits'] = {'train': f'0:{len(unique_episode_indices)}'}
    
    write_info(info_data, repo_path)
    print("'info.json' updated successfully.")
    
    print("\nRecovery script finished! Your dataset should now be loadable.")

except Exception as e:
    print(f"\nAn error occurred during recovery: {e}")
    import traceback
    traceback.print_exc()
