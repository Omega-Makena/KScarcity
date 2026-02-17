import kagglehub
import os
import shutil
from pathlib import Path

def download_datasets():
    # Define target directory
    project_root = Path(__file__).parent
    target_dir = project_root / "data" / "pulse"
    target_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        "tessytessy/kenyan-political-tweets",
        "edwardombui/hatespeech-kenya",
        "marthamwaura/bodaboda-menace-in-kenya-filtered-tweets",
        # "kariukiandrew/swahili-corpus"
    ]

    print(f"Target directory: {target_dir}")

    for ds in datasets:
        try:
            print(f"Downloading {ds}...")
            path = kagglehub.dataset_download(ds)
            print(f"✅ Downloaded to cache: {path}")
            
            # Copy files to our data/pulse directory
            dataset_name = ds.split("/")[-1]
            dest_folder = target_dir / dataset_name
            
            if dest_folder.exists():
                print(f"   Destination {dest_folder} exists, cleaning up...")
                shutil.rmtree(dest_folder)
            
            shutil.copytree(path, dest_folder)
            print(f"   📂 Copied to: {dest_folder}")
            
            # List files
            files = os.listdir(dest_folder)
            print(f"   Files: {files}")
            
        except Exception as e:
            print(f"❌ Failed to process {ds}: {e}")

if __name__ == "__main__":
    download_datasets()
