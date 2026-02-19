import os
import shutil

SOURCE = r"C:\Users\omegam\OneDrive - Innova Limited\Documents\scarce\backend"
DEST = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\backend"

def safe_copy(src, dst):
    for root, dirs, files in os.walk(src):
        # Construct destination path
        rel_path = os.path.relpath(root, src)
        dst_path = os.path.join(dst, rel_path)
        
        # Create destination folder if missing
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        # Copy only files that do NOT exist
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_path, file)

            # Skip duplicates (do not overwrite)
            if os.path.exists(dst_file):
                print(f"SKIP: {dst_file} already exists")
                continue

            # Copy file
            shutil.copy2(src_file, dst_file)
            print(f"COPIED: {src_file} -> {dst_file}")

if __name__ == "__main__":
    safe_copy(SOURCE, DEST)
    print("\n✅ Safe copy complete! No overwrites, no duplicates.")
