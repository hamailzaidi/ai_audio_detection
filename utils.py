import os
import random
import shutil

def copy_random_files(source_dir, dest_dir, num_files=500):
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # List all files in the source directory
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # Check if enough files are available
    if len(all_files) < num_files:
        raise ValueError(f"Not enough files in source directory. Found {len(all_files)}, need {num_files}")

    # Randomly select files
    selected_files = random.sample(all_files, num_files)

    # Copy selected files
    for file_name in selected_files:
        src_path = os.path.join(source_dir, file_name)
        dst_path = os.path.join(dest_dir, file_name)
        shutil.copy2(src_path, dst_path)

    print(f"Copied {num_files} files from '{source_dir}' to '{dest_dir}'.")



if __name__ == "__main__":
    copy_random_files(r'C:\Users\hp\Downloads\LJSpeech-1.1\LJSpeech-1.1\wavs',r'Datasets\LJspeech')