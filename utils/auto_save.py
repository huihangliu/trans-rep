import shutil
import os
from datetime import datetime

def backup_files(file_name, backup_path):
    # List of files and folders to copy
    files_to_copy = ['./utils', file_name]
    # List of excluded files and folders
    exclude = ['__pycache__']

    backup_path = os.path.join(backup_path, 'auto_save', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(backup_path, exist_ok=True)

    # Copy each file/folder to the new directory
    for item in files_to_copy:
      # Determine if this is a file or folder
      if os.path.isdir(item):
        if os.path.basename(item) in exclude:
          # print(f"Skipping {item}, as it is in the exclude list.")
          continue
        # It's a folder, so we'll copy the entire folder and its contents
        shutil.copytree(item, os.path.join(backup_path, os.path.basename(item)), dirs_exist_ok=True)
      elif os.path.isfile(item):
        # It's a file, so copy the file
        shutil.copy(item, backup_path)
      # else:
        # print(f"Skipping {item}, as it does not exist.")