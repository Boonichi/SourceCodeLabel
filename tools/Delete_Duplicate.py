from tkinter.filedialog import askdirectory
from tkinter import Tk
import os
import hashlib
from pathlib import Path
  
Tk().withdraw()
  
file_path = askdirectory(title="Select a folder")
  
list_of_files = os.walk(file_path)
  
unique_files = dict()
  
for root, folders, files in list_of_files:
  
    for file in files:
  
        file_path = Path(os.path.join(root, file))
  
        Hash_file = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
  
        if Hash_file not in unique_files:
            unique_files[Hash_file] = file_path
        else:
            os.remove(file_path)
            print(f"{file_path} has been deleted")