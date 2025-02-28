import os
import shutil
from pathlib import Path
from itertools import islice
import re
from collections import defaultdict


class Functor:
    def __init__(self, value):
        self.value = value

    def fmap(self, func, *args, **kwargs):
        return Functor(func(self.value, *args, **kwargs))

    def __repr__(self):
        return f"Functor({repr(self.value)})"


class ListFunctor:
    def __init__(self, values):
        self.values = values

    def fmap(self, func, *args, **kwargs):
        return ListFunctor([func(x, *args, **kwargs) for x in self.values])
    
    def apply(self, func, *args, **kwargs):
        return ListFunctor(func(self.values, *args, **kwargs))

    def __repr__(self):
        return f"ListFunctor({repr(self.values)})"


def extract_index(filename, pattern=r'(\d+)'):
    """Extracts the first numeric index from a filename."""
    match = re.search(pattern, filename)
    return int(match.group(1)) if match else float('inf')


def get_files_path(dirs, exts={'.jpg', '.png', 'jpeg'} , n=100):

    def get_files(dir_path):
        """Get up to `n` filtered files from a directory, sorted by index."""
        return sorted([
            str(f) for f in Path(dir_path).iterdir()
            if f.is_file() and f.suffix.lower() in exts
        ], key=lambda f: extract_index(Path(f).name))[:n]

    return ListFunctor(dirs)\
        .fmap(get_files)\
        .apply(lambda x: list(zip(*x)))\
        .values


def merge_folders(source_folders, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        
    for folder in source_folders:
        for file_name in os.listdir(folder):
            abs_path = os.path.join(folder, file_name)
            if os.path.isfile(abs_path):
                shutil.copy(abs_path, destination_folder)


def omit_filename(filepath):
    res = '/'.join(filepath.split('\\')[:-1])
    print(res)
    return res

def extract_subfolder(filepath, root_dir):
    # relative_path = os.path.relpath(filename, root_dir)
    # subfolder = os.path.dirname(relative_path).replace('\\', '/')
    # print(relative_path)
    # return subfolder if subfolder else ''
    # path = os.path.abspath(filename).replace('\\', '/')
    res = '/'.join(filepath.split('/')[:-1])  # omit filename
    # print(res)
    return res


def remove_folder_files(dir):    
    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        
        if os.path.isfile(file_path):
            os.remove(file_path)
            # print(f"Removed: {file_path}")
        else:
            print(f"Skipped folder: {file_path}")


def create_folders(target_dir, subfolders):
    
    for subfolder in subfolders:
        folder_path = os.path.join(target_dir, subfolder)
        
        if not os.path.exists(folder_path):
            print(f"Creating folder: {folder_path}")
            os.makedirs(folder_path, exist_ok=True)
        else:
            remove_folder_files(folder_path)
            print(f"Folder already exists: {folder_path}")


def copy_files_from_to(file_paths, root_dir='/data', target_dir='/data/temp_dir'):
    from itertools import chain

    # file_paths in the form [(1 2 3 ... n), (... n), ...]
    files_flatten = list(chain.from_iterable(file_paths))
    
    # used to create folders to save inside target dir
    subfolders_functor = ListFunctor(files_flatten)\
        .fmap(extract_subfolder, root_dir=root_dir)\
        .apply(lambda x: list(set(x)))\
        .fmap(os.path.abspath)\
        .fmap(lambda x: x.replace('\\', '/'))
    
    target_subfolders = subfolders_functor\
        .fmap(lambda path: path.replace(root_dir, target_dir))\
        .values
    # create_folders(target_dir, target_subfolders)
    
    to_paths_abs_functor = ListFunctor(files_flatten)\
        .fmap(lambda x: x.replace('\\', '/'))\
        .fmap(lambda x: x.replace(root_dir[1:], target_dir[1:]))\
        .fmap(os.path.abspath)
    
    from_path_abs_functor = to_paths_abs_functor\
        .fmap(lambda x: x.replace('\\', '/'))\
        .fmap(lambda x: x.replace(target_dir[1:], root_dir[1:]))
    
    for from_path in from_path_abs_functor.values:
        # save_folder = Functor(from_path)\
        #     .fmap(extract_subfolder, root_dir=root_dir)\
        #     .fmap(os.path.abspath)\
        #     .fmap(lambda x: x.replace('\\', '/'))\
        #     .fmap(lambda path: path.replace(root_dir, target_dir))\
        #     .value
        
        save_folder = Functor(from_path)\
            .fmap(lambda s: s.replace(root_dir, target_dir))\
            .fmap(omit_filename)\
            # .value
        
        # print(from_path, save_folder, sep='\n')
        print(save_folder)
        # shutil.copy(from_path, save_folder)
    
    # print(subfolders_functor)
    # print(files_flatten)
    # print(target_subfolders)
    # print(*zip(from_path_abs_functor.values, to_paths_abs_functor.values), sep='\n')
    


def folder_to_zip(folder_path, zip_path):
    import zipfile
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname=relative_path)


def remove_folder(dir):
    if os.path.exists(dir) and os.path.isdir(dir):
        shutil.rmtree(dir)
        print(f"Removed folder: {dir}")
    else:
        print(f"Folder does not exist: {dir}")
    

def zipify_dirs(dirs, zip_dest='data/wm_dataset.zip', n=100):
    temp_dir = '/data/temp_dir'    
    file_paths = get_files_path(dirs, n=n)
    copy_files_from_to(file_paths, root_dir='/data', target_dir=temp_dir)
    folder_to_zip(temp_dir, zip_dest)
    # remove_folder(os.path.abspath(temp_dir))


image_dirs = ['data/upscaled', 'data/watermark_upscaled', 
              'data/_alpha', 'data/_mask', 'data/_wm']

# get_files_path(image_dirs, n=10)
zipify_dirs(image_dirs, n=10)


