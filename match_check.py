import os
from os.path import splitext, join, isfile

def get_file_prefixes(directory):
    """
    Get the list of file prefixes (filename without extension) from the given directory.
    """
    return {splitext(f)[0] for f in os.listdir(directory) if isfile(join(directory, f))}

def rename_files(directory, prefix_to_remove):
    """
    Rename files in a directory by removing the specified prefix from their filenames.
    """
    for filename in os.listdir(directory):
        if isfile(join(directory, filename)) and filename.startswith(prefix_to_remove):
            new_name = filename.replace(prefix_to_remove, '', 1)  # Replace the prefix only at the start
            old_file_path = join(directory, filename)
            new_file_path = join(directory, new_name)
            try:
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {old_file_path} -> {new_file_path}")
            except Exception as e:
                print(f"Failed to rename {old_file_path}: {e}")

def delete_files(directory, unmatched_files):
    """
    Delete files from a directory based on the file prefix (unmatched files).
    """
    for file_prefix in unmatched_files:
        files_to_delete = [f for f in os.listdir(directory) if f.startswith(file_prefix)]
        for f in files_to_delete:
            file_path = join(directory, f)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

def match_check(original_dir, mask_dir, delete_unmatched=False):
    """
    Check if all files in original_dir have corresponding masks in mask_dir and vice versa.
    Optionally, delete files that do not have a one-to-one match.
    """
    # Get the file prefixes from both directories
    original_files = get_file_prefixes(original_dir)
    mask_files = get_file_prefixes(mask_dir)

    # Find unmatched files
    unmatched_original = original_files - mask_files
    unmatched_masks = mask_files - original_files

    # Print unmatched files
    if not unmatched_original and not unmatched_masks:
        print("All files are correctly matched.")
    else:
        if unmatched_original:
            print("Files in original folder with no matching mask:")
            for f in unmatched_original:
                print(f)
        if unmatched_masks:
            print("Files in mask folder with no matching original image:")
            for f in unmatched_masks:
                print(f)

    # Optionally delete unmatched files
    if delete_unmatched:
        if unmatched_original:
            print("\nDeleting unmatched files in original folder...")
            delete_files(original_dir, unmatched_original)
        if unmatched_masks:
            print("\nDeleting unmatched files in mask folder...")
            delete_files(mask_dir, unmatched_masks)

if __name__ == '__main__':
    original_dir = '/home/keith/Downloads/NU Works/Research/Data/only_s1/original_only_s1'  # Your actual original images folder path
    mask_dir = '/home/keith/Downloads/NU Works/Research/Data/only_s1/mask_only_s1'  # Your actual masks folder path
    delete_unmatched = True  # Set to True to enable deletion of unmatched files

    # Rename the files first
    rename_files(original_dir, 'img_')  # Remove 'img_' from original folder filenames
    rename_files(mask_dir, 'mask_')  # Remove 'mask_' from mask folder filenames

    # Run the matching check and delete unmatched files
    match_check(original_dir, mask_dir, delete_unmatched)

