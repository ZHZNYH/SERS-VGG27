import os
import glob

def merge_files(original_filename):
    """
    Find all part files and merge and restore them
    """
    # Match all volume files (e.g. SERD_Data_10%. xlsx.part1...)
    parts = glob.glob(f"{original_filename}.part*")
    
    # Sort in order of part1, part2, part3, part4
    parts.sort(key=lambda x: int(x.split('.part')[-1]))
    
    if not parts:
        print("Volume file not found！")
        return

    print(f"Found {len (parts)} volumes, merging in progress...")
    
    with open(original_filename, 'wb') as outfile:
        for part in parts:
            with open(part, 'rb') as infile:
                outfile.write(infile.read())
            print(f"Merged: {part}")
            
    print(f"Merge successful! The file has been restored to: {original_filename}")

if __name__ == "__main__":
    target_file = "SERS_Data_10%.xlsx"
    merge_files(target_file)