import os

def write_png_paths(folder_path, txt_path):
    with open(txt_path, 'w') as f:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.png'):
                    f.write(os.path.join(root, file) + '\n')

# Example usage:
folder_path = '/data2/Solar_Data/LSDIR/'
txt_path = '/data2/Solar_Data/PiSA-SR/preset/gt_path.txt'
write_png_paths(folder_path, txt_path)
