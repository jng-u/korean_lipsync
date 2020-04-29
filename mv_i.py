import os
import shutil
import glob

file_list = glob.glob('data/img/*.*', recursive=True)
file_list.sort()
for i, file in enumerate(file_list):
    if(i>3000): 
        break
    shutil.copy(file, 'data/aa/{}'.format(os.path.basename(file)))