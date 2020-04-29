import os
import glob
import shutil
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', dest='input_folder', type=str)
parser.add_argument('--output', dest='output_folder', type=str)
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder
os.makedirs(output_folder, exist_ok=True)

file_list = glob.glob(input_folder+'/**/*.*', recursive=True)
# file_list = np.array(file_list, dtype='int')
file_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
# file_list = np.array(file_list, dtype='str')
total = len(file_list)
wpath =  output_folder+'/0'
os.makedirs(wpath, exist_ok=True)
for i, file in enumerate(file_list):
    if (i+1)%500 == 0:
        print('{}/{} to {}'.format(i+1, total, wpath))
        wpath = output_folder+'/{}'.format(int((i+1)/500))
        os.makedirs(wpath, exist_ok=True)
    shutil.move(file, wpath+'/{}'.format(os.path.basename(file)))

    