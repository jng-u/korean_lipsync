import os
import sys
import glob
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--input', dest='input_folder', type=str)
parser.add_argument('--output', dest='output_folder', type=str)
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder

flist = glob.glob(input_folder+'/**/*.txt', recursive=True)
flist.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-8]))
for file in flist:
    print("reading file: %s" % file)
    
    wpath = output_folder+file[len(input_folder):]
    os.makedirs(os.path.dirname(wpath), exist_ok=True)
    count = int(os.path.basename(wpath)[:len(os.path.basename(wpath))-8])
    if count/500 > 0:
        count = count%500
    shutil.copy(file, os.path.dirname(wpath)+'/{}.txt'.format(count))