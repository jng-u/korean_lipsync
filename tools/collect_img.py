import os
import glob
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', dest='input_folder', type=str)
parser.add_argument('--output', dest='output_folder', type=str)
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder
os.makedirs(output_folder, exist_ok=True)

flist = glob.glob('{}/**/*.jpg'.format(input_folder), recursive=True)
cnt=0
for f in flist:
    if cnt%100==0:
        print('process %d image' %cnt)
    write_path = output_folder + ('/%6d' % cnt).replace(' ', '0') + '.jpg'
    shutil.copy(f, write_path)
    cnt+=1