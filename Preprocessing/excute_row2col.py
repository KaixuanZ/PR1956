import itertools
import os
import subprocess
import argparse
import glob
import shutil
import datetime
import pandas as pd

parser = argparse.ArgumentParser(description='Crop the image into two pages')
parser.add_argument('-r1', '--page_range_1', type=int,
                    help='range of pages')
parser.add_argument('-r2', '--page_range_2', type=int,
                    help='range of pages')
parser.add_argument('-i', '--input_folder', type=str,
                    help='directory of the input folder')
parser.add_argument('-o', '--output_folder', type=str,
                    help='directory of the output folder')
args = parser.parse_args()


def prepare_dir(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except FileNotFoundError:
            print("please create directory")
    else:
        files = glob.glob(os.path.join(dir, '*'))
        for f in files:
            try:
                os.remove(f)
            except IsADirectoryError:
                shutil.rmtree(f)

prepare_dir(args.input_folder)
prepare_dir(os.path.join(args.output_folder, 'col'))
prepare_dir(os.path.join(args.output_folder, 'cls'))

l1 = [x for x in range(args.page_range_1, args.page_range_2)]
l2 = [1,2] #[1,2,3,4,5]
l3 = [0,1]
l4 = [0, 1, 2, 3, 4, 5, 6, 7]

s3_output_log = 's3://teikoku/1957/log/'
item = pd.DataFrame(list(itertools.product(l1, l2, l3, l4)), columns=['file', 'page', 'subpage', 'row'])
item['error'] = 0
for index, row in item.iterrows():
    file = row['file']
    page = row['page']
    subpage = row['subpage']
    r = row['row']

    row = 'tk1957_f%d_%d_%d_%d' % (file, page, subpage, r)
    print('###########################')
    print('python3 row2col.py -f %s -i %s -o %s' % (row,
                                                    args.input_folder,
                                                    args.output_folder))
    print('###########################')
    try:
        subprocess.call('python3 row2col.py -f %s -i %s -o %s' % (row,
                                                                       args.input_folder,
                                                                       args.output_folder),
                        shell=True)
    except:
        item.loc[index, 'error'] = 1

filename = 'tk1957_row2col_log_%s.csv' % (datetime.today().isoformat())
item.to_csv(filename, sep=',')
subprocess.call('aws s3 cp %s %s' % (filename, s3_output_log), shell=True)
