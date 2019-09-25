import cv2
import os
from joblib import Parallel, delayed


def pdf_to_img(filename,input_dir, output_dir):
    print("processing "+filename)
    input_path=os.path.join(input_dir,filename)
    output_path=os.path.join(output_dir,filename[:-4]+'.png')

    img=cv2.imread(input_path)
    cv2.imwrite(output_path,img)

input_dir="../../raw_data/personnel-records/1947/scans/parsed/"
output_dir=input_dir
clean_names = lambda x: [i for i in x if i[0] != '.' and i.endswith('.tif')]
filenames=sorted(clean_names(os.listdir(input_dir)))

Parallel(n_jobs=-1)(map(delayed(pdf_to_img), filenames, [input_dir]*len(filenames), [output_dir]*len(filenames)))