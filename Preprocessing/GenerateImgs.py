import os
import io
import json
from joblib import Parallel, delayed
import argparse
import cv2
import sys
sys.path.append('../')
import Rect

def main(rect_json, args):
    if 'row' in args.rectdir:
        row=True
    else:
        row=False
    print('processing ' + rect_json)

    imgpath = os.path.join(args.imgdir, '_'.join(rect_json.split('_')[:-1])+ '.png')
    img=cv2.imread(imgpath)


    with open(os.path.join(args.rectdir, rect_json)) as file:
        rects = json.load(file)

    outputdir = os.path.join(args.outputdir, rect_json.split('.')[0])
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)


    if row:     #output row
        for key in sorted(rects.keys()):
            #import pdb;pdb.set_trace()
            i = 0
            for rect in rects[key]:
                warped, _ = Rect.CropRect(img, rect)
                filename = rect_json.split('.')[:-1]
                filename.append(str(key))
                filename.append(str(i).zfill(3))
                cv2.imwrite(os.path.join(outputdir, '_'.join(filename)+ '.png'), warped)
                i+=1
    else:       #output col
        i = 0
        for rect in rects:
            #import pdb;pdb.set_trace()
            warped, _ = Rect.CropRect(img, rect)
            filename=rect_json.split('.')[:-1]
            filename.append(str(i))
            cv2.imwrite(os.path.join(outputdir,'_'.join(filename) + '.png'),warped)
            i+=1

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Page Detection')
    parser.add_argument('--imgdir', type=str)
    parser.add_argument('--rectdir', type=str)
    parser.add_argument('--outputdir', type=str)
    args = parser.parse_args()
        #create output file

    if not os.path.isdir(args.outputdir):
        os.mkdir(args.outputdir)
        print('creating directory ' + args.outputdir)

    clean_names = lambda x: [i for i in x if i[0] != '.']

    rects = os.listdir(args.rectdir)
    rects = sorted(clean_names(rects))

    Parallel(n_jobs=-1)(map(delayed(main), rects, [args]*len(rects)))
