import os
import io
import json
from joblib import Parallel, delayed
import argparse
import multiprocessing
import cv2
import sys
sys.path.append('../')
import Rect

def GetImgFilename(jsonfile):
    book, f, n = jsonfile.split('.')[0].split('_')[0:3]
    f = f[0] + str(int(f[1:]))
    return book + '_' + f + '_' + n + '.tif'

def main(imgdir, rectdir, outputdir):
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
        print('creating directory ' + outputdir)

    clean_names = lambda x: [i for i in x if i[0] != '.']
    RectJsons = sorted(clean_names(os.listdir(rectdir)))

    imgpath = os.path.join(imgdir, GetImgFilename(RectJsons[0]))
    img=cv2.imread(imgpath)

    Rects, JsonNames = [], []
    for colRectJson in RectJsons:
        with open(os.path.join(rectdir, colRectJson)) as file:
            Rects.append(json.load(file))
            JsonNames.append(colRectJson)

    for i in range(len(Rects)):
        warped, _ = Rect.CropRect(img, Rects[i])
        cv2.imwrite(os.path.join(outputdir,JsonNames[i].split('.')[0]+'.png'),warped)

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
    rectdir = os.listdir(args.rectdir)
    rectdir = sorted(clean_names(rectdir))

    outputdir = [os.path.join(args.outputdir, dir) for dir in rectdir]
    rectdir = [os.path.join(args.rectdir, dir) for dir in rectdir]
    imgdir = [args.imgdir] * len(rectdir)

    Parallel(n_jobs=1)(map(delayed(main), imgdir, rectdir, outputdir))
    #Parallel(n_jobs=multiprocessing.cpu_count())(map(delayed(main), imgdir, rectdir, outputdir))
