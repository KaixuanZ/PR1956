# -*- encoding: utf-8 -*-
import os
import cv2
import csv
import argparse
#import pdb; pdb.set_trace()

def OutputImg(img,outputpath,label):
    filepath=os.path.join(outputpath,label)
    if not os.path.isdir(filepath):
        os.mkdir(filepath)
    n=str(len(os.listdir(filepath)))
    outpath=os.path.join(filepath,n.zfill(5)+'.png')
    cv2.imwrite(outpath,img)
    print("saving image to "+outpath)

def ReadImg(imgpath,row):
    #for teikoku1957
    #filename='tk1957_f'+row[1]+'_'+row[2]+'_'+row[3]+'_'+row[4]
    #filepath=os.path.join(imgpath,filename,row[5].zfill(3)+'.png')

    #for pr1954
    filename='pr1954_p'+row[1]+'_'+row[2]+'_'+row[3]
    filepath=os.path.join(imgpath,filename,filename+'_'+row[4]+'.png')
    img=cv2.imread(filepath,0)  #gray img (width height)
    return img

def GenerateDataset(imgpath,outputpath,labelfile,dict):
    if not os.path.isdir(outputpath):
        os.mkdir(outputpath)
    with open(labelfile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            elif row[0] is not '':
                img=ReadImg(imgpath,row)
                OutputImg(img,outputpath,dict[row[0]])
            else:
                break

def GetLabelDict(csv_file):
    dict={}
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            #import pdb; pdb.set_trace()
            if line_count == 0:
                line_count += 1
            else:
                if row[0] is not '':
                    dict[row[0]]=row[1]
                else:
                    break
    return dict


if __name__ == '__main__':
    # load_cls_file(246, 295, '../1954/ocr/cls/')
    parser = argparse.ArgumentParser(description='Generate Dataset')
    parser.add_argument('--imgpath', type=str)
    parser.add_argument('--labelfile', type=str)
    parser.add_argument('--labeldict', type=str)
    parser.add_argument('--outputpath', type=str)

    args = parser.parse_args()

    dict=GetLabelDict(args.labeldict)
    print(dict)

    GenerateDataset(args.imgpath,args.outputpath,args.labelfile,dict)
