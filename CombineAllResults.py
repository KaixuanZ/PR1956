#combine the results of row segmentation, row classification, and OCR output. Output format can be csv or json.

from joblib import Parallel, delayed
import argparse
import sys
import multiprocessing
import os
import json
import cv2
import numpy as np
from google.protobuf.json_format import MessageToDict,Parse
from google.cloud.vision_v1.proto import image_annotator_pb2
import pandas as pd
import Rect

clean_names = lambda x: [i for i in x if i[0] != '.']

class Row(object):
    def __init__(self, rect=None, rect_filename=None,cls=None):
        self.row_rect = rect  # rect on col image
        if rect:
            self.row_bbox = cv2.boxPoints(tuple(rect)).tolist()
        self.cls = cls
        self.words = []
        self.AOIs = []  # area of intersections (normalized by area of each word)
        self.key = {}  # file,page,subpage,col,row
        if rect_filename:
            self.FilenameToKey(rect_filename)

    def FilenameToKey(self, rect_filename):
        self.key['book'], self.key['file'], self.key['subfile'], self.key['page'], self.key['col'], self.key['row'] = rect_filename.split('.')[0].split('_')

    def ToDict(self):
        dict = {}
        dict['row_rect'] = self.row_rect
        dict['row_bbox'] = self.row_bbox
        dict['words'] = self.words
        dict['AOIs'] = self.AOIs
        dict['key'] = self.key
        dict['cls'] = self.cls
        return dict

    def ToDF(self):
        data = []
        label = ['book', 'file', 'subfile', 'page', 'col', 'row', 'cls', 'row_bbox', 'row_rect']
        row = [self.key['book'], self.key['file'], self.key['subfile'], self.key['page'], self.key['col'],
               self.key['row'], self.cls, self.row_bbox, self.row_rect]

        get_box = lambda dict: [[dict['vertices'][0]['x'], dict['vertices'][0]['y']],
                                [dict['vertices'][1]['x'], dict['vertices'][1]['y']],
                                [dict['vertices'][2]['x'], dict['vertices'][2]['y']],
                                [dict['vertices'][3]['x'], dict['vertices'][3]['y']]]

        label += ['AOI', 'word_bbox', 'word_confidence']
        label += ['symbol_bbox', 'symbol_confidence', 'symbol']
        text = ''
        for i in range(len(self.words)):
            try:
                row1 = row + [self.AOIs[i], get_box(self.words[i]['boundingBox']), self.words[i]['confidence']]
            except:
                row1 = row + [self.AOIs[i], None, self.words[i]['confidence']]
            for symbol in self.words[i]['symbols']:
                text += symbol['text']
                try:
                    row2 = row1 + [get_box(symbol['boundingBox']), symbol['confidence'], symbol['text']]
                except:
                    try:
                        row2 = row1 + [None, symbol['confidence'], symbol['text']]
                    except:
                        row2 = row1 + [None, None, symbol['text']]
                data.append(row2)
        if len(self.words)==0:
            data.append(row+[None]*6)
        df = pd.DataFrame.from_records(data, columns=label)
        df.insert(len(label), 'text', text)

        return df

    def ToJson(self, jsonfile=None,info=False):
        if jsonfile:
            with open(jsonfile, 'w') as outfile:
                json.dump(self.ToDict(), outfile)
            if info:
                print("writing data to " + jsonfile)
        else:
            return self.ToDict()

    def FromJson(self, jsonfile):
        with open(jsonfile) as file:
            data = json.load(file)
        self.row_rect = data['row_rect']
        self.row_bbox = data['row_bbox']
        self.words = data['words']
        self.AOIs = data['AOIs']
        self.key = data['key']
        self.cls = data['cls']


def assign_document_word_to_row(word, rows):
    areas,dists = [],[]
    box = np.array([[word.bounding_box.vertices[0].x, word.bounding_box.vertices[0].y],
                    [word.bounding_box.vertices[1].x, word.bounding_box.vertices[1].y],
                    [word.bounding_box.vertices[2].x, word.bounding_box.vertices[2].y],
                    [word.bounding_box.vertices[3].x, word.bounding_box.vertices[3].y]])
    word_rect = cv2.minAreaRect(box)
    for i in range(len(rows)):
        areas.append(Rect.AreaOfOverlap(word_rect, rows[i].row_rect))
        dists.append(Rect.DistOfRects(word_rect, rows[i].row_rect))
    max_area=0
    if max(areas)>0:
        #assign word to the row_img with max_area
        max_area=max(areas)
        index=areas.index(max_area)
    else:
        #if the word has no intersection with any row img, assign it to closest row_img
        index=dists.index(min(dists))
    rows[index].words.append(MessageToDict(word))
    rows[index].AOIs.append(max_area / word_rect[1][0] / word_rect[1][1])


def assign_document_words_to_row(ocr_file, rows):
    with open(ocr_file) as json_file:
        data = json.load(json_file)
        parsed = image_annotator_pb2.AnnotateImageResponse()
        Parse(data, parsed)
        # import pdb; pdb.set_trace()

    document = parsed.full_text_annotation
    # Collect specified feature bounds by enumerating all document features
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    assign_document_word_to_row(word, rows)
    return rows


def CombineResToRow(img, col_rect_json, row_rect_json, cls):
    with open(col_rect_json) as jsonfile:
        col_rect = json.load(jsonfile)
    _, M = Rect.CropRect(img, col_rect)
    M = np.linalg.inv(M)

    with open(row_rect_json) as jsonfile:
        row_rect = json.load(jsonfile)
        row_rect = Rect.RectOnSrcImg(cv2.boxPoints(tuple(row_rect)), M)
        row = Row(row_rect, row_rect_json.split('/')[-1], cls)

    return row

def SaveRowsToCSV(rows,output_file):
    if len(rows):
        res=rows[0].ToDF()
        for row in rows[1:]:
            res= res.append(row.ToDF())
        res.to_csv(output_file)
        print("saving output to "+output_file)
    else:
        print("no data for writing to CSV")

def main(page_dir, args):
    print("processing "+page_dir)

    #read in image
    book,file,subfile,page = page_dir.split('_')
    imgfile=book+"_"+file[0]+str(int(file[1:]))+"_"+subfile+".tif"
    img=cv2.imread(os.path.join(args.img_dir,imgfile))

    cls_json=os.path.join(args.row_cls_dir,page_dir+'.json')
    with open(cls_json) as jsonfile:
        cls=json.load(jsonfile)
    cls=cls["name"]
    row_rect_files=sorted(clean_names(os.listdir(os.path.join(args.row_rect_dir,page_dir))))

    rows=[]
    row_nums=[0,0,0,0,0]
    #get class Row
    for i in range(len(row_rect_files)):
        row_rect_json = row_rect_files[i]
        row_rect = os.path.join(args.row_rect_dir,page_dir,row_rect_json)
        col_rect_json = row_rect_json[:-9]+row_rect_json[-5:]
        col_rect = os.path.join(args.col_rect_dir, page_dir, col_rect_json)
        rows.append(CombineResToRow(img, col_rect, row_rect, cls[i]))
        row_nums[int(rows[i].key['col'])]+=1
    #combine OCR with row
    ocr_jsons = sorted(clean_names(os.listdir(os.path.join(args.OCR_dir, page_dir))))

    for i in range(len(ocr_jsons)):
        ocr_json = os.path.join(args.OCR_dir, page_dir, ocr_jsons[i])
        l,r = sum(row_nums[:i]),sum(row_nums[:i+1])
        rows = rows[:l]+ assign_document_words_to_row(ocr_json, rows[l:r]) +rows[r:]

    #save results to json
    if not os.path.isdir(os.path.join(args.output_dir,'json')):
        os.mkdir(os.path.join(args.output_dir,'json'))
        print('creating directory ' + os.path.join(args.output_dir,'json'))
    for i in range(len(rows)):
        json_dir = os.path.join(args.output_dir,'json',page_dir)
        if not os.path.isdir(json_dir):
            os.mkdir(json_dir)
        jsonfile = os.path.join(json_dir,page_dir+"_"+rows[i].key['row']+".json")
        rows[i].ToJson(jsonfile,i%100==0)

    #save results to csv
    if not os.path.isdir(os.path.join(args.output_dir,'csv')):
        os.mkdir(os.path.join(args.output_dir,'csv'))
        print(os.path.join(args.output_dir,'csv'))
    SaveRowsToCSV(rows,os.path.join(args.output_dir,'csv',page_dir+'.csv'))

        #import pdb;pdb.set_trace()

class Args(object):
    def __init__(self, args):
        self.img_dir = args.img_dir
        self.col_rect_dir = args.col_rect_dir
        self.row_rect_dir = args.row_rect_dir
        self.row_cls_dir = args.row_cls_dir
        self.OCR_dir = args.OCR_dir
        self.output_dir = args.output_dir


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='combine row segmentation, row classification, and OCR output')
    parser.add_argument( '--img_dir', type=str)
    parser.add_argument( '--col_rect_dir', type=str)
    parser.add_argument( '--row_rect_dir', type=str)
    parser.add_argument( '--row_cls_dir', type=str)
    parser.add_argument( '--OCR_dir', type=str)
    parser.add_argument( '--output_dir', type=str)

    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        print('creating directory ' + args.output_dir)

    page_dirs = sorted(clean_names(os.listdir(args.row_rect_dir)))
    n = len(page_dirs)
    args = [args] * n
    #Parallel(n_jobs=1)(map(delayed(main), page_dirs, args))
    Parallel(n_jobs=multiprocessing.cpu_count())(map(delayed(main), page_dirs, args))