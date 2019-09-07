import argparse
import io
import os
import json
import cv2
import numpy as np
from google.protobuf.json_format import MessageToDict,Parse
from google.cloud.vision_v1.proto import image_annotator_pb2
import pandas as pd
import Rect


class Row(object):
    def __init__(self, rect=None, rect_filename=None):
        self.row_rect = rect  # rect on col image
        if rect:
            self.row_bbox = cv2.boxPoints(tuple(rect)).tolist()
        self.cls = None
        self.words = []
        self.AOIs = []  # area of intersections (normalized by area of each word)
        self.key = {}  # file,page,subpage,col,row
        if rect_filename:
            self.FilenameToKey(rect_filename)

    def FilenameToKey(self, rect_filename):
        self.key['book'], self.key['file'], self.key['subfile'], self.key['page'], self.key['col'], self.key['row'] = \
        rect_filename.split('.')[0].split('_')

    def ToDict(self):
        dict = {}
        dict['row_rect'] = self.row_rect
        dict['row_bbox'] = self.row_bbox
        dict['words'] = self.words
        dict['AOIs'] = self.AOIs
        dict['key'] = self.key
        dict['cls'] = self.cls
        return dict

    def ToCSV(self, csvfile='tmp.csv'):
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
            row1 = row + [self.AOIs[i], get_box(self.words[i]['boundingBox']), self.words[i]['confidence']]
            for symbol in self.words[i]['symbols']:
                text += symbol['text']
                row2 = row1 + [get_box(symbol['boundingBox']), symbol['confidence'], symbol['text']]
                data.append(tuple(row2))

        df = pd.DataFrame.from_records(data, columns=label)
        df.insert(len(label), 'text', text)
        df.to_csv(csvfile)

    def ToJson(self, jsonfile='tmp.json'):
        with open(jsonfile, 'w') as outfile:
            json.dump(self.ToDict(), outfile)

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
        max_area=max(areas)
        index=areas.index(max_area)
    else:
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


def OCRwithRow(image_file, col_rect_json, row_rect_dir, ocr_file, fileout='tmp.json'):
    row_rects = os.listdir(row_rect_dir)
    clean_names = lambda x: [i for i in x if i[0] != '.']
    row_rects = sorted(clean_names(row_rects))
    dict = {}
    for row_rect in row_rects:
        key = int(row_rect.split('_')[-2])
        if key in dict.keys():
            dict[key].append(row_rect)
        else:
            dict[key] = [row_rect]

    img = cv2.imread(image_file)
    with open(col_rect_json) as jsonfile:
        col_rect = json.load(jsonfile)
    _, M = Rect.CropRect(img, col_rect)

    rows = []
    for row_rect_file in dict[0]:  # files for row_rect in first col
        # trans row_rect into the coordinate of col img
        with open(os.path.join(row_rect_dir, row_rect_file)) as jsonfile:
            row_rect = json.load(jsonfile)
            rows.append(Row(Rect.RectOnSrcImg(cv2.boxPoints(tuple(row_rect)), np.linalg.inv(M)), row_rect_file))

    assign_document_words_to_row(ocr_file, rows)

    rows[0].ToJson(fileout)


OCRwithRow('pr1956_f0047_0_0.png', 'pr1956_f0047_0_0_0.json', 'pr1956_f0047_0_0', 'OCR/pr1956_f0047_0_0_0.json',
           'tmp.json')

row_test = Row()
row_test.FromJson('tmp.json')
import pdb;

pdb.set_trace()
tmp = row_test.ToCSV()
