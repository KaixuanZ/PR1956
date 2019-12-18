#combine the results of row segmentation, row classification, and OCR output. Output format can be csv or json.

from joblib import Parallel, delayed
import argparse
import os
import json
import cv2
import numpy as np
from google.protobuf.json_format import MessageToDict,Parse
from google.cloud.vision_v1.proto import image_annotator_pb2
import pandas as pd
import Rect

clean_names = lambda x: [i for i in x if i[0] != '.']

class Page(object):
    def __init__(self,img=None,page_filename=None,ocr_jsonfiles=None,col_rects=None,row_rects=None):
        '''
        :param img:             image of this page
        :param page_filename:   filename of img
        :param ocr_jsonfiles:   a list of path of OCR output [ocr_jsonfile]
        :param col_rects:       a list of col_rect in this page [col_rect]
        :param row_rects:       a dict of row_rect in this page {'col_num':[row_rect]}
        each rect is a list contrains five parameters [[x,y],[H,W],theta]
        '''
        self.img=img
        self.cols=[]
        self.ocr_jsonfiles=ocr_jsonfiles
        self.col_rects=col_rects
        self.row_rects=row_rects
        col_width=[min(col_rect[1]) for col_rect in col_rects]
        self.col_width=np.median(col_width)
        self.page_index={}
        if page_filename:
            self.FilenameToKey(page_filename)

    def FilenameToKey(self, filename):
        self.page_index['book'], self.page_index['file'], self.page_index['subfile'],self.page_index['page']= filename.split('.')[0].split('_')

    def SetCols(self):
        row_nums=[0]
        for key in sorted(self.row_rects.keys()):
            row_nums.append(row_nums[-1]+len(self.row_rects[key]))
        for i in range(len(self.ocr_jsonfiles)):
            _,M_scan2col = Rect.CropRect(self.img, self.col_rects[i])
            self.cols.append(Col(self.ocr_jsonfiles[i],M_scan2col, self.row_rects[str(i)], i))
            self.cols[-1].SetRows()

    def SaveToCsv(self,outputpath):
        #save information of this page to one csv file
        df=self.ToDataFrame()
        df.index.name="symbol_id"
        df.to_csv(outputpath)
        print("saving output to " + outputpath)

    def ToDataFrame(self):
        #reshape inforamtion of this page to dataframe
        label=[*self.page_index.keys()]
        val=[*self.page_index.values()]
        label.append('col_width')
        val.append(self.col_width)
        df=[]
        for col in self.cols:
            df.append(col.ToDataFrame(val,label))
        return pd.concat(df)

class Col(object):
    def __init__(self,ocr_jsonfile=None,M_scan2col=None,row_rects=None,col_index=None):
        '''
        :param ocr_jsonfile:    path of OCR output
        :param M_scan2col:           transformation from scanned image to col
        :param row_rects:       a list of row_rect in this column [row_rect]
        :param col_index:       index of this column in page
        '''
        self.ocr_jsonfile=ocr_jsonfile
        self.rows=[]
        self.row_rects=row_rects
        self.M_scan2col=M_scan2col
        self.col_index=str(col_index)

    def SetRows(self,OCR=True):
        for i in range(len(self.row_rects)):
            #row rect on col img coordinate
            row_rect = Rect.RectOnDstImg(self.row_rects[i], self.M_scan2col)
            self.rows.append(Row(row_rect,i))
        if OCR:
            self.AssignDocumentWordsToRow()

    def AssignDocumentWordsToRow(self):
        '''
        assign OCR output to coorespondent row
        :return: self.rows with OCR output (self.rows.words, self.rows.AOI)
        '''
        with open(self.ocr_jsonfile) as json_file:
            data = json.load(json_file)
            parsed = image_annotator_pb2.AnnotateImageResponse()
            Parse(data, parsed)

        document = parsed.full_text_annotation
        # Collect specified feature bounds by enumerating all document features
        for page in document.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        #assign_document_word_to_row(word, self.rows)
                        for symbol in word.symbols:
                            assign_document_word_to_row(symbol, self.rows)
        #use OCR information to correct potential row segmentation errors
        self.CheckRowSeg()

    def CheckRowSeg(self):
        for row in self.rows:
            row.CheckRowSeg()

    def ToDataFrame(self,val,label):
        '''
        :param val:     values of correspondent labels from page level
        :param label:   labels come from page level
        :return:        a dataframe which contains information of this column
        '''

        label_col=label+['col']
        val_col=val+[self.col_index]
        df=[]
        for row in self.rows:
            df.append(row.ToDataFrame(val_col,label_col))
        return pd.concat(df)

class Row(object):
    def __init__(self, row_rect=None , row_index=None):
        '''
        :param row_rect:    row_rect of this row
        :param row_index:   index of this row in column
        :param subrows:     if there are multiple rows in this row image, indicate each row by subrow
        '''
        self.row_bbox = None
        self.row_rect = row_rect
        if row_rect:
            self.row_bbox = Rect.OrderPoints(cv2.boxPoints(tuple(row_rect))).tolist()
        self.words = []
        self.subrows= []
        self.AOIs = []  # area of intersections (normalized by area of each word)
        self.row_index = str(row_index).zfill(3)

    def ToDict(self):
        dict = {}
        dict['row_bbox'] = self.row_bbox
        dict['words'] = self.words
        dict['AOIs'] = self.AOIs
        dict['row_index']=self.row_index
        return dict

    def CheckRowSeg(self):
        get_word_height = lambda word: word['boundingBox']['vertices'][-1]['y'] - word['boundingBox']['vertices'][0]['y']
        if len(self.words) > 1:
            #get estimated word height
            word_heights=[]
            for i in range(len(self.words)-1,-1,-1):
                try:
                    word_heights.append(get_word_height(self.words[i]))
                except:
                    pass

            word_height_median=np.median(word_heights)

            # check if all the words are in one row, but ignore the extremely small one (<1/4 of median symbol height)
            y0,y1,i=0,10000,0
            for word in self.words:
                try:
                    if get_word_height(word)>0.25*word_height_median:
                        y0 = max(y0, word['boundingBox']['vertices'][0]['y'])
                        y1 = min(y1, word['boundingBox']['vertices'][-1]['y'])
                        if y1-y0<0.25*min(word_height_median,get_word_height(word)):
                            y0,y1=word['boundingBox']['vertices'][0]['y'],word['boundingBox']['vertices'][-1]['y']
                            i+=1
                    self.subrows.append(i)
                except:
                    self.subrows.append(i)
        else:
            self.subrows=[0]*len(self.words)
        #import pdb;pdb.set_trace()

    def ToDataFrame(self,val,label):
        '''
        :param val:     values of correspondent labels from column level
        :param label:   labels come from column level
        :return:        a dataframe which contains information of this row
        '''
        data = []
        label_row = label+['row', 'row_bbox']
        val_row = val+ [self.row_index,self.row_bbox]

        get_box = lambda dict: [[dict['vertices'][0]['x'], dict['vertices'][0]['y']],
                                [dict['vertices'][1]['x'], dict['vertices'][1]['y']],
                                [dict['vertices'][2]['x'], dict['vertices'][2]['y']],
                                [dict['vertices'][3]['x'], dict['vertices'][3]['y']]]
        label=['subrow','AOI', 'word_bbox', 'word_confidence']+['symbol_bbox', 'symbol_confidence', 'symbol']
        label_row += label
        text = ''
        for i,symbol in enumerate(self.words):
            try:
                val_row1 = val_row + [self.subrows[i],self.AOIs[i], get_box(self.words[i]['boundingBox']), self.words[i]['confidence']]
            except:
                try:
                    val_row1 = val_row + [self.subrows[i],self.AOIs[i], None, self.words[i]['confidence']]
                except:
                    val_row1 = val_row + [self.subrows[i], self.AOIs[i], None, None]

            text += symbol['text']
            try:
                val_row2 = val_row1 + [get_box(symbol['boundingBox']), symbol['confidence'], symbol['text']]
            except:
                try:
                    val_row2 = val_row1 + [None, symbol['confidence'], symbol['text']]
                except:
                    val_row2 = val_row1 + [None, None, symbol['text']]
            data.append(val_row2)

        if len(self.words)==0:
            data.append(val_row+[0]+[None]*(len(label)-1))
        df = pd.DataFrame.from_records(data, columns=label_row)
        df.insert(len(label_row), 'text', text)
        return df

def assign_document_word_to_row(word, rows):
    '''
    assign word to nearest row (L2 distance)
    :param word: a word returned by Google Cloud Vision
    :param rows: a list of rows in one column
    :return: rows with information of the input word
    '''
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
    try:
        rows[index].AOIs.append(max_area / word_rect[1][0] / word_rect[1][1])
    except: #   0/0
        rows[index].AOIs.append(0)

def main(page_index, args):
    '''
    :param page_index:  page to be processed
    :param args:
    :return:            an object page
    '''
    if "index" not in args.output_dir:
        print("this function is only for index section")
        return 0
    print("processing "+page_index)

    #read in image
    #import pdb;pdb.set_trace()
    imgfile='_'.join(page_index.split('_')[:-1])+'.png'
    img=cv2.imread(os.path.join(args.img_dir,imgfile),0)

    #get ocr json filenames
    ocr_jsons = [os.path.join(args.OCR_dir,page_index,ocrfile) for ocrfile in sorted(clean_names(os.listdir(os.path.join(args.OCR_dir, page_index))))]

    #get col_rects
    with open(os.path.join(args.rect_dir,'col_rect',page_index+'.json')) as jsonfile:
        col_rects=json.load(jsonfile)

    #get row_rects
    with open(os.path.join(args.rect_dir,'row_rect',page_index+'.json')) as jsonfile:
        row_rects=json.load(jsonfile)

    page=Page(img=img,page_filename=page_index,ocr_jsonfiles=ocr_jsons,col_rects=col_rects,row_rects=row_rects)
    page.SetCols()

    #save results to csv
    page.SaveToCsv(os.path.join(args.output_dir,page_index+'.csv'))


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='combine row segmentation, row classification, and OCR output')
    parser.add_argument( '--img_dir', type=str)
    parser.add_argument( '--rect_dir', type=str)
    parser.add_argument( '--row_cls_dir', type=str)
    parser.add_argument( '--OCR_dir', type=str)
    parser.add_argument( '--output_dir', type=str)

    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        print('creating directory ' + args.output_dir)

    page_index = sorted(clean_names(os.listdir(args.OCR_dir)))
    Parallel(n_jobs=-1)(map(delayed(main), page_index, [args]*len(page_index)))