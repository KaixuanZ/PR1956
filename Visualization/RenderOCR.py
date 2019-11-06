import argparse
from enum import Enum
import io
import os
import numpy as np
import cv2
import json
import sys
sys.path.append('../')
import Rect
from google.protobuf.json_format import Parse
from google.cloud.vision_v1.proto import image_annotator_pb2
from joblib import Parallel, delayed

class box(object):
    def __init__(self, bound, confidence):
        self.bound = bound  #coordinate of the bounding box
        self.confidence = confidence    #


class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


def draw_boxes(image, boxes):
    """Draw a border around the image using the hints in the vector list."""
    for box in boxes:
        if box.confidence > 0.8:
            color = [34, 139, 34]
        elif box.confidence > 0.5:
            color = [0, 140, 255]
        else:
            color = [0, 0, 255]

        box = np.array([[box.bound.vertices[0].x, box.bound.vertices[0].y],
                        [box.bound.vertices[1].x, box.bound.vertices[1].y],
                        [box.bound.vertices[2].x, box.bound.vertices[2].y],
                        [box.bound.vertices[3].x, box.bound.vertices[3].y]])

        cv2.drawContours(image, [box], 0, color, 3)
    return image

def GetImgFilename(string):
    book, p , _ = string.split('.')[0].split('_')
    p = p[0] + str(int(p[1:]))
    return book + '_' + p + '.png'

def get_document_boxes(ocr_file, feature):
    """Returns document bounds given an image."""
    boxes = []

    with open(ocr_file) as json_file:
        data = json.load(json_file)
        parsed = image_annotator_pb2.AnnotateImageResponse()
        Parse(data, parsed)
        # import pdb; pdb.set_trace()

    #texts = parsed.text_annotations
    #for text in texts:
    #    print('\n"{}"'.format(text.description))
    document = parsed.full_text_annotation

    # Collect specified feature bounds by enumerating all document features
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if (feature == FeatureType.SYMBOL):
                            boxes.append(box(symbol.bounding_box, symbol.confidence))

                    if (feature == FeatureType.WORD):
                        boxes.append(box(word.bounding_box, word.confidence))
    return boxes


def render_doc_text(page, args):
    gcv_outputs=sorted(clean_names(os.listdir(os.path.join(args.gcv_dir,page))))

    for i in range(len(gcv_outputs)):

        image = cv2.imread(os.path.join(args.img_path, page,gcv_outputs[i].split('.')[0]+'.png'))
        boxes = get_document_boxes(os.path.join(args.gcv_dir,page,gcv_outputs[i]), FeatureType.SYMBOL)
        draw_boxes(image, boxes)

        output_dir=os.path.join(args.output_dir,page)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
            print('creating directory ' + output_dir)
        path=os.path.join(output_dir,gcv_outputs[i].split('.')[0]+'.png')

        cv2.imwrite(path,image)
        print("saving visualization results to "+path)
    #import pdb;pdb.set_trace()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help='The original scanned image.')
    parser.add_argument('--gcv_dir')
    parser.add_argument('--output_dir', help='Optional output file')
    args = parser.parse_args()

    clean_names = lambda x: [i for i in x if i[0] != '.']
    pages=sorted(clean_names(os.listdir(args.img_path)))

    Parallel(n_jobs=1)(map(delayed(render_doc_text), pages, [args] * len(pages)))
