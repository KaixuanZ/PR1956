import argparse
from enum import Enum
import os
import json
import cv2
import numpy as np
from google.protobuf.json_format import Parse
from google.cloud.vision_v1.proto import image_annotator_pb2

class box(object):
    def __init__(self, bound, confidence):
        self.bound = bound
        self.confidence = confidence
        self.rect = self.BoundToRect()
        self.row_index = None

    def BoundToRect(self):
        box = np.array([[self.bound.vertices[0].x, self.bound.vertices[0].y],
                        [self.bound.vertices[1].x, self.bound.vertices[1].y],
                        [self.bound.vertices[2].x, self.bound.vertices[2].y],
                        [self.bound.vertices[3].x, self.bound.vertices[3].y]])
        return cv2.minAreaRect(box)


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
            color = [0, 128, 0]  # green
        elif box.confidence > 0.5:
            color = [0, 165, 255]  # orange
        else:
            color = [0, 0, 255]  # red

        pts = cv2.boxPoints(box.rect).astype(np.int32)
        cv2.polylines(image, [pts], True, color, 2)
    return image


def get_document_boxes(ocr_file, feature):
    """Returns document bounds given an image."""
    boxes = []

    with open(ocr_file) as json_file:
        data = json.load(json_file)
        parsed = image_annotator_pb2.AnnotateImageResponse()
        Parse(data, parsed)

    texts = parsed.text_annotations
    for text in texts:
        print('\n"{}"'.format(text.description))
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
                if (feature == FeatureType.PARA):
                    boxes.append(box(paragraph.bounding_box, paragraph.confidence))

        # The list `bounds` contains the coordinates of the bounding boxes.
    return boxes


def render_doc_text(image_file, row_rect_dir, ocr_file, fileout=None):
    row_rects = os.listdir(row_rect_dir)
    clean_names = lambda x: [i for i in x if i[0] != '.']
    row_rects = sorted(clean_names(row_rects))
    dict = {}

    for row_rect in row_rects:
        key = int(row_rect.split('_')[-2])
        if key in dict.keys():
            dict[key].append(os.path.join(row_rect_dir, row_rect))
        else:
            dict[key] = [os.path.join(row_rect_dir, row_rect)]

    img = cv2.imread(image_file)
    boxes = get_document_boxes(ocr_file, FeatureType.WORD)
    draw_boxes(img, boxes)

    if fileout:
        cv2.imwrite(fileout, img)
    else:
        cv2.imshow('tmp', img)
        cv2.waitKey(1)
        import pdb;pdb.set_trace()