import argparse
from enum import Enum
import io
import os
from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw


client = vision.ImageAnnotatorClient()

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


def draw_boxes(image, boxes, color):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    for box in boxes:
        if box.confidence > 0.8:
            color = 'green'
        elif box.confidence > 0.5:
            color = 'orange'
        else:
            color = 'red'
        # import pdb; pdb.set_trace()
        draw.polygon([
            box.bound.vertices[0].x, box.bound.vertices[0].y,
            box.bound.vertices[1].x, box.bound.vertices[1].y,
            box.bound.vertices[2].x, box.bound.vertices[2].y,
            box.bound.vertices[3].x, box.bound.vertices[3].y], None, color)
    return image


def get_document_boxes(image_file, feature):
    """Returns document bounds given an image."""
    client = vision.ImageAnnotatorClient()

    boxes = []

    with io.open(image_file, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    image_context = vision.types.ImageContext(language_hints=["ja zh*"])
    response = client.document_text_detection(image=image, image_context=image_context)
    document = response.full_text_annotation
    '''
    texts = response.text_annotations
    print('Texts:')
    for text in texts:
        print('\n"{}"'.format(text.description))
    '''
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


def render_doc_text(filein, fileout):
    image = Image.open(filein)
    # bounds = get_document_bounds(filein, FeatureType.PAGE)
    # draw_boxes(image, bounds, 'blue')
    # bounds = get_document_bounds(filein, FeatureType.WORD)
    # draw_boxes(image, bounds, 'red')
    boxes = get_document_boxes(filein, FeatureType.SYMBOL)
    draw_boxes(image, boxes, 'red')  # 'yellow'

    if fileout is not 0:
        image.save(fileout)
    else:
        image.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('detect_file', help='The image for text detection.')
    parser.add_argument('-out_file', help='Optional output file', default=0)
    args = parser.parse_args()

    render_doc_text(args.detect_file, args.out_file)

