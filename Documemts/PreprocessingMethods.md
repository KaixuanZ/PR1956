# Preprocessing Pipeline

Original Scanned Image ==> [Page Detection](##Page-Detection) ==> [ROI Detection](##Region of Interest) ==> Column Segmentation ==> Row Segmentation

## Page Detection

### Model
Input: color image with size H\*W\*3

Output: rect(s) of detected page(s)
For each pixel computing P(Y|X) where X is the information of a pixel, and Y is the classification results (page / not page) of that pixel
 
### Algorithm
RGB ==> HSV

P(Y|X) = P(Y|H,S,V) = P(Y|H,S) * P(Y|V) (assume H,S are independent with V)

P(Y|H,S) is a multi-variate Gaussian learned from data (crop a small area of page), P(Y|H) is simply a thresholding method.

Y=1 (classified as page) if P(Y|X) > threshold 

For the binary mask, use CCL to find the one/two largest region(s) and then compute the minAreaRect(s), which is(are) the page(s).
 
### Implementation
/Preprocessing/Img2Page.py and /Preprocessing/Img2Page.sh
 
### Tips
Randomly pick some pages and try the values of two thresholds several times.

## Region of Interest (ROI)/Text Region Detection

### Model
Input: original scan and a detected page rect

Output: a rect of ROI

Find the vertical lines and infer the ROI
 
### Algorithm
Binarize the original scan and crop the page rect

Use CCL to detect n vertical lines

Expend the width of detected ROI rect by (n+1)/(n-1)
 
### Implementation
/Preprocessing/Page2ROI.py and /Preprocessing/Page2ROI.sh
 
### Tips
A bit special when n=1

## Column Segmentation

### Model
Input: original scan and a detected ROI

Output: rects of columns in correspondent ROI
 
### Algorithm
Binarize the image and crop ROI area

Sum over height to get a 1*W signal

Remove noise by applying a Low Pass Filter (median filter also works)

Search for optimal segmentation point(s) on this 1*W signal
 
### Implementation
/Preprocessing/ROI2Col.py and /Preprocessing/ROI2Col.sh

## Row Segmentation

### Model
Input: original scan and a detected column rect

Output: rects of row
 
### Algorithm
Binarize the original scan and crop the column rect

Segment column by a naive method and get a set of [row_rect]

    for row_rect in [row_rect]:
            if row_rect.height > threshold:
                    rotate row_rect a little bit and segment it again with the naive method

naive method is similar to the idea of column segmentation

threshold is an adaptive value equals to the median of [row_rect.height]
 
### Implementation
/Preprocessing/Col2Row.py and /Preprocessing/Col2Row.sh