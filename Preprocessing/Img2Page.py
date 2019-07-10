import os
import numpy as np
import cv2


def MahalonobisDistance(x, mean, cov):
    # definition of M-distance, not used in this file
    v = x - mean
    dis = np.dot(v, np.linalg.inv(cov))
    dis = np.dot(dis, v.T)
    return dis ** 0.5


def SegByMahalonobisDistance(matrix, mean, cov, thr):
    # fast implementation
    cov_inv = np.linalg.inv(cov)
    mat1 = matrix[:, :, 0] - mean[0]
    mat2 = matrix[:, :, 1] - mean[1]
    dis = np.multiply(mat1, mat1) * cov_inv[0, 0] + np.multiply(mat2, mat2) * cov_inv[1, 1] + np.multiply(mat1, mat2) * \
          cov_inv[0, 1] * 2
    return dis < thr ** 2  # mask for segmentation


def CropRect(img, rect):
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # get width and height of the detected rectangle
    width, height = int(rect[1][0]), int(rect[1][1])

    src_pts = box.astype("float32")
    # corrdinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped


def TrainGaussian(file):
    img = cv2.imread(file)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img_hsv)

    data = np.concatenate((h.reshape(1, -1), s.reshape(1, -1)), axis=0)

    # parameters to estimate
    mean = np.mean(data, axis=1)
    cov = np.cov(data)
    return mean, cov


def DecodeFilename(filename):
    filename = filename.split('.')[0]
    book, f, n = filename.split('_')
    f = f[0] + f[1:].zfill(4)
    n = 'n' + n
    return book, f, n


file = '1.tif'
mean, cov = TrainGaussian(file)
thr = 2.25

inputdir = '../testFileGateway/1956/scans/parsed'
outputdir = '../testFileGateway/1956/tmp'
clean_names = lambda x: [i for i in x if i[0] != '.']

target_names = os.listdir(inputdir)
target_names = clean_names(target_names)

for target_name in target_names:
    print("processing "+target_name)

    book, f, n = DecodeFilename(target_name)

    img = cv2.imread(os.path.join(inputdir, target_name))

    img_downsample = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(img)))
    k = 2 ** 3
    img_rgb = cv2.cvtColor(img_downsample, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    mask = SegByMahalonobisDistance(img_hsv[:, :, 0:2], mean, cov, thr)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

    ret, labels = cv2.connectedComponents(mask.astype(np.uint8))
    size, label = 0, 0
    # find the largest region
    for i in range(1, ret):
        if np.sum((labels == i).astype(int)) > size:
            size = np.sum((labels == i).astype(int))
            label = i
    # fit a rect
    _, cnts, _ = cv2.findContours((labels == label).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(cnts[0] * k)

    # crop the rect (pages)
    warped = CropRect(img, rect)

    # seg pages to page if necessary
    if warped.shape[0] * warped.shape[1] > 0.5 * img.shape[0] * img.shape[1]:
        index = int(warped.shape[1] / 2)
        page1 = warped[:, 0:index, :]
        cv2.imwrite(os.path.join(outputdir, f + '_' + n + '_' + 'p0.png'), page1)
        page2 = warped[:, index:, :]
        cv2.imwrite(os.path.join(outputdir, f + '_' + n + '_' + 'p1.png'), page2)
    else:
        cv2.imwrite(os.path.join(outputdir, f + '_' + n + '_' + 'p0.png'), warped)
