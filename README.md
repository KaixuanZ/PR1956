Due to copy right issue, we public images of PR1954 instead of PR1956 (their data and method are nearly the same)

This project is mainly based on our NeurIPS 2019 workshop paper [Information Extraction from Text Regionwith Complex Tabular Structure](https://openreview.net/pdf?id=Hkx0zpccLr)

# Repo Structure

### `Documents/`

Files explaining the methods of [preprocessing](Documemts/PreprocessingMethods.md) and [classification](Documemts/ClassificationMethods.md)

### `Visualization/`

Code for visualization results

### `OCR/`

Code for using Google Cloud Vision API

### `Preprocessing/`

Code for preprocessing pipeline

### `CNN/`, `GraphicalModel/`, and `Postprocessing/`

Code for classification pipeline

# Dataset

## Introduction
Personnel Record 1954 (PR1956) contains
number of images, number of label, category, etc
 
## Download

[Code](DownloadPR1954.sh) for downloading PR1954. [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html) is required to download PR1954.

## AWS S3 Directory Structure

### `Raw Image`
AWS S3 path: `s3://harvardaha-raw-data/personnel-records/1954/`

### `Labeled Data`
AWS S3 path: `s3://harvardaha-results/personnel-records/1954/labeled_data/`

### `Image Segmentation Results`
AWS S3 path: `s3://harvardaha-results/personnel-records/1954/seg/`

### `Classification Results`
CNN probability output: `s3://harvardaha-results/personnel-records/1954/prob/`

CRF output: `s3://harvardaha-results/personnel-records/1954/cls/CRF/`

# Demo (will update soon)

A demo [code]() which visualizes row segmentation result.

A demo [code]() which visualizes row classification result.