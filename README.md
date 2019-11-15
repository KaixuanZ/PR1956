(Still updating)

This project is mainly based on our NeurIPS 2019 workshop paper [Information Extraction from Text Regionwith Complex Tabular Structure](https://openreview.net/pdf?id=Hkx0zpccLr)

## Repo Structure

#### `Documents/`

Files explaining the methods of [preprocessing](Documemts/PreprocessingMethods.md) and [classification](Documemts/ClassificationMethods.md)

#### `Visualization/`

Code for visualization results

#### `OCR/`

Code for using Google Cloud Vision API

#### `Preprocessing/`

Code for preprocessing pipeline

#### `CNN/`, `GraphicalModel/`, and `Postprocessing/`

Code for classification pipeline

## Dataset

### Introduction
Personnel Record 1956 (PR1956) contains
number of images, number of label, category, etc
 
### Download

[Code](DownloadPR1956.sh) for downloading PR1956. [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html) is required to download PR1956.

### Directroy Structure
####Raw Image

####Labeled Data

####Image preprocessing results

####Classification results

## Demo

A demo [code]() which visualizes row segmentation result.

A demo [code]() which visualizes row classification result.