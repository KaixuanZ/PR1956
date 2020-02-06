Due to copy right issue, we public images of PR1954 instead of PR1956 (their data and method are nearly the same)

This project is mainly based on our NeurIPS 2019 workshop paper [Information Extraction from Text Region with Complex Tabular Structure](https://openreview.net/pdf?id=Hkx0zpccLr)

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

### `Introduction`

Please look at the second section of our [paper](https://openreview.net/pdf?id=Hkx0zpccLr). Notice that PR1954 doesn't have class `Table`.

|   |  |
| ------------- | ------------- |
| Number of Raw Scans | 684 |
| Page Bounding Box  | Included  |
| ROI Bounding Box   | Included  |
| Column Bounding Box   | Included  |
| Row Bounding Box   | Included  |
| Row Classification   | Included  |

### `Download`

[AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html) is required for downloading data from AWS.

[Code](DownloadPR1954.sh) for downloading all the data of PR1954.

[Code]() for downloading the model and train&test set of PR1954. (will update soon)

### `AWS S3 Directory Structure`

|  Data |  AWS S3 Path|
| ------------- | ------------- |
| Raw Image | s3://harvardaha-raw-data/personnel-records/1954/scans/firm/ |
| Labeled Data | s3://harvardaha-results/personnel-records/1954/labeled_data/ |
| Segmentation Results | s3://harvardaha-results/personnel-records/1954/seg/firm/ |
| CNN Output | s3://harvardaha-results/personnel-records/1954/prob/firm/ |
| CRF Output | s3://harvardaha-results/personnel-records/1954/cls/CRF/firm/ |

### `Classification Accuracy`
| Book  | Method | Accuracy |
| -------- | ----- | ------------- |
| PR1954 | CNN | 98.4% |
| PR1954 | CNN+CRF | 99.7% |
| PR1956 | CNN | 95.6% |
| PR1956 | CNN+CRF | 96.8% |

# Environment

### `Anaconda`

Using anaconda to create a virtual environment with the command below:

`conda env create -f environment.yml`

# Demo

If you have download the whole dataset and want to reproduce the results, please read the files in `Documents/` and run correspondent shell one by one. Remember to modify the input/output path.

If you have download the whole dataset and want to visualize the results, run correspondent shell in `Visualization/`. Remember to modify the input/output path.

A demo [code](demo/download&visualization.sh) which downloads one sample data and visualizes results.

A demo [code]() which downloads the model and test the model with test data.  (will update)