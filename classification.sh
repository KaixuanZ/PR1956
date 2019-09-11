#!/usr/bin/env bash

#preparing dataset
cd Preprocessing

    ./Img2Page.sh
    ./RemoveAdPage.sh
    ./Page2ROI.sh
    ./ROI2Col.sh
    ./RemoveMinistry.sh
    ./Col2Row.sh
    ./GenerateDataset.sh

cd ..

#ocr each col
cd OCR
    ./ColOCR
cd ..

#train and test the neural net
cd CNN

    #finetune the pretrained model with a few data from PR1956
    ./finetune.sh

    #test performance on test data of PR1956
    ./test.sh

cd ..

#improve the performance
cd GraphicalModel

    #save predicted probability of each class as json
    ./output_prob.sh

    #improve the classification performance by applying graphical model
    ./GraphicalModel.sh

cd ..

./CombineRes.sh