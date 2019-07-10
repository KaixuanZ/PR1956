#!/usr/bin/env bash

#preparing dataset
cd Preprocessing

    #./GenerateDataset.sh

cd ..

#train and test the neural net
cd CNN

    #train a model from scratch with a huge amount of data from teikoku1924
    #./train.sh

    #finetune the pretrained model with a few data from teikoku1957
    #./finetune.sh

    #test performance on test data of teikoku1957
    ./test.sh

cd ..

#improve the performance
cd GraphicalModel

    ProbFile='probability'

    #save predicted probability of each class as json
    #./output_prob.sh   --InputPath='testimg'   --OutputPath=$ProbFile

    #improve the classification performance by applying graphical model
    ./GraphicalModel.sh --InputPath=$ProbFile

cd ..