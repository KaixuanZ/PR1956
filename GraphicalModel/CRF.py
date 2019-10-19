import pandas as pd
import json
import os
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from collections import Counter
import argparse
from joblib import Parallel, delayed

def GetData(csvfile,label_dict,args):
    X,y,prob={},{},{}
    df = pd.read_csv(csvfile)
    df = df.dropna(thresh=1)

    for index, row in df.iterrows():
        if row['File'].astype(int) in X.keys():
            col_num,row_num=str(int(row['Col'])),str(int(row['Row'])).zfill(3)
            X[row['File'].astype(int)].append(prob[row['File'].astype(int)][col_num][row_num])
            y[row['File'].astype(int)].append(label_dict[str(int(row['cls']))])
        else:
            book,file,subfile,page,col_num,row_num='pr1956','f'+str(int(row['File'])).zfill(4),str(int(row['Image'])),str(int(row['Side'])),str(int(row['Col'])),str(int(row['Row'])).zfill(3)
            prob_filename=os.path.join(args.probpath,'..','firm','_'.join([book,file,subfile,page])+'.json')
            with open(prob_filename) as json_file:
                prob[row['File'].astype(int)]=json.load(json_file)
            X[row['File'].astype(int)]=[prob[row['File'].astype(int)][col_num][row_num]]
            y[row['File'].astype(int)]=[label_dict[str(int(row['cls']))]]
    #import pdb;pdb.set_trace()
    return [*X.values()],[*y.values()]

def test(args):
    df = pd.read_csv(os.path.join(args.labelpath,'Id2Name_label.csv'))
    df = df.dropna(thresh=1)
    label_dict={}
    for index, row in df.iterrows():
        label_dict[str(int(row['label']))]=row['name']

    #read in data
    X_train,y_train=GetData(os.path.join(args.labelpath,'trainset_pr1956.csv'),label_dict,args)
    X_test,y_test=GetData(os.path.join(args.labelpath,'testset_pr1956.csv'),label_dict,args)
    #crf
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=False
    )

    #import pdb; pdb.set_trace()

    #train crf
    crf.fit(X_train, y_train)

    #test crf
    y_pred = crf.predict(X_test)
    print(metrics.flat_classification_report(y_test, y_pred, labels = [*label_dict.values()]))

    def print_transitions(trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
    print("\n Transitions:")
    print_transitions(Counter(crf.transition_features_).most_common())

    return crf,label_dict

def main(probfilename,args,crf):
    print("processing "+probfilename)
    with open(os.path.join(args.probpath,probfilename)) as json_file:
        prob = json.load(json_file)

    #inference
    X=[]
    for key in prob.keys():
        X+=[*prob[key].values()]
    y_pred = crf.predict([X])[0]

    #import pdb;pdb.set_trace()
    with open(os.path.join(args.outputpath,probfilename), 'w') as outputfile:
        json.dump(y_pred, outputfile)

    return y_pred

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Output Prediction Results')
    parser.add_argument( '--probpath', type=str)
    parser.add_argument( '--outputpath', type=str)
    parser.add_argument( '--labelpath', type=str)

    args = parser.parse_args()

    clean_names = lambda x: [i for i in x if i[0] != '.']
    probs = sorted(clean_names(os.listdir(args.probpath)))

    crf,label_dict=test(args)

    Parallel(n_jobs=-1)(map(delayed(main), probs, [args] * len(probs), [crf] * len(probs)))

