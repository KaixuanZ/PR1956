import argparse
import os
import json
import csv
import Viterbi
import numpy as np
import copy
from joblib import Parallel, delayed

DEFAULT,MANUAL,AUTO=0,1,2

#use linear chain CRF to improve the classfication accuracy. Output are saved in json format.

class Graph(object):
    def __init__(self):
        self.Nodes=[]
        self.Edges=None
        self.cls = []

    def AddNodes(self,Nodes):
        Nodes=np.array(Nodes)
        #if Nodes[1]>0.8:    #give more confidence on company name
        #    Nodes[1]*=5
        Nodes/=np.sum(Nodes)        #check the sum of prob should be 1
        self.Nodes.append(Nodes.tolist())

    def SetEdges(self,TransMat):
        self.Edges=[]
        for i in range(len(TransMat)):
            TransMat[i] = [ele / sum(TransMat[i]) for ele in TransMat[i]]
        self.Edges=TransMat

    def Decode(self):
        if self.Edges==None:
            print("No graph for decoding")
        elif self.Nodes==[]:
            return []
        elif len(self.Nodes)==1:
            return self.Nodes[0].index(max(self.Nodes[0]))
        else:
            self.cls=[]
            self.cls=Viterbi.Viterbi([self.Edges]*(len(self.Nodes)-1),copy.copy(self.Nodes))
            return self.cls

    def CNNClassification(self):
        for prob in self.Nodes:
            self.cls.append(prob.index(max(prob)))
        return self.cls

def GetMappingDict(id2name_label, id2name_cls, f=0):    # f=0: cls2GT ; f=1: GT2cls
    with open(id2name_cls) as jsonfile:
        Id2Name_cls = json.load(jsonfile)
    #print(Id2Name_cls)
    with open(id2name_label) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        Name2Id_GT={}
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            elif row[0] is not '':
                Name2Id_GT[row[1]]=row[0]

    Dict={}
    for key in Id2Name_cls:
        Dict[int(key)]=int(Name2Id_GT[Id2Name_cls[key]])    #cls2GT
    if f:   #cls2GT ==> GT2cls
        Dict = dict([(value, key) for key, value in Dict.items()])
    return Dict

def GetGroundTruth(path,RemoveBlank=False):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        GroundTruth=[]
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            elif row[0] is not '':
                if RemoveBlank:
                    if row[0] is not '0':
                        GroundTruth.append(int(row[0]))
                else:
                    GroundTruth.append(int(row[0]))
            else:
                return GroundTruth
        return GroundTruth

def EstTransMat(labelfile, id2name_label, id2name_cls, method):
    #transition matrix labeled by human knowledge. array[i][j]=1 : transition from class i to class j is possible
    # matrix for PR1954
    Dim = 5
    manual = [[1, 0, 0, 1, 1],  # 0  address
              [1, 1, 0, 0, 0],  # 1   company
              [0, 1, 1, 0, 0],  # 2   personnel
              [0, 1, 1, 1, 1],  # 3   variable
              [0, 1, 1, 1, 1], ]  # 4 value
    if method==DEFAULT:
        return [[1/Dim]*Dim]*Dim
    elif method==MANUAL:
        for i in range(len(manual)):
            manual[i]=[ele/sum(manual[i]) for ele in manual[i]]
        return manual
    elif method==AUTO:
        count=np.ones([Dim,Dim])
        labels=GetGroundTruth(labelfile,RemoveBlank=False)
        GT2cls=GetMappingDict(id2name_label, id2name_cls, 1)
        for i in range(len(labels)-1):
            count[GT2cls[labels[i]]][GT2cls[labels[i+1]]]+=1
        return np.multiply(count, np.array(manual)).tolist()
    print("input error for function EstTransMat()")
    return None

def main(file,args):
    print("processing "+file)
    with open(args.id2name_cls) as jsonfile:
        Id2Name_cls = json.load(jsonfile)

    graph = Graph()
    # get values on edges
    graph.SetEdges(EstTransMat(args.trainset, args.id2name_label, args.id2name_cls, MANUAL))



    with open(os.path.join(args.inputpath, file)) as jsonfile:
        probs = json.load(jsonfile)

    graph.Nodes=[]
    graph.cls=[]
    cls={}

    for col_num in sorted(probs.keys()):
        for row_num in sorted(probs[col_num].keys()):
            # get values on nodes
            graph.AddNodes([*probs[col_num][row_num].values()])

    #output cls
    graph.Decode()
    cls['id'] =graph.cls
    cls['id'] = [int(i) for i in cls['id']]
    cls['name'] = [Id2Name_cls[str(i)] for i in cls['id']]
    with open(os.path.join(args.outputpath, file), 'w') as outputfile:
        json.dump(cls, outputfile)


    #import pdb;pdb.set_trace()

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Graphical model for improving performance')
    parser.add_argument( '--inputpath', type=str)
    parser.add_argument( '--outputpath', type=str)
    parser.add_argument( '--trainset', type=str)
    parser.add_argument( '--id2name_label', type=str)
    parser.add_argument( '--id2name_cls', type=str)

    args = parser.parse_args()
    clean_names = lambda x: [i for i in x if i[0] != '.']
    files=sorted(clean_names(os.listdir(args.inputpath)))

    Parallel(n_jobs=-1)(map(delayed(main), files, [args] * len(files)))
    #main(args.inputpath,args.outputpath,args.trainset,args.id2name_label,args.id2name_cls)