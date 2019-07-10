import argparse
import os
import json
import csv
import Viterbi
import numpy as np

DEFAULT,MANUAL,AUTO=0,1,2

class Graph(object):
    def __init__(self):
        self.Nodes=[]
        self.Edges=None

    def AddNodes(self,Nodes):
        Nodes=np.array(Nodes)
        Nodes/=np.sum(Nodes)        #check the sum of prob should be 1
        self.Nodes.append(Nodes.tolist())

    def SetEdges(self,TransMat):
        self.Edges=[]
        for i in range(len(TransMat)):
            TransMat[i] = [ele / sum(TransMat[i]) for ele in TransMat[i]]
        self.Edges=TransMat

    def Decode(self):
        if self.Nodes==[] or self.Edges==None:
            print("No graph for decoding")
            return []
        else:
            self.cls=[]
            self.cls=Viterbi.Viterbi([self.Edges]*(len(self.Nodes)-1),self.Nodes)
            return self.cls

    def CNNClassification(self):
        self.cls=[]
        for prob in self.Nodes:
            self.cls.append(prob.index(max(prob)))
        return self.cls

def GetMappingDict(f=0):    # f=0: cls2GT ; f=1: GT2cls
    with open('IdNameMap_pr1954.json') as jsonfile:
        Id2Name_cls = json.load(jsonfile)
    #print(Id2Name_cls)
    with open('label_pr1954.csv') as csv_file:
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

def TestAcc(graph):
    cls2GT=GetMappingDict() # mapping from classification result to ground truth
    GT = GetGroundTruth('../Preprocessing/testset_pr1954.csv')

    cls,acc = graph.CNNClassification(),0
    for i in range(len(cls)):
        if cls2GT[cls[i]]==GT[i]:
            acc+=1
    print('classification accuracy of neural network is : ',acc/len(cls))

    cls,acc = graph.Decode(),0
    for i in range(len(cls)):
        if cls2GT[cls[i]] == GT[i]:
            acc += 1
    print('classification accuracy after applying graphcial model is : ', acc/len(cls))

def GetGroundTruth(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        GroundTruth=[]
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            elif row[0] is not '':
                GroundTruth.append(int(row[0]))
            else:
                return GroundTruth
        return GroundTruth

def EstTransMat(method):
    #transition matrix labeled by human knowledge. array[i][j]=1 : transition from class i to class j is possible
    '''
    # matrix for teikoku 1957
    Dim=9
    manual = [[1, 1, 0, 1, 0, 0, 0, 1, 0],  # 0
             [0, 0, 0, 1, 0, 0, 0, 1, 0],  # 1
             [1, 1, 1, 1, 1, 1, 1, 1, 0],  # 2
             [1, 1, 1, 1, 1, 1, 1, 1, 1],  # 3
             [1, 1, 1, 1, 1, 1, 1, 1, 0],  # 4
             [0, 0, 1, 1, 1, 1, 0, 0, 0],  # 5
             [0, 0, 1, 1, 0, 0, 0, 0, 0],  # 6
             [0, 0, 0, 1, 0, 0, 0, 1, 1],  # 7
             [0, 0, 0, 1, 1, 0, 0, 0, 1], ]  # 8
    '''

    Dim=7
    manual = [[1, 1, 0, 1, 0, 1, 1],  # 0
             [1, 1, 1, 1, 1, 1, 1],  # 1
             [1, 1, 1, 1, 0, 0, 0],  # 2
             [1, 1, 1, 1, 1, 1, 1],  # 3
             [0, 1, 1, 1, 1, 0, 0],  # 4
             [0, 1, 0, 1, 1, 1, 1],  # 5
             [0, 1, 0, 1, 1, 1, 1], ]  # 6

    if method==DEFAULT:
        return [[1/Dim]*Dim]*Dim
    elif method==MANUAL:
        for i in range(len(manual)):
            manual[i]=[ele/sum(manual[i]) for ele in manual[i]]
        return manual
    elif method==AUTO:
        count=np.ones([Dim,Dim])
        labels=GetGroundTruth('../Preprocessing/trainset_pr1954.csv')
        GT2cls=GetMappingDict(1)
        for i in range(len(labels)-1):
            count[GT2cls[labels[i]]][GT2cls[labels[i+1]]]+=1
        import pdb;
        pdb.set_trace()
        return count.tolist()
        #return np.multiply(count, np.array(manual)).tolist()
    print("input error for function EstTransMat()")
    return None

def main(path):
    clean_names = lambda x: [i for i in x if i[0] != '.']

    graph=Graph()
    #get values on nodes
    for file in sorted(clean_names(os.listdir(path))):
        for jsonfile in sorted(clean_names(os.listdir(os.path.join(path,file)))):
            with open(os.path.join(path,file,jsonfile)) as inputfile:
                prob = json.load(inputfile)
                graph.AddNodes([*prob.values()])
    #get values on edges
    graph.SetEdges(EstTransMat(AUTO))

    TestAcc(graph)

    #import pdb;pdb.set_trace()

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Graphical model for improving performance')
    parser.add_argument( '--inputpath', type=str)

    args = parser.parse_args()

    main(args.inputpath)