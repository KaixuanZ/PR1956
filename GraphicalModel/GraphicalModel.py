import argparse
import os
import json
import csv
import Viterbi
import numpy as np
import copy

DEFAULT,MANUAL,AUTO=0,1,2

class Graph(object):
    def __init__(self):
        self.Nodes=[]
        self.Edges=None
        self.cls = []
        self.Blanks = []  # index of blank on original sequence

    def RemoveBlank(self):  # for output prob of CNN, "1":"blank"
        self.Blanks=[]
        for i in range(len(self.Nodes) - 1, 0 - 1, -1):
            if self.Nodes[i].index(max(self.Nodes[i])) == 1:
                self.Blanks.append(i)
        for Blank in self.Blanks:
            self.Nodes.pop(Blank)

    def InsertBlank(self):
        # import pdb;pdb.set_trace()
        for i in range(len(self.Blanks) - 1, 0 - 1, -1):
            self.cls.insert(self.Blanks[i], 1)

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

def EstTransMat(labelfile,method):
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
    '''
    manual = [[1, 1, 1, 1, 1, 1, 1],  # 0
              [1, 1, 1, 1, 1, 1, 1],  # 1
              [1, 1, 1, 1, 0, 0, 0],  # 2
              [1, 1, 1, 1, 1, 1, 1],  # 3
              [1, 1, 1, 1, 1, 1, 1],  # 4
              [1, 1, 0, 1, 1, 1, 1],  # 5
              [1, 1, 0, 1, 1, 1, 1], ]  # 6
    '''
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
        labels=GetGroundTruth(labelfile,RemoveBlank=True)
        GT2cls=GetMappingDict(1)
        for i in range(len(labels)-1):
            count[GT2cls[labels[i]]][GT2cls[labels[i+1]]]+=1
        #import pdb;pdb.set_trace()
        count[1,:]=1
        count[:,1]=1
        #return count.tolist()
        return np.multiply(count, np.array(manual)).tolist()
    print("input error for function EstTransMat()")
    return None

def main(inputpath,outputpath,labelfile):
    with open('../../personnel-records/1954/Id2Name_cls.json') as jsonfile:
        Id2Name_cls = json.load(jsonfile)

    clean_names = lambda x: [i for i in x if i[0] != '.']

    graph = Graph()
    # get values on edges
    graph.SetEdges(EstTransMat(labelfile, AUTO))

    for dir in sorted(clean_names(os.listdir(inputpath))):  #one dir includes prob for one page
        print("processing "+dir)
        graph.Nodes=[]
        graph.cls=[]
        cls={}
        for jsonfile in sorted(clean_names(os.listdir(os.path.join(inputpath,dir)))):
            with open(os.path.join(inputpath,dir,jsonfile)) as inputfile:
                prob = json.load(inputfile)
                # get values on nodes
                graph.AddNodes([*prob.values()])
        #output cls
        graph.RemoveBlank()
        graph.Decode()
        graph.InsertBlank()
        cls['id'] =graph.cls
        cls['id'] = [int(i) for i in cls['id']]
        cls['name'] = [Id2Name_cls[str(i)] for i in cls['id']]
        with open(os.path.join(outputpath, dir+'.json'), 'w') as outputfile:
            json.dump(cls, outputfile)


    #import pdb;pdb.set_trace()

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Graphical model for improving performance')
    parser.add_argument( '--inputpath', type=str)
    parser.add_argument( '--outputpath', type=str)
    parser.add_argument( '--labelfile', type=str)

    args = parser.parse_args()

    main(args.inputpath,args.outputpath,args.labelfile)