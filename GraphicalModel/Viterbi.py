import numpy as np

def Viterbi(edges_list,vertices_list):    #a list of edges, a list of vertices
    if len(edges_list)!=len(vertices_list)-1 or len(edges_list)<=0:
        print("input format error for Viterbi algorithm!")
        return None
    #print("Viterbi algorithm start")
    indexes_list=[]
    '''
    info_dict={}    #for debug
    info_dict['probs']=[]
    info_dict['edges']=[]
    info_dict['vertices']=[]
    '''
    pre_probs=np.array(vertices_list.pop(0))
    #pre_probs[pre_probs<np.max(pre_probs)]=0
    for i in range(len(vertices_list)):
        #if i%100==0:
        #    print(i)
        #import pdb;pdb.set_trace()
        cur_edges=np.array(edges_list[i])
        cur_vertices=np.array(vertices_list[i])     #(Dim_pre,Dim_cur)

        #calculate the prob of all states in current vertices
        probs=cur_vertices*cur_edges                #(Dim_pre,Dim_cur)
        probs=probs.T*pre_probs                     #(Dim_cur,Dim_pre)
        #the pre_vertices that can gives the highest prob
        indexes_list.append(np.argmax(probs,1))     #Dim_cur
        #highest prob in current vertices
        probs=np.max(probs,1)                       #Dim_cur

        norm=np.linalg.norm(probs)

        if norm:
            pre_probs=probs/np.linalg.norm(probs)       #normalization, or the prob may goes to 0
        else:
            probs=pre_probs
            indexes_list.pop()
            print('warning: in Viterbi decoding all the probabilities are 0, tracking ends at frame '+str(i))
            break
        '''
        #info_dict['edges'].append(cur_edges)
        #info_dict['vertices'].append(cur_vertices)
        #info_dict['probs'].append(probs)
        '''
    #backtracking
    index=np.argmax(probs)
    indexes=[index]
    while indexes_list:
        index=indexes_list.pop()[index]
        indexes.append(index)
    #print("Viterbi algorithm finish")
    return indexes[::-1]

def test():
    '''
    v1 = [1, 2]
    v2 = [8, 7]
    e1 = [[1, 2], [3, 4]]
    print(Viterbi([e1], [v1, v2]))

    v1=[1,2]
    v2=[2,1]
    e1=[[1,1],[0,2]]
    print(Viterbi([e1],[v1,v2]))
    '''
    v1 = [1, 2]
    v2 = [2, 1]
    v3= [1,3]
    e1 = [[1, 1], [0, 2]]
    e2=[[1,2],[2,1]]
    print(Viterbi([e1,e2], [v1, v2,v3]))
    import pdb;pdb.set_trace()



if __name__=='__main__':
    test()
