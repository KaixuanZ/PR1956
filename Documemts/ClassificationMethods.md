# Classification Pipeline

CNN classification ==> Linear Chain CRF ==> Dynamic String Warping (optional)

## Convotional Neural Net
### CNN Model
MobileNet with 80*800 input size
 
### Implementation
/CNN/fintune_models.py and /CNN/finetune.sh
 
### Tips
Transfer Learning, Data Augmentation

## Linear Chain CRF

### Model

Input: observation sequence X 

Output: classification sequence Y

Emission Score U(x_k, y_k): how likely is y_k given observation x_k

Transition Score T(y_k, y_{k+1}): how likely is class y_k followed by class y_{k+1}

Maximize \prod U_i*T_i
 
### Implementation
/GraphicalModel/CRF.py and /GraphicalModel/CRF.sh (use [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/) package)

## Dynamic String Warping

### Model
Target String: T (block structure of a company)

Source String: S (classification results after Linear Chain CRF)

Cost: number of changed classification results when T and S are matched
 
### Algorithm
Dynamic programming

    if T[i]==S[j]:
        DSW[i,j]=min(DSW[i-1,y-1],DSW[i,y-1])
    else:
        DSW[i,j]=min(DSW[i-1,y-1],DSW[i,y-1])+1
 
### Implementation
/Postprocessing/DynamicStringWarping.py
 
### Tips

Dynamic string warping (haven't been applied to PR1956) is very similar to edit distance.

If in one book there are multiple block structures, we can define multiple target string and choose the results with minimum cost

This method can be applied for fixing wrong classification (when cost is very small), or as a retrieval method to find missing companies (cost would be large)