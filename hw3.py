# -*- coding: utf-8 -*-
"""
HOMEWORK 3:
-----------
Implement CRF training for POS tagging using data from 
the previous assignment, and the same feature set. Use SGD to optimize 
the weights. 
What is your accuracy on the test file when training on the train file?

@author: Md Iftekhar Tanveer (itanveer@cs.rochester.edu)
"""
import numpy as np
from collections import defaultdict

# Reads the list of tags from file
def readalltags(tagsetfile):
    with open(tagsetfile) as f:
        tags = [item.strip() for item in f]
    return tags

# Save the weights 
def saveweights(filename,E,T):
    with open(filename,'w') as f:
        for item in E.items():
            print>>f,'E_'+item[0][0]+'_'+item[0][1]+' '+str(item[1])
        for item in T.items():
            print>>f,'T_'+item[0][0]+'_'+item[0][1]+' '+str(item[1])

# Reads the emission and transition weights from file
# Note: this is used for debug purpose only. I didn't use this
def readwt(weightfile):
    with open(weightfile) as f:
        E = {}
        T = {}
        for aline in f:
            tag,weight = aline.strip().split(' ')
            tag = tag.split('_')
            if tag[0]=='E':
                E[tag[1],tag[2]]=float(weight)
            else:
                T[tag[1],tag[2]]=float(weight)
    return E,T

# Applies dynamic programming to find the best tag sequence
# Will be needed for calculating accuracy
def viterbi(line,E,T,tags):
    wrdlist = line.split(' ')
    x = np.ones((len(tags),len(wrdlist)))*-1*np.inf
    b = np.zeros((len(tags),len(wrdlist)))
    for i,aword in enumerate(wrdlist):
        # As I didn't see any start or end tag in the tagset, I am assuming
        # all the weights for transition from the start tag to any other tag
        # is zero (which is not true in reality).
        # So for the first word, I don't consider the transition prob
        if i==0:
            for tagid,atag in enumerate(tags):
                x[tagid,i] = E.get((atag,aword),-1*np.inf)
                b[tagid,i] = -1 # Means this is the first word
            continue

        # if not the first word, consider both transition and emission prob
        for atagid,atag in enumerate(tags):
            # theoretically, the weights should be -ve inf if a specific
            # pair is not found in the corpus. However, something didn't
            # appear in the corpus doesn't mean that its probability is
            # totally zero. So, I am assigning a small value instead of
            # -ve inf.
            emmval = E.get((atag,aword),-1*1e6) #emission prob
            for atagid_prev,atag_prev in enumerate(tags):
                trval = T.get((atag_prev,atag),-1*1e6)  #transition prob
                total = x[atagid_prev,i-1]+emmval+trval 
                # Debug
#                print 'currtag',atag+'('+str(atagid)+')','prevtag',atag_prev+\
#                '('+str(atagid_prev)+')','i',str(i),'word',aword,\
#                'emm',emmval,'trans',trval,'tot',total                
                if total>x[atagid,i]:
                    x[atagid,i] = total  # Take the maximum logprob
                    b[atagid,i] = atagid_prev # keep a backward pointer
    idx = np.argmax(x[:,-1])
    annot=[]
    # Trace back the sequence using the back pointer
    for idx_ in xrange(np.size(b,axis=1),0,-1):
        annot.append(tags[int(idx)])
        idx = b[idx,idx_-1]
    annot.reverse()
    return wrdlist,annot

# Calculate the accuracy of viterbi over a given test file
def calcaccuracy(file,E,T,tags):
    with open(file) as f:
        totalWords=0.
        countCorrect=0.
        for aline in f:
            data = [item.strip() for index, item in \
            enumerate(aline.strip().split(' ')) if not index==0]
            testline = ' '.join(data[0::2])
            annotGT = data[1::2]
            wrdlst,annt=viterbi(testline,E,T,tags)
            countCorrect=countCorrect+sum([a1==a2 for a1,a2 in zip(annotGT,annt)])
            totalWords=totalWords+len(annotGT)
    return float(countCorrect)/totalWords,countCorrect,totalWords

# Forward algorithm
def buildalpha(words,E,T,tags):
    # build alpha table in log space
    alpha = np.zeros((len(tags),len(words)))
    for t,xt in enumerate(words): # run along the columns
        # for the first column (Base case)
        if t==0:
            for idx_j,j in enumerate(tags): # run along the rows
                alpha[idx_j,t]=np.exp(E[j,xt])   # Only emission for first col
            alpha[:,t]/=np.sum(alpha[:,t])
        else:
            # for rest of the columns
            for idx_j,j in enumerate(tags): # run along the rows
                for idx_i,i in enumerate(tags): # run along the rows of prev. col
                    tr = T.get((i,j),-1*1e6)
                    alpha[idx_j,t]+=np.exp(tr)*alpha[idx_i,t-1]
                # Brought the emmission term outside of the inner sum
                alpha[idx_j,t]*=np.exp(E.get((j,xt),-1*1e6))
            alpha_sums[t]=np.sum(alpha[:,t])
            alpha[:,t]/=alpha_sums[t]
    return alpha,alpha_sums

# Backward Algorithm
def buildbeta(words,alpha_sums,E,T,tags):
    beta = np.zeros((len(tags),len(words)))
    for t in xrange(len(words)-1,-1,-1):
        # last column (base case)
        if t==len(words)-1:
            for idx_j,j in enumerate(tags):
                beta[idx_j,t]=1.0
        else: # rest of the columns
            for idx_i,i in enumerate(tags): # run along the rows
                for idx_j,j in enumerate(tags): #run along the rows of next col
                    em=E.get((j,words[t+1]),-1*1e6)
                    tr=T.get((i,j),-1*1e6)
                    beta[idx_i,t]+=np.exp(tr)*np.exp(em)*beta[idx_j,t+1]
            beta[:,t]/=alpha_sums[t+1]
    return beta

def buildc(alpha,beta,E,T,tags):
    c = np.ones((len(tags),len(tags)))
    return c/np.sum(c)


def splitlines(line):
    words=[]
    tags=[]
    splitted = line.strip().split(' ')
    for i,word in enumerate(splitted[1:]):
        if i%2==0:
            words.append(word)
        else:
            tags.append(word)
    return words,tags


# Main method
def main():
    # Initialize the emission and transition weights as blank hash maps    
    E=defaultdict(float)
    T=defaultdict(float)
    totalIter = 5
    tags = readalltags('./alltags') # read all the tags
    f = open('./train','r')
    # Repeat until a maximum number of iteration
    for iter in xrange(totalIter):
        eta = 1./np.sqrt(iter+1) # Decreasing learning rate
        # Read one line from the training data
        for aline in f:
            words,pos = splitlines(aline)
            # Building alpha and beta using forward-backward
            alpha,alpha_sums = buildalpha(words, E, T, tags)
            beta = buildbeta(words,alpha_sums,E,T,tags)
            c = buildc(alpha,beta,E,T,tags)
            tempE=E.copy()
            tempT=T.copy()
            
            # Scan through the training line
            for t in xrange(len(words)-1):
                # First term in the SGD update
                tempT[pos[t],pos[t+1]]+=eta
                tempE[pos[t],words[t]]+=eta
                # Second term in the SGD update
                tempT[pos[t],pos[t+1]]-=eta*c[tags.index(pos[t]),tags.index(pos[t+1])]
                tempE[pos[t],words[t]]-=eta*c[tags.index(pos[t]),tags.index(pos[t+1])]
            # Last word in the line
            tempT[pos[t+1],'</s>']+=eta
            tempE[pos[t+1],words[t+1]]+=eta
            # Need to work out
#            tempE[pos[t+1],words[t+1]]-=eta*c[tags.index(pos[t]),tags.index(pos[t+1])]
                        
            # Update T and E
            T.update(tempT)
            E.update(tempE)


def debug():
    # Debug the for forward backward
    # read the previous weights
    E,T = readwt('./train.weights')
    tags = readalltags('./alltags') # read all the tags
    words,pos = splitlines('5 Time NN flies VBZ like IN an DT arrow NN')
    alpha,alpha_sums = buildalpha(words, E, T, tags)
    beta = buildbeta(words,alpha_sums,E,T,tags)
    
    pass
    
            
if __name__=='__main__':
    #debug()
    main()
