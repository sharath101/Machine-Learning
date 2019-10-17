####################################################################################
## PROBLEM3: 2-CLASS IMAGE CLASSIFICATION
## Given the ETHZ-Shape dataset, we want to learn a classifier between
## two classes. Namely: Mugs and Swans. In this regard, svm and random 
## forest based classification techniques will be investigated.
##
##
## input: directory of faces in ./data/ETHZShapeClasses-V1.2/ 
## functions for reading input features and generating train/test splits are provided.
##
##
## your task: using sklearn, fill the following functions
## train_random_forest
## train_svm
## test_classifier
## NOTE: set the hyperparameters of classifiers appropriately.
##
##
## output: lowest (1-f1score) between svm and rf classifiers  (lower the better)
##
##
## NOTE: all required modules are imported. DO NOT import new modules.
## NOTE: references are given intline
## tested on Ubuntu14.04, 07Aug2018, Abhilash Srikantha
####################################################################################

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy
import os
from sklearn import datasets, svm, metrics, ensemble
import time

def train_test_split(fdata,ldata,trainSize):
    np.random.shuffle(fdata)
    np.random.shuffle(ldata)
    splitLoc = np.floor(trainSize*len(ldata))
    splitLoc = splitLoc.astype(int)
    # create the random split
    ftrain = fdata[0:splitLoc,:]
    ltrain = ldata[0:splitLoc]
    ftest = fdata[splitLoc+1:,:]
    ltest = ldata[splitLoc+1:]
    return ftrain, ftest, ltrain, ltest

def train_random_forest(feat,label):
    pass

def train_svm(feat,label):
    pass
    
def test_classifier(ftest,classifier):
    pass
    
def eval_performance(pred,ltest,classifier):
    return metrics.f1_score(ltest, pred)

def read_data(rdir,cNames,fExt,refSize):
    featData = None
    labelData = None
    for c,cname in enumerate(cNames):
        dirName = rdir + cname
        for root, dirs, files in os.walk(dirName):
            for file in files:
                if(file.endswith(fExt)):
                    # read Image
                    img = np.asarray(Image.open(os.path.join(root,file)))
                    if(len(img.shape)!=3):
                        continue
                    img = scipy.misc.imresize(img,(refSize))
                    # collapse into vector
                    feat = np.reshape(img,(1,np.prod(refSize)))
                    # append to dataset
                    if featData is None:
                        featData = feat
                        labelData = c
                    else:
                        featData = np.vstack((featData,feat))
                        labelData = np.hstack((labelData,c))
                    #print("{}".format(img.shape))
                    #imgplot = plt.imshow(img)
                    #plt.show()
    
    print("feat shape: {}, label shape:{}".format(featData.shape,labelData.shape))
    return featData, labelData

opts = {'rdir': './data/ETHZShapeClasses-V1.2/',
        'classNames' : {'Mugs','Swans'},
        'fExt':'jpg',
        'refSize' : [10,10,3],
        'trainSplit' : 0.7,
                'inf' : 1e10,
        'seed':0}


if __name__ == "__main__":
    

    np.random.seed(opts['seed'])

    # time stamp
    start = time.time()

    # read the data
    feat,label = read_data(opts['rdir'],
                        opts['classNames'],
                        opts['fExt'],
                        opts['refSize'])

    # train test split
    # ref: https://www.youtube.com/watch?v=lSwvUmZCvco
    ftrain,ftest,ltrain,ltest = train_test_split(feat,label,opts['trainSplit'])

    try:
        # train test svm classifier
        # ref: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        # ref: https://www.youtube.com/watch?v=N1vOgolbjSc
        # ref: https://www.youtube.com/watch?v=_PwhiWxHK8o
        classifier_svm = train_svm(ftrain, ltrain)
        predicted = test_classifier(ftest, classifier_svm)
        f1ScoreSVC = eval_performance(predicted,ltest,classifier_svm)
        
        # train test random forest classifier
        # ref: http://scikit-learn.org/stable/modules/ensemble.html
        # ref: https://www.youtube.com/watch?v=3kYujfDgmNk
        classifier_rf = train_random_forest(ftrain, ltrain)
        predicted = test_classifier(ftest, classifier_rf)
        f1ScoreRF = eval_performance(predicted,ltest,classifier_rf)
        
        # print report
        # ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        print("f1 scores: (svm) {} , (rf) {}".format(f1ScoreSVC, f1ScoreRF))
        f1ScoreBest = f1ScoreSVC if(f1ScoreSVC>f1ScoreRF) else f1ScoreRF
        f1ScoreReport = 1-f1ScoreBest
    except:
        f1ScoreReport = opts['inf']
        
    # final output
    print('time elapsed: {}'.format(time.time() - start))
    print("lowest 1-f1Score: {} (lower the better)".format(f1ScoreReport))
