####################################################################################
## PROBLEM2: EIGENFACES
## Given the list of faces in the ./data/lfw2_subset/ dataset, we want to learn 
## an orthonormal basis for the given data. This is learned by obtaining the 
## eigenvectors of the inut images This technique is generally employed for 
## various purposes such as recognition, classification, compression.
## ref: https://www.youtube.com/watch?v=kw9R0nD69OU
## ref: https://en.wikipedia.org/wiki/Eigenface
##
##
## input: directory of faces in ./data/lfw2_subset/
## functions for reading images as vectors is provided
##
##
## your task: fill the following functions:
## extract_mean_stdd_faces
## normalize_faces
## compute_covariance_matrix
## compute_eigval_eigvec
##
##
## output: rmse of the test image preserving 80% energy in eigenvalues
## (lower the better)
##
##
## NOTE: all required modules are imported. DO NOT import new modules.
## NOTE: references are given intline
## tested on Ubuntu14.04, 22Oct2017, Abhilash Srikantha
####################################################################################

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc                              # doesn't read if only scipy is imported
import os
import time

def readImages(dirName,refSize,fExt):
    imFeatures = None
    for root, dirs, files in os.walk(dirName):
        for file in files:
            if file.endswith(fExt):
                img = np.asarray(Image.open(os.path.join(root,file)))
                img = scipy.misc.imresize(img,(refSize[0],refSize[1]))
                img = np.reshape(img,(refSize[0]*refSize[1],1))
                if imFeatures is None:
                    imFeatures = img
                else:
                    imFeatures = np.hstack((imFeatures,img))
                #img = np.reshape(img,(refSize[0],refSize[1]))
                #print("{}".format(img.shape))
                #imgplot = plt.imshow(img)
                #plt.show()
    imFeatTrain = imFeatures[:,:-1]
    imFeatTest  = imFeatures[:,imFeatures.shape[1]-1]
    return imFeatTrain, imFeatTest

def extract_mean_stdd_faces(featFaces):
    mean=np.mean(featFaces,axis=1,keepdims= True)
    stdd=np.std(featFaces,axis=1,keepdims=True)
    #print(mean)
    return mean,stdd

def normalize_faces(featFaces, meanFaces, stddFaces):
    std=featFaces-meanFaces
    norm=np.divide(std,stddFaces)
    return norm

def compute_covariance_matrix(normFaces):
    cov=np.cov(normFaces)
    return cov

def compute_eigval_eigvec(covrFaces):
    a,b=np.linalg.eig(covrFaces)
    return a,b

def show_eigvec(eigvec, cumEigval, refSize, energyTh):
    for idx in range(len(cumEigval)):
        if(cumEigval[idx] < energyTh):
            img = np.reshape(eigvec[:,idx],(refSize[0],refSize[1]))
            print("eigenvector: {} cumEnergy: {} of shape: {}".format(idx, cumEigval[idx], img.shape))
            #imgplot = plt.imshow(img)
            #plt.show()
        else:
            break

def reconstruct_test(featTest, meanFaces, stddFaces, eigvec, numSignificantEigval):
    # projection
    feat = np.expand_dims(featTest,1)- np.expand_dims(meanFaces,1)
    norm = feat / np.expand_dims(stddFaces,1)
    weights = np.inner(np.transpose(eigvec[:,0:numSignificantEigval-1]), np.transpose(norm))
    # reconstruction
    recon = 0*np.squeeze(feat)
    for idx,w in enumerate(weights):
        recon += w[0]*eigvec[:,idx]
    # rmse 
    diff = recon - np.squeeze(norm)
    rmse = np.sqrt(np.inner(np.transpose(diff) , diff) / len(recon))
    return rmse

opts = {'dirName': './data/lfw2_subset',
        'refSize' : [60,60],
        'fExt':'.jpg',
        'energyTh' : 0.80,
        'eps' : 1e-10,
        'inf' : 1e10}


if __name__ == "__main__":

    # time stamp
    start = time.time()

    try:
        # extract features of all faces
        featFaces, featTest = readImages(opts['dirName'],opts['refSize'],opts['fExt'])
        print("featFaces: {}, featTest {}".format(featFaces.shape, featTest.shape))
        
        
        # extract mean face
        meanFaces, stddFaces = extract_mean_stdd_faces(featFaces)
        print("meanFaces: {}, stddFaces: {}".format(meanFaces.shape, stddFaces.shape))
        
        # normalize faces
        # ref: https://stats.stackexchange.com/questions/69157/why-do-we-need-to-normalize-data-before-principal-component-analysis-pca
        # ref: https://stackoverflow.com/questions/23047235/matlab-how-to-normalize-image-to-zero-and-unit-variance
        normFaces = normalize_faces(featFaces, meanFaces, stddFaces)
        print("normFaces: {}".format(normFaces.shape))
            
        # covariance matrix
        covrFaces = compute_covariance_matrix(normFaces) + opts['eps']
        print("covrFaces: {}".format(covrFaces.shape))
        
        # eigenvalues and eigenvectors
        eigval, eigvec = compute_eigval_eigvec(covrFaces)
        print("eigval: {} eigvec: {}".format(eigval.shape, eigvec.shape))
        
        # find number of eigvenvalues cumulatively smaller than energhTh
        cumEigval = np.cumsum(eigval / sum(eigval))
        numSignificantEigval = next(i for i,v in enumerate(cumEigval) if v > opts['energyTh'])
        
        # show top 90% eigenvectors
        # call this function to visualize eigenvectors
        show_eigvec(eigvec, cumEigval, opts['refSize'],opts['energyTh'])
        
        # reconstruct test image
        rmse = reconstruct_test(featTest, meanFaces, stddFaces, eigvec, numSignificantEigval)
        print('#eigval preserving {}% of energy: {}'.format(100*opts['energyTh'],numSignificantEigval))
    except:
        rmse = opts['inf']

    # final output
    print('time elapsed: {}'.format(time.time() - start))
    print('rmse on compressed test image: {} (lower the better)'.format(rmse))
