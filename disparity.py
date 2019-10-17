####################################################################################
## PROBLEM4: DISPARITY MAP GENERATION
## Given the Aloe dataset with two view of a plant from two different views, we 
## want to construct a disparity map. In this regard, sum of absolute differences
## will be used as a metric to compute patch matches. 
##
##
## input: directory of faces in ./data/cones/ 
## functions for reading input are provided.
## skeleton to compute the disparity is provided
##
##
## your task: fill the following functions
## compute_patch_similarity (sum of absolute error metric)
## similarity_to_disparity (patch disparity)
## NOTE: set hyperparameters 'downsample' and 'patchsize' carefully
##
##
## output: meanSSE between computed disparity and groundtruth  (lower the better)
##
## NOTE: see http://mccormickml.com/2014/01/10/stereo-vision-tutorial-part-i/
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

def read_data(root,downsample=10):
    view1 = np.asarray(Image.open(os.path.join(root,'view1.png')))
    view1 = view1[0:-1:downsample,0:-1:downsample,:]
    view2 = np.asarray(Image.open(os.path.join(root,'view5.png')))
    view2 = view2[0:-1:downsample,0:-1:downsample,:]
    gth12 = np.asarray(Image.open(os.path.join(root,'disp1.png')))
    gth12 = gth12[0:-1:downsample,0:-1:downsample]
    gth21 = np.asarray(Image.open(os.path.join(root,'disp5.png')))
    gth21 = gth21[0:-1:downsample,0:-1:downsample]
    print('reading data successful')
    return view1, view2, gth12, gth21

# lower the better
def compute_patch_similarity(ref,tgt):
    pass 

# compute closest patch distance
def similarity_to_disparity(dissimilarity_array, pos):
    pass 
    
def compute_patch_disparity(patch, patchpos, row):
    # sum of absolute error array
    sad_array = np.zeros((row.shape[1],1)) + 1e6
    
    # compute patch dissimilarity
    psize = patch.shape[0]//2
    for cc in range(psize, row.shape[1]-psize):
        tgt = row[:,cc-psize:cc+psize,:]
        sad_array[cc] = compute_patch_similarity(patch,tgt)
        '''plt.subplot(121)
        plt.imshow(patch)
        plt.subplot(122)
        plt.imshow(tgt)
        plt.show()
        print('dist: {}'.format(sad_array[cc]))'''
        
    disparity = similarity_to_disparity(sad_array, patchpos)
    return disparity

def compute_disparity(view1, view2, psize=5):
    disp = np.zeros((view1.shape[0],view1.shape[1]))
    for rr in range(psize,disp.shape[0]-psize):
        # conduct row search as the images are rectified
        # ref: https://en.wikipedia.org/wiki/Image_rectification
        view2row = view2[rr-psize:rr+psize,:,:]
        for cc in range(psize,disp.shape[1]-psize):
            patch = view1[rr-psize:rr+psize, cc-psize:cc+psize, :]            
            disp[rr,cc] = compute_patch_disparity(patch, cc, view2row)
        '''print('row: {} out of {}'.format(rr,disp.shape[0]))
        plt.imshow(disp)
        plt.title('computed disparity map')
        plt.show()'''
    return disp

opts = {'rdir': './data/cones/',
        'halfPatchSize' : 11,
        'downsample': 4,
        'inf' : 1e10}


if __name__ == "__main__":
    # time stamp
    start = time.time()

    # read the data
    view1,view2,gth12,gth21 = read_data(opts['rdir'],opts['downsample'])

    try:
        disp12 = compute_disparity(view1, view2, opts['halfPatchSize'])
        disp21 = compute_disparity(view2, view1, opts['halfPatchSize'])
        sse = np.sum(np.power((disp12-gth12),2)) + np.sum(np.power((disp21-gth21),2))
        sse /= (disp12.shape[0]*disp12.shape[1])
        
        # display results
        plt.subplot(321)
        plt.imshow(view1)
        plt.title('view1')
        plt.subplot(322)
        plt.imshow(view2)
        plt.title('view2')
        plt.subplot(323)
        plt.imshow(gth12)
        plt.title('gth12')
        plt.subplot(324)
        plt.imshow(gth21)
        plt.title('gth21')
        plt.subplot(325)
        plt.imshow(disp12)
        plt.title('computed12')
        plt.subplot(326)
        plt.imshow(disp21)
        plt.title('computed21')
    except:
        sse = opts['inf']
        
    # final output
    print('time elapsed: {}'.format(time.time() - start))
    print("total sum of squared error: {} (lower the better)".format(sse))
