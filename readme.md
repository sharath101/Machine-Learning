**If you are forking the project to your gitlab account, please make sure your repo is private**

> Getting started:

1. Fork this repository

2. Clone(download) the forked repository to your computer using the command: 
`git clone https://gitlab.com/<gitlab-username>/mlabs-2019-problem-set.git`

3. Setup a virtual environment to run the files without dependency issues
	1. Install virtualenv package: 
		`pip install virtualenv`
	2. Create a virtual environment: 
		`virtualenv mlabs`
	3. Activate the environment
		a. Windows: 
			`mlabs\Scripts\activate`
		b. Ubuntu: 
			`source mlabs/bin/activate`
		
		You'll see the command line starting with **(mlabs)**
	
	**From the next time onwards you need only do Step 3 before starting to work on the problems**
	
4. Change working directory to the problem set folder: 
`cd mlabs-2019-problem-set`

5. Install the required packages: 
`pip install -r version_list.txt`

6. Start working on the problem statements. Happy Machine Learning :)

> Keep in mind - **When in trouble Google is always your best friend**


The assignment consists of three problems based on basic machine learning and one on computer vision. 
Numerous problems in these areas are well studied in statistics and applied mathematics. 
Solutions are to be implemented in python by filling out required functions in each python file. 
A basic framework for data I/O and evaluation is already provided (see header comments in each python file).
Please note that all required libraries are already imported so please DO NOT import anything new.

The four problems are briefly discussed below. 

1. **Gradient Descent**: This is a popular optimization problem to find solutions to differentiable equations.
Typically, learning problems involve minimizing a cost function by appropriately setting model parameters. 
In this task, we are given a set of (noisy) points on a line and we wish to retrieve model parameters (intercept and slope) through gradient descent.
Please refer to '*gradient_descent.py*' and inline comments for further details. 

2. **Eigenfaces**: This is a popular application of learning a basis representation of input data. 
The application of this technique is the basis for simple recognition/compression algorithms. 
In this task, we want to learn orthonormal basis using PCA of images that correspond to faces.
Please refer to '*eigenfaces.py*' and inline comments for further details. 

3. **Classification**: This is among the basic tasks of machine learning problems. 
Here, we will learn a classifier to using groundtruth labels on the training data to be able to distinguish between two object classes.
You will use the scikit library to learn two classifiers (svm and random forest).
Feel free to explore the parameters of both models to maximize classifier performance. 
Please refer to '*classification.py*' and inline comments for further details. 

4. **Disparity map**: This is among the basic tasks of 3D computer vision
Here, given two different perspectives of the same scene, we will reconstruct an approximate of the depth map.
This is called the disparity map (higher disparity is similar to lower depth).
You will use the scikit library to implement the module. Feel free to explore the parameters 'downsample' and 'patchsize'
Please refer to '*disparity.py*' and inline comments for further details.

