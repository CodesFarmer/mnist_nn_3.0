This file I imitate the Author of http://neuralnetworksanddeeplearning.com/  
And I realize it by reference the http://ufldl.stanford.edu/wiki/index.php  
Author: Lowell  
Email: Jhonjoe.c@gmail.com  
To run the program, just run the test_nn.py file on your system  
Recommended envoriment:  
python2.7  
numpy  
You can run the test_nn.py to fast start  
The main function is TrainNet,  
TrainNet(train,test,epochs, lr, mode, ofmode, parameters)  
train and test are training data and test data seperately  
epochs control the iterative times for training  
lr is learning rate  
mode you can select "SGD"/"BGD"/"ACSGD" to train the network with （Stochastic gradient descent
）/（Batch gradient descent）/(Sparse coding)  
ofmode you can select different overfit mode, "L1"/"L2"/"DP"/"DPL1"/"DPL2"/"MO"/"MOL1"/"MOL2", where "MO" is momentum-based gradient decent, "DP" represent the "drop out" method, "L1" and "L2" are L1 regularization and L2 regularization saperately.
parameters are used to inputs the parameters  
"-l x" lambda control the weights of regulrization and cost function  
"-r x" rho, control the intensity of sparse coding  
"-b x" beta, control the weights between cost function and sparse coding  
"-p x" percent, control the percent of active neurons  
"-m x" mu, control the velocity inherit from previous learning  
