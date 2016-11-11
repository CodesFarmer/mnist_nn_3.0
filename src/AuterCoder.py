from PIL import Image
import read_mnist
import nnetwork
import numpy as np
#from matplotlib import image as matim
from matplotlib import pyplot as matplot

tr_d, va_d, te_d = read_mnist.load_data_wrapper()

##Adjust the dataset to form our dataset
#tr_d = [(x,x) for x,y in tr_d]
#va_d = [(x,x) for x,y in va_d]
#te_d = [(x,x) for x,y in te_d]

tr_d = [(x/np.sqrt(np.sum(x*x,0)), x/np.sqrt(np.sum(x*x,0))) for x,y in tr_d]
va_d = [(x/np.sqrt(np.sum(x*x,0)), x/np.sqrt(np.sum(x*x,0))) for x,y in va_d]
te_d = [(x/np.sqrt(np.sum(x*x,0)), x/np.sqrt(np.sum(x*x,0))) for x,y in te_d]

#"""
grayimg = np.zeros((28,28))
for i in xrange(28):
    for j in xrange(28):
        grayimg[i,j] = np.uint8((tr_d[0][0][i*28+j]*255))

matplot.imshow(grayimg, cmap='Greys_r')
matplot.axis('off')
matplot.show()
#"""

#build a network whose objective output is its input
nnclf = nnetwork.Network([784,100,784])
#nnclf.TrainNet(tr_d,va_d,1,0.1,"SGD")
nnclf.TrainNet(tr_d,va_d,10,0.1,"ACSGD", 0.2, 0.1)

np.save("weights.npy", nnclf.weights[0])

for k in xrange(100):
    grayimg1 = np.zeros((28, 28))
    weights = [x - np.min(nnclf.weights[0][k]) for x in nnclf.weights[0][k]]
    weights = weights/np.sqrt(sum(x*x for x in weights))
    for i in xrange(28):
        for j in xrange(28):
            ind = i*28+j
            grayimg1[i,j] = np.uint8(255*weights[ind])

    matplot.imshow(grayimg1, cmap='Greys_r')
    matplot.axis('off')
    matplot.savefig("image_%03d"%(k))
    #matplot.show()
