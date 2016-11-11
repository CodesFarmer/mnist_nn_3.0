import read_mnist
import nnetwork

print "Loading the dataset......"
tr_d, va_d, te_d = read_mnist.load_data_wrapper()

tr_d_1 = tr_d[0:1000]

tr_d_2 = tr_d[10000:20000]

te_inputs = [(x, read_mnist.vectorized_result(y)) for x,y in te_d]
te_d = te_inputs

print "Initializing the network......"
nnclf = nnetwork.Network([784, 30, 10])
#nnclf.update_network_dp()
#print nnclf.weights
print "Training the network......"

nnclf.reset()
nnclf.TrainNet(tr_d[0:10000],va_d,30, 0.5, "SGD", "MO", "-m 0.2")
nnclf.reset()
nnclf.TrainNet(tr_d[0:10000],va_d,30, 0.5, "SGD", "", "-m 0.0")

"""
nnclf.reset_qinit()
nnclf.TrainNet(tr_d_1,va_d,200, 0.5, "SGD", "DP")

nnclf.reset_qinit()
nnclf.TrainNet(tr_d_1,va_d,200, 0.5, "SGD")

nnclf.reset_qinit()
nnclf.TrainNet(tr_d_1,va_d,200, 0.5, "SGD", "L1", "-l 0.1")

nnclf.reset_qinit()
nnclf.TrainNet(tr_d_1,va_d,200, 0.5, "SGD", "L2", "-l 0.1")

nnclf.reset_qinit()
nnclf.TrainNet(tr_d_1,va_d,200, 0.5, "SGD", "DPL2", "-l 0.1")
"""


# print "Size:", len(tr_d_1)
# nnclf.plotimg(tr_d,30, 0.05, te_d)
# nnclf.reset()
# nnclf.plotimg(tr_d,30, 0.5, te_d)
# nnclf.reset()
# nnclf.plotimg(tr_d,30, 5, te_d)


"""
print "Evaluating on test data......"
print "{0}/{1} on test data !".format(nnclf.Evaluate(va_d), len(va_d))
"""