import numpy as np
import random
import plotfunc
import string

class Network(object):
    def __init__(self, sizes):
        #The number of layers
        self.sizes = sizes
        self.numlayers = len(sizes)
        #define the network by allocate its weights and biases
        #The number of biases are equal to the number of neurons except the first layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        """
        The number of weights are equal to n(i)*n(i+1), where i is the serial number of layer.
        As we can see from above description, the last layer doesn't have weights
        """
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:], sizes[:-1])]
        self.velw = [np.zeros(w.shape) for w in self.weights]
        self.velb = [np.zeros(b.shape) for b in self.biases]
    def reset(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(x,y) for x,y in zip(self.sizes[1:], self.sizes[:-1])]
        self.velw = [np.zeros(w.shape) for w in self.weights]
        self.velb = [np.zeros(b.shape) for b in self.biases]
    def reset_qinit(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(x,y)/np.sqrt(y) for x,y in zip(self.sizes[1:], self.sizes[:-1])]
        self.velw = [np.zeros(w.shape) for w in self.weights]
        self.velb = [np.zeros(b.shape) for b in self.biases]
    def TrainNet(self,tr_d,va_d,epochs,lr,mode, relz="", parameters=""):
        """
        :param tr_d: training data
        :param va_d: validation data
        :param epochs: the number of max epochs for training
        :param lr: the learning rate
        :param mode: the approach for training, BGD or SGD
        """
        lmbda, rho, beta, mu = translate_par(parameters)

        if(mode == "BGD"):
            self.BGD(tr_d,va_d,epochs,lr, relz,lmbda, mu)
        elif(mode == "SGD"):
            self.SGD(tr_d,va_d,epochs,lr, relz,lmbda, mu)
        elif(mode == 'ACSGD'):
            self.ACSGD(tr_d,va_d,epochs,lr, rho, beta)
        else:
            print "Wrong mode !"
    def BGD(self, tr_d, va_d, epochs, lr, relz=None, lmbda=0.0):
        """
        This function optimize the network by all training data
        """
        print "Training the network with BGD......"
        trlen = len(tr_d)
        for j in xrange(epochs):
            random.shuffle(tr_d)
            self.update_network(tr_d, lr)
            #for i in xrange(trlen):
            #   self.update_network([tr_d[i]],lr)
            if(va_d):
                print "Epoch {0}: {1}/{2}".format(j, self.Evaluate(va_d),len(va_d))
            else:
                print "Epoch {0}:".format(j)
    def SGD(self, tr_d, va_d, epochs, lr, relz="", lmbda=0.0, mu=0.0):
        """
        This function realizes the stochastic gradient descent
        First, we split the dataset into m batches, the we update the
        network based on those batches
        """
        print "Training the network with SGD......"
        trlen = len(tr_d)
        batch_size = 10
        j = 0
        accuracy = []
        while j < epochs:
            random.shuffle(tr_d)
            batches = [tr_d[k:k+batch_size] for k in xrange(0,trlen,batch_size)]
            for tr_batch in batches:
                if(relz[0:2] != "DP"):
                    self.update_network(tr_batch,lr, relz, lmbda, mu)
                else:
                    self.update_network_dp(tr_batch,lr,relz[2:], lmbda, mu)
            acy = self.Evaluate(va_d)
            accuracy.append(acy)
            if (va_d):
                print "Epoch {0}: {1}/{2}".format(j, acy, len(va_d))
            else:
                print "Epoch {0}:".format(j)
            j += 1
        np.save("test_acy_%s_qinit_1.npy"%(relz), accuracy)

    def update_network(self, tr_d, lr, relz="", lmbda=0.0, mu=0.0):
        """
        Update the network by execute a feed forward step and a back propagation step
        on every training sample
        """
        trlen = float(len(tr_d))
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in tr_d:
            delta_b_single, delta_w_single = self.backppg_ce(x,y)
            delta_b = [db+dbs for db,dbs in zip(delta_b, delta_b_single)]
            delta_w = [dw+dws for dw,dws in zip(delta_w, delta_w_single)]
        #update the parameters in network
        if(relz==""):
            mu=0.0
        elif(relz[0:2] == "MO"):
            relz = relz[2:]
        self.velw = [mu*vw-(lr/trlen)*dw for vw,dw in zip(self.velw, delta_w)]
        self.velb = [mu*vb-(lr/trlen)*db for vb,db in zip(self.velb, delta_b)]
        self.biases = [b + vb for b,vb in zip(self.biases, self.velb)]
        if(relz == "L2"):
            self.weights = [w + vw - (lr/trlen/100)*lmbda*w for w,vw in zip(self.weights, self.velw)]
        elif(relz == "L1"):
            self.weights = [w + vw - (lr/trlen/100)*lmbda*np.sign(w) for w,vw in zip(self.weights, self.velw)]
        else:
            self.weights = [w + vw for w,vw in zip(self.weights, self.velw)]
    def Evaluate(self,te_d):
        ncor = [(np.argmax(self.forward(x)), y) for x,y in te_d]
        #for x,y in te_d:
        #    ncor.append([(np.argmax(self.forward(x)), y)])
        #print "Size: ", ncor.shape
        return sum(int(x==y) for x,y in ncor)
    def forward(self,x):
        a = x
        for b,w in zip(self.biases, self.weights):
            a = sigmod(np.dot(w, a) + b)
        return a
    def ACSGD(self, tr_d, va_d, epochs, lr, rho, beta):
        print "Training the network for autocoder with ACSGD......"
        trlen = len(tr_d)
        batch_size = 10
        j = 0
        while j < epochs:
            random.shuffle(tr_d)
            batches = [tr_d[k:k + batch_size] for k in xrange(0, trlen, batch_size)]
            for tr_batch in batches:
                self.update_network_sparse(tr_batch, lr, rho, beta)
            if va_d:
                print "Epochs {0}: {1}".format(j, self.devation(va_d))
            else:
                print "Epochs {0}".format(j)
            j += 1
    def Forward_AT(self,x):
        activations = [x]
        for b,w in zip(self.biases,self.weights):
            x = sigmod(np.dot(w,x)+b)
            activations.append(x)
        return activations
    def devation(self,te_d):
        ncor = [[self.forward(x), y] for x, y in te_d]
        tmp2 = [x-y for x,y in ncor]
        tmp3 = sum(x*x for x in (ent for ent in tmp2))
        tmp3 = sum(float(x) for x in tmp3)
        return tmp3
    def backppg_ce(self,x,y):
        """
        This function employ the cross-entroy cost function instead of quadratic cost function
        which is J(y,a) = -1/n*Sigmod{y*log(a)+(1-y)*log(1-a)}
        """
        activation = x
        activations = [x]
        zs = []
        #feed forward
        for w,b in zip(self.weights,self.biases):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmod(z)
            activations.append(activation)
        #back propagation
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta = activations[-1]-y
        delta_w[-1] = np.dot(delta, activations[-2].transpose())
        delta_b[-1] = delta
        for j in xrange(2, self.numlayers):
            delta = np.dot(self.weights[-j+1].transpose(), delta)*sigmod_deri(zs[-j])
            delta_b[-j] = delta
            delta_w[-j] = np.dot(delta, activations[-j-1].transpose())
        return (delta_b, delta_w)
    def Evaluate_a_c(self,tr_d):
        ncor = [(self.forward(x), y) for x,y in tr_d]
        acy=sum(int(np.argmax(x)==np.argmax(y)) for x,y in ncor)/float(len(tr_d))
        qcost_te = [x-y for x,y in ncor]
        qcost_sum = [np.sqrt(np.dot(x.transpose(), x)) for x in qcost_te]
        cost = np.mean(qcost_sum)
        return (acy,cost)
    def update_network_dp(self, tr_d, lr, relz="", lmbda =0.0):#
        #random discard some neurons
        nnw_dp = []
        perc = 0.8#0.8
        for j in xrange(0,self.numlayers):
            len = self.sizes[j]
            pos = range(len)
            random.shuffle(pos)
            nnw_dp.append(pos)
        weights_bk = [np.copy(w) for w in self.weights]
        biases_bk = [np.copy(b) for b in self.biases]
        for j in xrange(1,self.numlayers-1):
            len = self.sizes[j]
            #left
            wi = self.weights[j-1]
            bi = self.biases[j-1]
            pos = nnw_dp[j]
            for iter in pos[int(len*perc):]:
                tmp = np.zeros((1, self.sizes[j-1]))
                wi[iter] = tmp
                bi[iter] = 0
            self.weights[j-1] = wi
            self.biases[j-1] = bi
            #right
            wi = self.weights[j]
            wi = wi.transpose()
            for iter in pos[int(len*perc):]:
                tmp = np.zeros((self.sizes[j+1]))
                wi[iter] = tmp
            wi = wi.transpose()
            self.weights[j] = wi
        self.update_network(tr_d, lr, relz, lmbda)
        for j in xrange(1,self.numlayers-1):
            len = self.sizes[j]
            #left
            wi = self.weights[j-1]
            bi = self.biases[j-1]
            wwi = weights_bk[j-1]
            bbi = biases_bk[j-1]
            pos = nnw_dp[j]
            for iter in pos[int(len*perc):]:
                wi[iter] = wwi[iter]
                bi[iter] = bbi[iter]
            self.weights[j-1] = wi
            self.biases[j-1] = bi
            #right
            wi = self.weights[j]
            wwi = weights_bk[j]
            wi = wi.transpose()
            wwi = wwi.transpose()
            for iter in pos[int(len*perc):]:
                wi[iter] = wwi[iter]
            wi = wi.transpose()
            self.weights[j] = wi
    def backppg(self, x, y):
                """
                This function execute a feed forward step, then we can get the residual between
                output and the expected output
                Then a back propagation step are execute to calculate the delta_w and delta_b
                """
                # feed forward
                activation = x
                activations = [x]
                zs = []
                for w, b in zip(self.weights, self.biases):
                    z = np.dot(w, activation) + b
                    zs.append(z)
                    activation = sigmod(z)
                    activations.append(activation)

                # back propagation, start from the last layer
                delta_b = [np.zeros(b.shape) for b in self.biases]
                delta_w = [np.zeros(w.shape) for w in self.weights]
                # The residual of last layer equal (a[l]-y)*f'(z[l])
                delta = (activations[-1] - y) * sigmod_deri(zs[-1])
                delta_b[-1] = delta
                delta_w[-1] = np.dot(delta, activations[-2].transpose())

                for l in xrange(2, self.numlayers):
                    delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmod_deri(zs[-l])
                    delta_b[-l] = delta
                    delta_w[-l] = np.dot(delta, activations[-l - 1].transpose())
                return (delta_b, delta_w)

def sigmod(z):
    return 1.0/(1.0+np.exp(-z))
def sigmod_deri(z):
    return sigmod(z)*(1-sigmod(z))
def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
def tanh_deri(z):
    return 1-tanh(z)*tanh(z)
def translate_par(parms):
    str = string.split(parms,sep=",")
    rho=0.2
    beta=0.0
    lmbda=0.0
    mu = 0.0
    for par in str:
        par = string.strip(par, " ")
        tandv = string.split(par, " ")
        if(tandv[0] == '-r'):
            rho = float(tandv[1])
        elif(tandv[0] == '-b'):
            beta = float(tandv[1])
        elif(tandv[0] == '-l'):
            lmbda = float(tandv[1])
        elif(tandv[0] == '-m'):
            mu = float(tandv[1])
    return (lmbda,rho,beta,mu)