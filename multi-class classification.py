import matplotlib.pyplot as plt
import numpy as np
import gzip, pickle
with gzip.open("mnist_2_vs_9.gz") as f:
   data = pickle.load(f, encoding='bytes')
import gzip, pickle
with gzip.open('mnist.pkl.gz') as f: train_set, valid_set, test_set = pickle.load(f, encoding='bytes')

Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev = data[b"Xtrain"], data[b"Ytrain"], data[b"Xtest"], data[b"Ytest"], data[b"Xdev"], data[b"Ydev"]

Xtrain[:, 783] = 1
Xdev[:, 783] = 1
Xtest[:, 783] = 1

# calculatest the average squared error based on w and two data sets
def avgError(Y, w, X):
    return 0.5 / X.shape[0] * np.linalg.norm(np.subtract(Y, np.matmul(X, w)))**2

# calculates the misclassification error based on w and two data sets
def miserror(Y, Yhat, b):
   Yhat_labels=(Yhat-b)>=0
   errors = np.abs(Yhat_labels-Y)
   return 100*sum(errors)/(Yhat.shape[0]*1.0)

# calculate w using gradient descent
def GD(Y, w, X, step, l):
   r = -1.0 / X.shape[0]
   return np.subtract(w, step * (r * np.matmul(np.transpose(X), np.subtract(Y, np.matmul(X, w))) + l * w))
   
def SGD(Y, w, X, step, l, i):
   value = Y[i] - np.dot(w, X[i, :])
   return np.subtract(w, step * (-value * X[i, :] + l * w))

# Problem 1.1.2   
def p1():
   N = Xtrain.shape[0]
   b = 0.3
   # list contains values as candidates of lambda
   list = np.linspace(0.1, 5, 50)
   # wlist stores candidates of w, and elist stores the corresponding error
   wlist = []
   elist = []
   one = np.identity(784)
   second = 1.0 / N * np.matmul(np.transpose(Xtrain), Ytrain)
   for i in range(0, len(list)):
      first = np.linalg.inv(1.0 / N * np.matmul(np.transpose(Xtrain), Xtrain) + list[i] * one)
      w = np.matmul(first, second)
      error = 0.5 / N * np.linalg.norm(Ytrain - np.matmul(Xtrain, w))**2 + list[i] * 0.5 * np.linalg.norm(w)**2
      wlist.append(w)
      elist.append(error)
   
   # gets the index of min error and the corresponding w
   index = np.argmin(elist)
   w = wlist[index]
   print('lambda: ' + str(list[index]))
   print('avg training squared error: ' + str(avgError(Ytrain, w, Xtrain)))
   print('avg dev squared error: ' + str(avgError(Ydev, w, Xdev)))
   print('avg test squared error: ' + str(avgError(Ytest, w, Xtest)))
   print('misclassification error on train: ' + str(miserror(Ytrain, np.matmul(Xtrain, w), b)))
   print('misclassification error on dev: ' + str(miserror(Ydev, np.matmul(Xdev, w), b)))
   print('misclassification error on test: ' + str(miserror(Ytest, np.matmul(Xtest, w), b)))

def p2():
   # Problem 1.2.3b  
   l = 0.01
   step = 0.034
   b = 0.3
   x = [x for x in range(1, 200)]
   w = np.zeros(784).transpose()
   # error1 saves avg squared error, where error2 saves misclassification error
   trainerror1 = []
   deverror1 = []
   testerror1 = []
   trainerror2 = []
   deverror2 = []
   testerror2 = []
   for j in range(0, len(x)):
      w = GD(Ytrain, w, Xtrain, step, l)
      trainerror1.append(avgError(Ytrain, w, Xtrain))
      deverror1.append(avgError(Ydev, w, Xdev))
      testerror1.append(avgError(Ytest, w, Xtest))
      trainerror2.append(miserror(Ytrain, np.matmul(Xtrain, w), b))
      deverror2.append(miserror(Ydev, np.matmul(Xdev, w), b))
      testerror2.append(miserror(Ytest, np.matmul(Xtest, w), b))
   plt.plot(x, trainerror1, 'ro', label = 'training avg squared error')
   plt.plot(x, deverror1, 'go', label = 'dev avg squared error')
   plt.plot(x, testerror1, 'bo', label = 'test avg squared error')
   plt.xlabel('Iteration')
   plt.ylabel('Error')
   plt.title('Linear regression using gradient descent')
   plt.legend()
   plt.show()

   # Problem 1.2.3c
   x = x[19:]
   trainerror2 = trainerror[19:]
   deverror2 = deverror[19:]
   testerror2 = testerror[19:]
   plt.plot(x, trainerror2, 'r', label = 'misclassification error on training')
   plt.plot(x, deverror2, 'g', label = 'misclassification error on dev')
   plt.plot(x, testerror2, 'b', label = 'misclassification error on test')
   plt.xlabel('Iteration')
   plt.ylabel('Error')
   plt.title('Linear regression using gradient descent')
   plt.legend()
   plt.show()
   print(min(testerror))

def p3():
   # Problem 1.3b
   b = 0.5
   l = 0.01
   step = 0.0
   x = [x for x in range(1, 60000)]
   
   w = np.zeros(784).transpose()
   trainerror1 = []
   deverror1 = []
   testerror1 = []
   trainerror2 = []
   deverror2 = []
   testerror2 = []
   for j in range(0, len(x)):
      i = np.random.randint(0, 784)       
      if j % 500 == 0:
         trainerror1.append(avgError(Ytrain, w, Xtrain))
         deverror1.append(avgError(Ydev, w, Xdev))
         testerror1.append(avgError(Ytest, w, Xtest))
         trainerror2.append(miserror(Ytrain, np.matmul(Xtrain, w), b))
         deverror2.append(miserror(Ydev, np.matmul(Xdev, w), b))
         testerror2.append(miserror(Ytest, np.matmul(Xtest, w), b))
         # decreases the stepsize by 1% every 500 updates
         step *= 0.99
         
      w = SGD(Ytrain, w, Xtrain, step, l, i)
         
   t = [k for k in range(0, len(trainerror1))]
   plt.plot(t, trainerror1, 'ro', label = 'training avg squared error')
   plt.plot(t, deverror1, 'go', label = 'dev avg squared error')
   plt.plot(t, testerror1, 'bo', label = 'test avg squared error')
   plt.xlabel('Iteration')
   plt.ylabel('Error')
   plt.title('Linear regression using stochastic gradient descent')
   plt.legend()
   plt.show()
   
   # Problem 1.3c
   t = t[2:]
   trainerror2 = trainerror2[2:]
   deverror2 = deverror2[2:]
   testerror2 = testerror2[2:]
   plt.plot(t, trainerror2, 'r', label = 'misclassification error on training')
   plt.plot(t, deverror2, 'g', label = 'misclassification error on dev')
   plt.plot(t, testerror2, 'b', label = 'misclassification error on test')
   plt.xlabel('Iteration')
   plt.ylabel('Error')
   plt.title('Linear regression using stochastic gradient descent')
   plt.legend()
   plt.show()
   print(min(testerror2))
   
def miserror2(Y,Yhat):
   indsYhat=np.argmax(Yhat,axis=1)
   indsY=np.argmax(Y,axis=1)
   errors = (indsYhat-indsY)!=0
   return 100*sum(errors)/(Yhat.shape[0]*1.0)
    
def p4():
   train = np.array(train_set[0])
   trainLabel = np.array(train_set[1])
   dev = np.array(valid_set[0])
   devLabel = np.array(valid_set[1])
   test = np.array(test_set[0])
   testLabel = np.array(test_set[1])
   Yntrain = np.zeros((train.shape[0], 10))
   Yndev = np.zeros((dev.shape[0], 10))
   Yntest = np.zeros((test.shape[0], 10))
   
   for i in range(0, trainLabel.shape[0]):
      Yntrain[i, trainLabel[i]] = 1
   
   for i in range(0, devLabel.shape[0]):
      Yndev[i, devLabel[i]] = 1

   for i in range(0, testLabel.shape[0]):
      Yntest[i, testLabel[i]] = 1
      
   l = 0.01
   step = 0.034
   x = [x for x in range(1, 100)]
   w = np.zeros((784, 10))
   
   trainerror1 = []
   deverror1 = []
   testerror1 = []
   trainerror2 = []
   deverror2 = []
   testerror2 = []
   for j in range(0, len(x)):
      w = GD(Yntrain, w, train, step, l)
      trainerror1.append(avgError(Yntrain, w, train))
      deverror1.append(avgError(Yndev, w, dev))
      testerror1.append(avgError(Yntest, w, test))
      trainerror2.append(miserror2(Yntrain, np.matmul(train, w)))
      deverror2.append(miserror2(Yndev, np.matmul(dev, w)))
      testerror2.append(miserror2(Yntest, np.matmul(test, w)))
   plt.plot(x, trainerror1, 'ro', label = 'training avg squared error')
   plt.plot(x, deverror1, 'go', label = 'dev avg squared error')
   plt.plot(x, testerror1, 'bo', label = 'test avg squared error')
   plt.xlabel('Iteration')
   plt.ylabel('Error')
   plt.title('Linear regression using gradient descent')
   plt.legend()
   plt.show()
   print(trainerror[len(trainerror) - 1])
   print(deverror[len(deverror) - 1])
   print(testerror[len(testerror) - 1])
   
   plt.plot(x, trainerror2, 'r', label = 'misclassification error on training')
   plt.plot(x, deverror2, 'g', label = 'misclassification error on dev')
   plt.plot(x, testerror2, 'b', label = 'misclassification error on test')
   plt.xlabel('Iteration')
   plt.ylabel('Error')
   plt.title('Linear regression using gradient descent')
   plt.legend()
   plt.show()
   print(min(testerror))
   
def main():
   #p1()
   #p2()
   p3()
   #p4()
   
main()