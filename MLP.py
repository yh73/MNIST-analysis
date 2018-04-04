import torch
import gzip, pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler as sampler
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import random

batch_size = 200
train_dataset = datasets.MNIST(root= './data', train = True, 
               transform = transforms.ToTensor(), download = True)
valid_dataset = datasets.MNIST(root= './data', train = True, 
               transform = transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root= './data', train = False, 
               transform = transforms.ToTensor())             

#split data to train and dev
indices = [x for x in range(0, len(train_dataset))]
train_sampler = sampler(indices[10000:])
dev_sampler = sampler(indices[:10000])

#load all data
train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, batch_size=batch_size, sampler=train_sampler)
dev_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset, batch_size=batch_size, sampler=dev_sampler)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size)

class Model(nn.Module):
   def __init__(self):
      super(Model, self).__init__()
      self.l1 = nn.Linear(784, 100)
      self.l2 = nn.Linear(100, 10)
      

   def forward(self, x):
      x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
      x = F.sigmoid(self.l1(x))
      X = F.log_softmax(self.l2(x), dim = 1)
      return x


model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr= 0.05, momentum=0.9, weight_decay=1e-5)

def train(epoch, trainmse, trainmissrate, devmse, devmissrate, testmse, testmissrate):
   model.train()
   size = len(train_loader)
   for batch_idx, (data, target) in enumerate(train_loader):
      data, target = Variable(data), Variable(target)
      optimizer.zero_grad()
      output = model(data)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      if (batch_idx == size / 2 or batch_idx == size - 1):
         tm1, tr1 = test(train_loader,50000)
         trainmse.append(tm1)
         trainmissrate.append(tr1)
         
         tm2, tr2 = test(dev_loader,10000)
         devmse.append(tm2)
         devmissrate.append(tr2)

         tm3, tr3 = test(test_loader,10000)
         testmse.append(tm3)
         testmissrate.append(tr3)

def test(test, len):
   model.eval()
   mse = 0
   correct = 0
   for data, target in test:
      data, target = Variable(data, volatile = True), Variable(target)
      output = model(data)
      mse += (criterion(output, target).data[0])**2
      pred = output.data.max(1, keepdim = True)[1]
      correct += pred.eq(target.data.view_as(pred)).cpu().sum()
   mse /=len
   missrate = 100.0 * (1.0- correct / len)
   return (mse, missrate)

def stand(x):
   x=x-np.min(x[:])
   x=x/(np.max(x[:])+1e-12)
   return x

def main():
   w = []
   trainmse = []
   trainmissrate = []
   devmse = []
   devmissrate = []
   testmse = []
   testmissrate = []
   for epoch in range(1, 15):
      print(epoch)
      w.append(epoch-0.5)
      w.append(epoch)
      train(epoch,trainmse,trainmissrate,devmse,devmissrate
               ,testmse, testmissrate)
   plt.plot(w, trainmse, 'r', label = 'trainset')
   plt.plot(w, devmse, 'g', label = 'devset')
   plt.plot(w, testmse, 'b', label = 'testset') 
   plt.xlabel('epoch')
   plt.ylabel('average square error')
   plt.title("average square error over half epoch")
   plt.legend()
   plt.show()
   plt.plot(w, trainmissrate, 'r', label = 'trainset')
   plt.plot(w, devmissrate, 'g', label = 'devset')
   plt.plot(w, testmissrate, 'b', label = 'testset') 
   plt.xlabel('epoch')
   plt.ylabel('missclassificationrate')
   plt.title("missclassification rate over half epoch")
   plt.legend()
   plt.show()

   print(min(testmissrate))
   weight = model.state_dict()
   weightlist = weight['l1.weight']
   vis = weightlist.cpu().numpy()
   # choose random index
   lst = random.sample(range(100), 8)
   for i in range(0, 8):
      d = stand(vis[lst[i]])
      plt.subplot(241 + i),plt.imshow(np.reshape(d, (28, 28)), cmap='gray')
   plt.show()

main()
