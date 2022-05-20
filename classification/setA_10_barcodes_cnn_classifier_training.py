#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math as m
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data_dir = "" # directory containing training data


# In[ ]:


'''
Loading Data
'''
Y_file = np.load(os.path.join(data_dir, 'setA_10_barcodes_classes.npy'))
X_file = np.load(os.path.join(data_dir, 'setA_10_barcodes_raw_windows.npy'))
print('Data loaded successfully')

Y_file = Y_file.flatten()

window_length = 19881 
X_file = X_file[:,:window_length]


# In[ ]:


'''
Train, test split
'''
X_train = X_file.reshape(
        len(X_file), X_file.shape[1], 1)
labels_train = Y_file

X_tr, X_vld, lab_tr, lab_vld = train_test_split(
        X_train, labels_train, stratify = labels_train, train_size = 0.8)

X_vld, X_test, lab_vld, lab_test = train_test_split(
    X_vld, lab_vld, stratify = lab_vld)

y_tr = lab_tr.astype(int)
y_vld = lab_vld.astype(int)
y_test = lab_test.astype(int)

print('Data split done')


# In[ ]:


'''
If gpu is available we will use it
'''
use_cuda = True


# In[ ]:


'''
Reshaping data
'''
reshape = 141

X_tr = X_tr.reshape(len(X_tr),1,reshape,reshape)
X_vld = X_vld.reshape(len(X_vld),1,reshape,reshape)
X_test = X_test.reshape(len(X_test),1,reshape,reshape)
print('Data reshaping done')


# In[ ]:


'''
Zipping data together and storing in trainloader objects
'''
train_set = list(zip(X_tr, y_tr))
val_set = list(zip(X_vld, y_vld))
test_set = list(zip(X_test, y_test))								  
print('Done zipping and converting')


# In[ ]:


'''
Creating the neural net
'''
best_accuracy = -float('Inf')
best_params = []

batch_size = 30

trainloader = torch.utils.data.DataLoader(
		train_set, batch_size=batch_size,shuffle=True, num_workers=2)
vldloader = torch.utils.data.DataLoader(
		val_set, batch_size=batch_size,shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
		test_set, batch_size=batch_size,shuffle=True, num_workers=2)

lr = 0.001
epochs = 250
momentum = 0.7557312793639288
		
O_1 = 17
O_2 = 18
O_3 = 32
O_4 = 37

K_1 = 3
K_2 = 1
K_3 = 4
K_4 = 2

KP_1 = 4
KP_2 = 4
KP_3 = 1
KP_4 = 1

conv_linear_out = int(m.floor((m.floor((m.floor((m.floor((m.floor((reshape - K_1 + 1)/KP_1) - 
	K_2 + 1)/KP_2) - K_3 + 1)/KP_3) - K_4 + 1)/KP_4)**2)*O_4))
	
FN_1 = 148

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1,O_1,K_1),nn.ReLU(), 
                                   nn.MaxPool2d(KP_1))

        self.conv2 = nn.Sequential(nn.Conv2d(O_1,O_2,K_2),nn.ReLU(),
                                   nn.MaxPool2d(KP_2))

        self.conv3 = nn.Sequential(nn.Conv2d(O_2,O_3,K_3),nn.ReLU(),
                                   nn.MaxPool2d(KP_3))

        self.conv4 = nn.Sequential(nn.Conv2d(O_3,O_4,K_4),nn.ReLU(),
                                   nn.MaxPool2d(KP_4))

        self.fc1 = nn.Linear(conv_linear_out, FN_1, nn.Dropout(0.2))


        self.fc2 = nn.Linear(FN_1, 10)


    def forward(self, x):
        x = x.float()
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(len(x), -1)
        x = F.logsigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

net = CNN()
if use_cuda and torch.cuda.is_available():
	net.cuda()


# In[ ]:


'''
Train
'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

for epoch in range(250): 

    running_loss = 0.0
    for i,data in enumerate(trainloader, 0):
        inputs,labels = data
        if use_cuda and torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        outputs = outputs.to(dtype = torch.float64)
        labels = labels.to(dtype = torch.long)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print('Finished epoch number ' + str(epoch))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in vldloader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the validation set: %d %%' 
    % (100 * correct / total))


# In[ ]:


'''
Test
'''
correct = 0
total = 0
all_true = []
all_pred = []
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum().item()
        all_true.extend(labels)
        all_pred.extend(predicted)

    print('Accuracy of the network on the test set: %d %%' 
    % (100 * correct / total))

text_file = open("setA_10_barcodes_trained_cnn_results_20191015.txt", "w")
text_file.write("Accuracy of the network on the test set: %d %%" % (100 * correct / total))
text_file.close()

all_true = [x.item() for x in all_true]
all_pred = [x.item() for x in all_pred]


# In[ ]:


'''
Saving the trained net
'''
torch.save(net.state_dict(), "../utils/model/setA_10_barcodes_trained_cnn_20191015.pt")

