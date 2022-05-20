#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math as m
from sklearn.metrics import confusion_matrix
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data_dir = "" # directory containing training data


# In[2]:


'''
Loading Data
'''
Y_file = np.load(os.path.join(data_dir, 'final_10_orthogonal_barcodes_classes.npy'))
X_file = np.load(os.path.join(data_dir, 'final_10_orthogonal_barcodes_raw_windows.npy'))
print('Data loaded successfully')

Y_file = Y_file.flatten()

window_length = 19881

X_file = X_file[:,:window_length]

print(X_file.shape)
print(Y_file.shape)


# In[3]:


'''
Train, test split
'''
X_train = X_file.reshape(
        len(X_file), X_file.shape[1], 1)
labels_train = Y_file

X_tr, X_vld, lab_tr, lab_vld = train_test_split(
        X_train, labels_train, stratify = labels_train, train_size = 0.8)

X_vld, X_test, lab_vld, lab_test = train_test_split(X_vld, lab_vld, stratify = lab_vld)

y_tr = lab_tr.astype(int)
y_vld = lab_vld.astype(int)
y_test = lab_test.astype(int)

print('Data split done')

'''
If gpu is available we will use it
'''
use_cuda = True

'''
Reshaping data
'''
reshape = 141

X_tr = X_tr.reshape(len(X_tr),reshape,reshape)
X_vld = X_vld.reshape(len(X_vld),reshape,reshape)
X_test = X_test.reshape(len(X_test),reshape,reshape)

print(X_tr.shape)  # (64, 224, 224)
X_tr = np.repeat(X_tr[..., np.newaxis], 3, -1)
print(X_tr.shape)  # (64, 224, 224, 3)

X_vld = np.repeat(X_vld[..., np.newaxis], 3, -1)
X_test = np.repeat(X_test[..., np.newaxis], 3, -1)

print(X_tr.shape)

print('Data reshaping done')

'''
Zipping data together and storing in trainloader objects
'''
train_set = list(zip(X_tr, y_tr))
val_set = list(zip(X_vld, y_vld))
test_set = list(zip(X_test, y_test))								  
print('Done zipping and converting')


# In[5]:


net = models.resnet18()

if use_cuda and torch.cuda.is_available():
	net.cuda()


# In[6]:


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
momentum = 0.7557312793639288


# In[7]:


'''
Training
'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

for epoch in range(75):  # loop over the dataset multiple times
	running_loss = 0.0
	for i,data in enumerate(trainloader, 0):
		inputs,labels = data
		if use_cuda and torch.cuda.is_available():
			inputs = inputs.permute(0, 3, 1, 2)
			inputs = torch.nn.functional.interpolate(inputs,size=(224,224), mode='bilinear')
			inputs = inputs.float()
            
			inputs = inputs.cuda()
			labels = labels.cuda()

		optimizer.zero_grad()
		outputs =net(inputs)

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
            
			inputs = inputs.permute(0, 3, 1, 2)
			inputs = torch.nn.functional.interpolate(inputs,size=(224,224), mode='bilinear')
			inputs = inputs.float()
            
			inputs = inputs.cuda()
			labels = labels.cuda()
			outputs = net(inputs)
			_, predicted = torch.max(outputs.data, 1)
			total += len(labels)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the validation set: %d %%' 
	% (100 * correct / total))


# In[8]:


'''
Testing
'''
correct = 0
total = 0
all_true = []
all_pred = []
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        
        inputs = inputs.permute(0, 3, 1, 2)
        inputs = torch.nn.functional.interpolate(inputs,size=(224,224), mode='bilinear')
        inputs = inputs.float()
        
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

text_file = open("final_10_orthogonal_barcodes_cnn_results_20220221.txt", "w")
text_file.write("Accuracy of the network on the test set: %d %%" % (100 * correct / total))
text_file.close()

all_true = [x.item() for x in all_true]
all_pred = [x.item() for x in all_pred]


# In[23]:


'''
Saving the trained net
'''
torch.save(net.state_dict(), "../utils/model/final_10_orthogonal_barcodes_trained_cnn_20220221.pt")

