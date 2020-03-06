import sys
import numpy as np
import torch
from torch.optim import Adam

from dlt.basic.batch import make_batch
from dlt.basic.cross_entropy import CrossEntropyLoss
from dlt.basic.mse_loss import MSELoss
from dlt.basic.pytorch_utils import torch_to_np, np_to_torch
from dlt.basic.summary import regression_summary, classification_summary
from dlt.basic.unet import UNet
from sentinel_dataset import Dataset

import os

data_bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']


GPU_NO = 0
batch_size = 16
win_size = [256, 256]
n_iteration = 5000
lr = 0.0004


#Make output folder
if not os.path.isdir('saved_models'):
    os.mkdir('saved_models')


label_bands = ['roads']
output_path = 'saved_models/saved_model_new.pt'

criterion = CrossEntropyLoss()
summary = classification_summary
n_outputs = 2
mask_clouds = True
change_class_numbers = True

training_tiles = ['T32VLK_20170705T105026','T32VLK_20180608T112325', 'T32VLK_20190710T125432',
                  'T32VLL_20170705T105026', 'T32VLL_20190628T131710', 'T32VNR_20170630T105305',
                  'T32VNR_20170721T110758', 'T32VNR_20180705T130423', 'T32VNR_20190726T122036',
                  'T33WXT_20150822T104035', 'T33WXT_20160723T105623', 'T33WXT_20180728T131913',
                  'T33WXT_20190718T113853']
validation_tiles = ['T33WWS_20170725T105028', 'T33WWS_20180701T145738', 'T33WWS_20190615T110207']

#Model and optimizer
model = UNet(in_channels=len(data_bands), n_classes=n_outputs, use_bn=True).cuda(GPU_NO)
optimizer = Adam(model.parameters(),lr=lr)

# Datasets
train_dataset = Dataset([os.path.join('data', p) for p in training_tiles],
                        band_identifiers=data_bands,
                        label_identifiers=label_bands,
                        )

val_dataset = Dataset([os.path.join('data', p) for p in validation_tiles],
                      band_identifiers=data_bands,
                      label_identifiers=label_bands,
                      )

#Traing steps
acc = 0.0
for iteration in range(n_iteration+1):
    model.train()

    data, target = make_batch(train_dataset, win_size, batch_size,
                              mask_clouds=mask_clouds, change_class_numbers=change_class_numbers)

    #Put data on gpu
    data = np_to_torch(data).cuda(GPU_NO)
    target = np_to_torch(target).cuda(GPU_NO)

    #Run data through network
    pred = model(data)

    #Run training step
    loss = criterion(pred,target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Move data from torch to numpy
    data = torch_to_np(data)
    pred = torch_to_np(pred)
    target = torch_to_np(target)
    loss = torch_to_np(loss)

    #Print results for batch
    summary(iteration, 'Training', data, target, pred, loss)


    #Print results for validation every 500th epoch
    if iteration%500==0:
        model.eval()

        #Loop through 50 batches:
        pred = []
        target = []
        data = []
        for i in range(50):
            d, t = make_batch(val_dataset, win_size, batch_size,
                              mask_clouds=mask_clouds, change_class_numbers=change_class_numbers)
            p = torch_to_np(model(np_to_torch(d).cuda(GPU_NO)))

            pred.append(p)
            target.append(t)
            data.append(d)

        pred = np.concatenate(pred, 0)
        target = np.concatenate(target, 0)
        data = np.concatenate(data, 0)

        acc_i = summary(iteration, 'Validation', data, target, pred)

        if acc_i >= acc:
            torch.save(model.state_dict(), output_path)
            acc = acc_i
            print('Saving model at iteration', iteration)

