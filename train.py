import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
import json
from time import time
from tqdm import tqdm
from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
from data_process import get_dataloader

SHAPE = (1024,1024)
ROOT = './dataset/train/'
imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT))
trainlist = list(map(lambda x: x[:-8], imagelist))
NAME = './submits/log01_dink34.log'
WEIGHT_NAME = './weights/dlink34.pth'
JSON_NAME = './loss_json/dlink32.json'
BATCHSIZE_PER_CARD = 4
with open(NAME, "w+") as mylog:
    mylog.write("Begin Training!")

solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=4)
# data_loader,_,_,_ = get_dataloader()
# print(len(data_loader))
tic = time()
no_optim = 0
total_epoch = 10
train_epoch_best_loss = 100.
loss = []
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    with tqdm(total= len(data_loader_iter), desc=f"{epoch}/{total_epoch},", unit="image") as pbar:
        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
            pbar.update(1)
        train_epoch_loss /= len(data_loader_iter)
        loss.append(train_epoch_loss)
        
    with open(NAME, "a") as mylog:
        mylog.write('********')
        mylog.write(f"epoch:,{epoch}, time: int({time()-tic})")
        mylog.write(f"train_loss: {train_epoch_loss}")
        mylog.write(f"SHAPE: {SHAPE}")
    print('********')
    print(f"epoch:,{epoch}, time: int({time()-tic})")
    print(f"train_loss: {train_epoch_loss}")
    print(f"SHAPE: {SHAPE}")
    # print('********')
    # print( 'epoch:',epoch,'    time:',int(time()-tic))
    # print('train_loss:',train_epoch_loss)
    # print( 'SHAPE:',SHAPE)
    
    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save(WEIGHT_NAME)
    if no_optim > 6:
        with open(NAME, "a") as mylog:
            mylog.write(f"early stop at {epoch} epoch")
        print(f"early stop at {epoch} epoch")
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load(WEIGHT_NAME)
        with open(NAME, "a") as mylog:
            solver.update_lr(5.0, mylog=mylog, factor = True)
with open(NAME, "a") as mylog:
    mylog.write("Finish")
with open(JSON_NAME, "w+") as json_file:
    json.dump(loss, json_file, indent=2)
print('Finish!')