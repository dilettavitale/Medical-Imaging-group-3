# python3 train.py --model output/model.pth --plot output/plot.png
# set the matplotlib backend so figures can be saved in the background
import sys
import matplotlib
import os
matplotlib.use("Agg")
import config as config
#import the necessary packages
from dataset import ClassificationDataset
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import gc
#import segmentation_models_pytorch as smp
#from SeNet import SeNetEncoder
from gradNorm import gradNorm
from MultiTaskLoss import MultiTaskLoss
#from segmentation_models_pytorch.decoders.unet.model import Unet
from modelgithub import UNet
from tqdm import tqdm
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=True, help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 8
EPOCHS = 30

# define the path to the images dataset
LETTER = 'A'
root_dir = config.IMAGE_DATASET_PATH
csv_file = '../../dataset/folds/fold' +LETTER+ '_train.csv'

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the train dataset
trainData = ClassificationDataset(csv_file=csv_file, root_dir=root_dir)
    
print(trainData)

print(f"[INFO] found {len(trainData)} examples in the training set...")

torch.manual_seed(12345)
# create the training loader
trainLoader = DataLoader(trainData, shuffle=True,	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,	num_workers=1)   
# calculate steps per epoch for training and test set
trainSteps = len(trainData) // config.BATCH_SIZE

# initialize the model
print("[INFO] initializing the model...")
sys.stdout.flush()
unet = UNet(3,1).to(config.DEVICE)
    
# initialize our optimizer and loss function
opt = Adam(unet.parameters(), lr=INIT_LR)
#lossFn = nn.CrossEntropyLoss()
multitaskloss=MultiTaskLoss()
gradNorm=gradNorm(unet,opt)
# initialize a dictionary to store training history
H = {"train_loss0": [],"train_loss1": [],"train_loss2": []}

# measure how long training is going to take
print("[INFO] training the network...")
sys.stdout.flush()
startTime = time.time()
# loop over our epochs
for e in tqdm(range(config.NUM_EPOCHS)):
  # set the model in training mode
  unet.train()
  i=1
  # initialize the total training and validation loss
  totalTrainLoss0 = 0
  totalTrainLoss1 = 0
  totalTrainLoss2 = 0
  # loop over the training set
  for img,tlabel,tmask,tintensity in trainLoader:
      torch.cuda.empty_cache()
      img,tlabel,tmask,tintensity=img.to(config.DEVICE), tlabel.to(config.DEVICE), tmask.to(config.DEVICE), tintensity.to(config.DEVICE)
      pred= unet(img)
      loss = multitaskloss(pred,mask=tmask,label=tlabel,intensity=tintensity)

      # zero out the gradients, perform the backpropagation step,
      # and update the weights
      
      gradloss=gradNorm.forward(loss)
      # add the loss to the total training loss so far and
      # calculate the number of correct predictions
      totalTrainLoss0 += loss[0].item()
      totalTrainLoss1 += loss[1].item()
      totalTrainLoss2 += loss[2].item()
      print("Batch n.o: ",i)
      sys.stdout.flush()
      i+=1
  # calculate the average training
  #totalTrainLoss = totalTrainLoss1+totalTrainLoss0+totalTrainLoss2
  avgTrainLoss0 = totalTrainLoss0/trainSteps
  avgTrainLoss1 = totalTrainLoss1/trainSteps
  avgTrainLoss2 = totalTrainLoss2/trainSteps

  # update our training history
  H["train_loss0"].append(avgTrainLoss0)
  H["train_loss1"].append(avgTrainLoss1)
  H["train_loss2"].append(avgTrainLoss2) 
  # print the model training information
  print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
  sys.stdout.flush()
  #print("Average train loss 0: {:.6f}",avgTrainLoss0)
  sys.stdout.flush()
  #print("Average train loss 1: {:.6f}",avgTrainLoss1)
  sys.stdout.flush()
  #print("Average train loss 2: {:.6f}",avgTrainLoss2)
  sys.stdout.flush()
  #print("Numero di elementi in total train loss 0",len(totalTrainLoss0))
   
print("H train_loss 0",H["train_loss0"])   
sys.stdout.flush()
print("H train_loss 1",H["train_loss1"])
sys.stdout.flush()
print("H train_loss 2",H["train_loss2"]) 
sys.stdout.flush()        
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
sys.stdout.flush()
    
# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss0"], label="train_loss0")
plt.title("Training Loss on Segmentation")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("/workspace/FinalProject/Classificazione/output/plot_dice_segmentation.png")

plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss1"], label="train_loss1")
plt.title("Training Loss on Classification")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("/workspace/FinalProject/Classificazione/output/plot_crossentropy_classification_with_dice.png")

plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss2"], label="train_loss2")
plt.title("Training Loss on Intensity")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("/workspace/FinalProject/Classificazione/output/plot_bce_intensity_with_dice.png")
# serialize the model to disk
torch.save(unet, config.MODEL_PATH_A)
