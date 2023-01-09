# import the necessary packages
from torch.utils.data import Dataset
import cv2
import pandas as pd
import torch
import os
from skimage import io
import imutils
import numpy as np

class ClassificationDataset(Dataset):
    def __init__(self, csv_file, root_dir):
         
        # store the image filepaths, the labels associated
        self.frame = pd.read_csv(csv_file, names=["Image", "Mask", "Label", "Intensity"])
        self.root_dir = root_dir     
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get the image and the associated label, and return them
		    # grab the image path from the current index
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
        image = io.imread(img_name)
        
        label = self.frame.iloc[idx, 2]
        if label == 'homogeneous ':
            lab = 0
        elif label == 'speckled ':
            lab = 1
        elif label == 'nucleolar ':
            lab = 2
        elif label == 'centromere ':
            lab = 3
        elif label == 'golgi ':
            lab = 4
        elif label == 'numem ':
            lab = 5
        else: # mistp
            lab = 6
        lab = torch.as_tensor(lab, dtype=torch.long)
        
        
        #image = imutils.resize(image, width=384, height=384)
        image2 = np.zeros((3,384,384))
        image2[0,:,:] = image
        image2[1,:,:] = image
        image2[2,:,:] = image
        image = image2
        image2=[]
        mask_name = os.path.join(self.root_dir, self.frame.iloc[idx, 1])
        mask = io.imread(mask_name)
        mask[mask == 255.0] = 1.0
        #mask = imutils.resize(mask, width=384, height=384)
        mask2 = np.zeros((1,384,384))
        mask2[0,:,:] = mask
        mask = mask2
        mask2=[]
        
        intensity = self.frame.iloc[idx, 3]
        if intensity == 'intermediate':
            inte = 0
        elif intensity == 'positive':
            inte = 1
        inte = torch.as_tensor(inte, dtype=torch.float32)
        image = torch.as_tensor(image, dtype=torch.float32) 
        mask = torch.as_tensor(mask, dtype=torch.float32)        
                    
		    # return a tuple of the image and its label
        return image, lab,mask,inte   
    def __len__(self):
    	# return the number of total samples contained in the dataset
    	return len(self.frame)
    