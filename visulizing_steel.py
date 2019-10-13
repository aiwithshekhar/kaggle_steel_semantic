from ptsemseg.segmentation_models_pytorch.unet.model import Unet
import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor
pd.set_option('display.max_columns', None)
# donot print the warning
warnings.filterwarnings("ignore")
print ('visualization steel')
# *****************to reproduce same results fixing the seed and hash*******************
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
# *****************to reproduce same results fixing the seed and hash*******************

# *****************create a 4d mask of 4 error classes in an array*********************
def make_mask(row_id, df):
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32) # float32 is V.Imp

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    return fname, masks
# *****************create a 4d mask of 4 error classes in an array*********************

# ******************DATA SETTER*********************************
class SteelDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, "train_images", image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']  # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        return img, mask

    def __len__(self):
        return len(self.fnames)
# ******************DATA SETTER*********************************

# ******************TRANSFORMATION*********************************
def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend([HorizontalFlip(p=0.5)])
    list_transforms.extend([Normalize(mean=mean, std=std, p=1),ToTensor()])
    list_trfms = Compose(list_transforms)

    return list_trfms
# ******************TRANSFORMATION*********************************

# ****************************DATA LOADER*****************************
def provider(data_folder,df_path,phase,mean=None,std=None,batch_size=8,num_workers=4,):
    '''Returns dataloader for the model training'''
    df = pd.read_csv(df_path)
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)
    df = train_df if phase == "train" else val_df
    image_dataset = SteelDataset(df, data_folder, mean, std, phase)
    dataloader = DataLoader(image_dataset,batch_size=batch_size,num_workers=num_workers,pin_memory=True,shuffle=True,)
    return dataloader
# ****************************DATA LOADER*****************************

# ****************************thresholds sigmoid output prediction***********************
def predict(X, threshold):
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds
# ****************************thresholds sigmoid output prediction***********************

# **********************************calculates dice,dice neg and dice pos LOSS************************************
def metric(probability, truth, threshold=0.5, reduction='none'):
    batch_size = len(truth)
    with torch.no_grad():                                   #NEGATIVE means no label POSITIVE have labels (IMAGE IMBALANCE ARE MANAGED).
        probability = probability.view(batch_size, -1)      #all the predictred values are converted to list
        truth = truth.view(batch_size, -1)                  #all the ground truth values are also converted into list
        assert(probability.shape == truth.shape)
        p = (probability > threshold).float()               #threhold is provided for list of predicte values to ezactly 0/1.
        t = (truth > 0.5).float()                           #threshold is provided for list of ground truth values
        t_sum = t.sum(-1)                                   #after thresholding ground truth values are added
        p_sum = p.sum(-1)                                   #after thresholding predicted values are added
        neg_index = torch.nonzero(t_sum == 0)               #in a batch wherever GT sum is 0 those are negative index
        pos_index = torch.nonzero(t_sum >= 1)               #in a batch wherever GT sum is positive those are positive index
        dice_neg = (p_sum == 0).float()                     #in a batch wherever predicted sum is 0 is dice neg
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))        #using the formula dice pos is calculated (CLASS IMBALANCE WILL BE MANAGED)
        dice_neg = dice_neg[neg_index]                      #using GT index for negatives, negative predicted values are fetched from dice neg.
        dice_pos = dice_pos[pos_index]                      #using GT index for positives, positive predicted values are fetched from dice neg.
        dice = torch.cat([dice_pos, dice_neg])              #both dice neg and dice pos are concateneted.
        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0) #negative mean for batch
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0) #positive mean for batch
        dice = dice.mean().item()                           #pos neg combine mean
        num_neg = len(neg_index)
        num_pos = len(pos_index)
    return dice, dice_neg, dice_pos, num_neg, num_pos
# **********************************calculates dice,dice neg and dice positive************************************

# **********************************Meter keep track of iou and dice scores throughout an epoch************************************
class Meter:
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)                                                   #convert the outputs to 0/1 ()approx.
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        preds = predict(probs, self.base_threshold)                                     #threhold is provided for list of predicte values to ezactly 0/1 & uint8.
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)                    #combine pos neg mean for entire dataset
        dice_neg = np.mean(self.dice_neg_scores)                 #neg mean for entire dataset
        dice_pos = np.mean(self.dice_pos_scores)                 #pos mean for entire dataset
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)                        #iou mean for entire dataset.
        return dices, iou
# **********************************Meter keep track of iou and dice scores throughout an epoch************************************

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()            #retun dices and IOU
    dice, dice_neg, dice_pos = dices            #return all relavent dices
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (epoch_loss, iou, dice, dice_neg, dice_pos))
    return dice, iou

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:                                            #classes is 1 ,because labels are one hot encoded and all the places it is 1.
        label_c = label == c                                     # T/F in GT all the places where label=1
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c                                       #T/F in prediced all the places where label is 1.
        intersection = np.logical_and(pred_c, label_c).sum()     #Intersection
        union = np.logical_or(pred_c, label_c).sum()             #Union
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]

def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs)                                            # copy is imp
    labels = np.array(labels)                                           # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))     #take 6 preds and 6 outs and send to compute iou after calulate mean
    iou = np.nanmean(ious)                                              #for entire dataset calculate iou mean
    return iou
model = Unet("resnet18", encoder_weights="imagenet", classes=4, activation=None)


class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model):
        self.num_workers = 4
        self.batch_size = {"train": 8, "val": 1}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = 20
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {phase: provider(data_folder=data_folder,df_path=train_df_path,phase=phase,mean=(0.485, 0.456, 0.406),
                                            std=(0.229, 0.224, 0.225),batch_size=self.batch_size[phase],num_workers=self.num_workers,) for phase in self.phases}
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)       #calculate the binary cross enrtopy from logits.
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")            #True/False if true BN is not calculated.
        dataloader = self.dataloaders[phase]        #List of dataloader for train & validation.
        running_loss = 0.0
        total_batches = len(dataloader)             #total number of batch (10054/8)
        # tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):  # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            loss, outputs = self.forward(images, targets)       #loss output will be mean per pixel (loss/(8*4*256*1600))
            loss = loss / self.accumulation_steps               #mean per pixel per 32 samples. (loss/4*8*4*256*1600)
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:    #after reaching 32 samples optimization takes place.
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)                     #calculate iou,dice, dice_pos,dice_neg
        # tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches   #total running loss perpixel per total batch
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)           #return dice & iou
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model.pth")
            print()

sample_submission_path = '/media/shashank/New Volume/severstal-steel-defect-detection/sample_submission.csv'
train_df_path = '/media/shashank/New Volume/severstal-steel-defect-detection/train.csv'
data_folder = "/media/shashank/New Volume/severstal-steel-defect-detection"
test_data_folder = "/media/shashank/New Volume/severstal-steel-defect-detection/test_images"

model_trainer = Trainer(model)
model_trainer.start()
