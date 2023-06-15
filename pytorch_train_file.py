#%%
import glob
import torch
from torch.utils.data import Dataset, DataLoader, sampler
import numpy as np
from osgeo import gdal
import segmentation_models_pytorch as smp
from torch import nn
import time
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
import rasterio
import cv2
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import CSVLogger

# Set the CUDA device to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)
torch.set_float32_matmul_precision('medium')
# %%
class myDataset(Dataset):
    def __init__(self, data_files,mean,std,resize=False):
        super().__init__()
        self.files = data_files   
        self.mean=mean
        self.std=std  
        self.num_bands=len(mean) 
        self.resize=resize                                         
    def __len__(self):
        
        return len(self.files)
     
    def open_as_array(self, idx):
        img_file=self.files.loc[idx,'img']    
        raw_rgb = rasterio.open(img_file).read()
        raw_rgb=raw_rgb/np.iinfo(raw_rgb.dtype).max

        raw_rgb=raw_rgb.astype(np.float32)
        #for i in range(self.num_bands):
        #    raw_rgb[i,:,:]=(raw_rgb[i,:,:]-self.mean[i])/self.std[i]
        # normalize
        if self.resize:
            bands,rows,cols=raw_rgb.shape
            rows_new=int(int(rows/32)/4)*32
            cols_new=int(int(cols/32)/4)*32
            raw_rgb=raw_rgb.transpose((1,2,0))
            raw_rgb=cv2.resize(raw_rgb,(rows_new,cols_new))
            raw_rgb=raw_rgb.transpose((2,0,1))
        return raw_rgb
    

    def open_mask(self, idx, add_dims=False):
        mask_file=self.files.loc[idx,'label']
        raw_mask = np.squeeze(rasterio.open(mask_file).read()) 
        if self.resize:
            rows,cols=raw_mask.shape
            rows_new=int(int(rows/32)/4)*32
            cols_new=int(int(cols/32)/4)*32
            raw_mask=cv2.resize(raw_mask,(rows_new,cols_new))       
        return raw_mask
    
    def __getitem__(self, idx):
        
        x = torch.tensor(self.open_as_array(idx), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.long)
        
        return x, y
class myModel(pl.LightningModule):
    
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.Unet(
    encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=out_classes,                      # model output channels (number of classes in your dataset)
       )
        self.loss_fun=nn.CrossEntropyLoss()
    def forward(self, image):
        mask = self.model(image)
        return mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self.model(x)
        loss = self.loss_fun(outputs, y)
        self.log('train_loss', loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.model(x)
        loss = self.loss_fun(outputs, y)
        self.log('val_loss', loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)
csv_logger = CSVLogger(
    save_dir='/data/taoliu/taoliufile/NASA_LULAS_2023/',
    name='semantic_segmentation_fire_lightning_divideuint16'
)

mean=[0,0,0,0,0,0,0,0,0,0]
std=[1,1,1,1,1,1,1,1,1,1]  
# Create an instance of your custom dataset
data_file="/data/taoliu/taoliufile/fire/activefire/activefire/dataset/masks/semantic_fire_voting.csv"
data_files=pd.read_csv(data_file)
dataset = myDataset(data_files,mean,std,resize=False)
dataset_length = len(dataset)
train_length = int(0.8* dataset_length)
valid_length = dataset_length - train_length

train_ds, valid_ds = torch.utils.data.random_split(dataset, [train_length, valid_length])
train_loader = DataLoader(train_ds, batch_size=100, shuffle=True,num_workers=20)
val_loader = DataLoader(valid_ds, batch_size=100, shuffle=False,num_workers=20)
model = myModel("resnet50", "resnet50", in_channels=10, out_classes=2)
# training
trainer = pl.Trainer(precision='16-mixed', logger=csv_logger,limit_train_batches=0.5,max_epochs=50,default_root_dir="/data/taoliu/taoliufile/NASA_LULAS_2023/fire_lightning_output_divideuint16/")
#trainer.resume_from_checkpoint('"/data/taoliu/taoliufile/NASA_LULAS_2023/lightning_logs/version_6/checkpoints/epoch=6-step=588.ckpt"')
trainer.fit(model, train_loader, val_loader)
