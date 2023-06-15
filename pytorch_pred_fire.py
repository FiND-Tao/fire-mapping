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
# Set the CUDA device to use
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
        self.log('train_loss', loss)
        return loss
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.model(x)
        loss = self.loss_fun(outputs, y)
        self.log('val_loss', loss)

def open_as_array(img_file,mean,std,resize=False):
    raw_rgb = rasterio.open(img_file).read()
    raw_rgb=raw_rgb/np.iinfo(raw_rgb.dtype).max

    raw_rgb=raw_rgb.astype(np.float32)
    #for i in range(self.num_bands):
    #    raw_rgb[i,:,:]=(raw_rgb[i,:,:]-self.mean[i])/self.std[i]
    # normalize
    if resize:
        bands,rows,cols=raw_rgb.shape
        rows_new=int(int(rows/32)/4)*32
        cols_new=int(int(cols/32)/4)*32
        raw_rgb=raw_rgb.transpose((1,2,0))
        raw_rgb=cv2.resize(raw_rgb,(rows_new,cols_new))
        raw_rgb=raw_rgb.transpose((2,0,1))
    return raw_rgb

mean=[0,0,0,0,0,0,0,0,0,0]
std=[1,1,1,1,1,1,1,1,1,1]  
# Create an instance of your custom dataset
data_file="/data/taoliu/taoliufile/fire/activefire/activefire/dataset/masks/predict.csv"
data_files=pd.read_csv(data_file)
dataset = myDataset(data_files,mean,std,resize=False)

val_loader = DataLoader(dataset, batch_size=400, shuffle=False,num_workers=20)
model = myModel("resnet50", "resnet50", in_channels=10, out_classes=2)
model = model.load_from_checkpoint("/data/taoliu/taoliufile/NASA_LULAS_2023/fire_lightning_output_divideuint16/lightning_logs/version_5/checkpoints/epoch=49-step=4200.ckpt",arch="resnet50", encoder_name="resnet50", in_channels=10, out_classes=2)
model.eval()
#checkpoint = torch.load("/data/taoliu/taoliufile/NASA_LULAS_2023/fire_lightning_output_divideuint16/lightning_logs/version_5/checkpoints/epoch=49-step=4200.ckpt")
#model.load_state_dict(checkpoint["state_dict"])
#model.eval()
def predict(tensor, model):
    yhat = model(tensor.unsqueeze(0))
    yhat = yhat.clone().detach()
    return yhat

file="/data/taoliu/taoliufile/fire/activefire/activefire/dataset/images/patches/LC08_L1TP_106069_20200828_20200828_01_RT_p00768.tif"
#file="/data/taoliu/taoliufile/fire/activefire/activefire/Selected_Bands_rows_coms.tif"
#file="/data/taoliu/taoliufile/fire/activefire/activefire/Selected_Bands_rows_coms_mtbs.tif"#AZ3358211070120200818


x=open_as_array(file,mean,std,resize=False)
x=np.expand_dims(x, 0)

print(x.shape)
x=torch.from_numpy(x).cuda()
#model = myModel("resnet50", "resnet50", in_channels=10, out_classes=2)
prediciton = model(x)    
print(prediciton.shape)
print('done')
_, labels = torch.max(prediciton, dim=1)
labels=np.squeeze(labels.cpu().numpy())
print(labels.shape)
print(np.sum(labels))
plt.imshow(labels)


# %%
img_path=file
y_pred_map=labels
no_data_value=-1
output_folder="/data/taoliu/taoliufile/NASA_LULAS_2023/NASA_RIA_Output/"
output_path=os.path.join(output_folder,'cnn_pred_line_'+os.path.basename(img_path))

driver = gdal.GetDriverByName('GTiff')  # Choose the appropriate driver for your desired image format
rows, cols = y_pred_map.shape  # Get the shape of the NumPy array
bands = 1  # Number of bands in the image (e.g., 1 for grayscale, 3 for RGB)
output_dataset = driver.Create(output_path, cols, rows, bands, gdal.GDT_UInt16)  # Adjust the data type (GDT) if necessary
output_dataset.GetRasterBand(1).WriteArray(y_pred_map)  # Adjust the band number if necessary (1 for grayscale)

dataset = gdal.Open(img_path)
geotransform = dataset.GetGeoTransform()
projection = dataset.GetProjection()
output_dataset.GetRasterBand(1).SetNoDataValue(no_data_value)

output_dataset.SetGeoTransform(geotransform)  # Adjust the geotransform parameters if you have spatial information
output_dataset.SetProjection(projection)  # Set the appropriate projection information
output_dataset = None  # Close the dataset to flush the data to the disk
        
