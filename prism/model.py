from . import tilify
from torch.utils.data import DataLoader
from .datawrappers import FireCastDataset
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import torch.optim as optim
from . import trainingroutine
from datetime import datetime, timedelta
import torch
import os 

class FireCastModel():

    def __init__(self):
        self.model = self.get_model()


    def get_loss(self):
        #Our loss functions; binary cross entropy combined with Dice loss
        loss_fn = nn.BCEWithLogitsLoss()  # Combines sigmoid + binary cross entropy
        dice_loss = DiceLoss(mode='binary')
        loss_fn = lambda outputs, targets: 0.5 * nn.BCEWithLogitsLoss()(outputs, targets) + 0.5 * dice_loss(outputs, targets)
        return loss_fn
    
    def get_optim(self):
        #Optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=5e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        return (optimizer,scheduler)
    
    def tile_raster(self,d1,d2):
        #Data in rough numpy format
        t = tilify.tilify(date1=d1,date2=d2)
        
        return t.gen()
    
    def get_dataset(self,d1,d2):
        #Convert to FireCast Wrapper Dataset, and into the DataLoader
        X_train, X_test, X_val, Y_train, Y_test, Y_val = self.tile_raster(d1,d2)
        train_dataset = FireCastDataset(x_array=X_train, y_array=Y_train)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        val_dataset = FireCastDataset(x_array=X_val, y_array=Y_val)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

        return (train_loader,val_loader)
    
    def get_model(self):

        model = smp.UnetPlusPlus(

            encoder_name="resnet34",        # You can try efficientnet-b0 or others
            encoder_weights="imagenet",     # Use pretrained weights
            in_channels=9,  # Number of channels in your input tiles
            classes=1,                      # Binary classification
            activation=None                 # We'll apply sigmoid manually
        )

       
        if os.path.exists('best_model.pth'):
            print("✅ Model exists!")
            model.load_state_dict(torch.load('best_model.pth'))
        else:
            print("❌ Model does not exist.")

        return model
    
    
    def train(self,date):
        date2 = date

        # Convert to datetime object
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        # Add one day
        next_day = date_obj - timedelta(days=1)
        date1 = next_day.strftime("%Y-%m-%d")

        loss_fn = self.get_loss()
        model = self.get_model()
        (train_loader,val_loader)  = self.get_dataset(date1,date2)
        (optimizer,scheduler) = self.get_optim()

        trainingroutine.FireCastTrainer(model,train_loader,val_loader,loss_fn,optimizer,scheduler)

