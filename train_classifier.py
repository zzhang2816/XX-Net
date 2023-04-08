import torch
import clip
from PIL import Image
import json
import numpy as np
from torch.utils import data
from glob import glob
from torchvision import transforms
from torch import nn, optim
from time import time
torch.manual_seed(42)

class my_dataset(data.Dataset):
    def __init__(self, preprocess, mode):
        pos_data = glob("data_postprocess/pos_data/*.png")
        neg_data = glob("data_postprocess/neg_data/*.png")
        if mode == 'train':
            self.image_paths = pos_data[:6000] + neg_data[:6000]
            self.labels = torch.zeros(len(self.image_paths),dtype=int)
            self.labels[:len(pos_data[:6000])] = 1
        else:
            self.image_paths = pos_data[6000:] + neg_data[6000:]
            self.labels = torch.zeros(len(self.image_paths),dtype=int)
            self.labels[:len(pos_data[6000:])] = 1
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,index):
        img = Image.open(self.image_paths[index]).convert('RGB').resize((32,72))
        img = self.preprocess(img).to(device)
        label = self.labels[index].to(device)
        return img, label
    
class test_dataset(data.Dataset):
    def __init__(self, preprocess):
        self.image_paths = sorted(glob("data_postprocess/patches/*.png"))
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,index):
        img = Image.open(self.image_paths[index]).convert('RGB').resize((32,72))
        img = self.preprocess(img).to(device)
        return img,self.image_paths[index]
    
def train():
    # set essentials
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 25)
    torch.set_grad_enabled(True)
    model.train()
    # training
    for epoch in range(epochs):
        epoch_loss = 0
        timestamp = time()
        for i, (img, label) in enumerate(train_loader):
            outputs = model(img)
            l=loss(outputs,label)
            epoch_loss += l.cpu().detach().numpy()
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            scheduler.step()

            if i%50 == 0:
                print(f"loss [{i}/{len(train_loader)}]: {l.data.item():.4f}")
                

        print(f'Epoch: {epoch}; loss: {epoch_loss/len(train_loader)}; time: {time()-timestamp}')
        torch.save(model, f"data_postprocess/ckpts/epoch_{epoch}.pth")
        
def eval():
    model.eval()
    corr_sum = 0
    total_sum = 0
    for i, (img, label) in enumerate(eval_loader):
        with torch.no_grad():
            outputs = model(img)
            corr_sum += (outputs.argmax(axis=-1) == label).sum()
            total_sum += len(label)
            print(f"accuracy: {corr_sum/total_sum*100}")
    print(f"accuracy: {corr_sum/total_sum*100}")
    

# set parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
epochs = 5
n_classes = 2

# set up model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights="ResNet50_Weights.DEFAULT")
model.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True)

model = model.to(device)

# set dataset
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = my_dataset(preprocess,"train")
eval_dataset = my_dataset(preprocess,"eval")
test_dataset = test_dataset(preprocess)

train_loader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
eval_loader = data.DataLoader(eval_dataset,batch_size=batch_size,shuffle=True)
test_loader = data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

# train model
train()
eval()



