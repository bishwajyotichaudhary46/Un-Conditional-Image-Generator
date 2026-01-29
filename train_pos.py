import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import  DataLoader,random_split
from Generator.Dataset import CustomDataset, CustomNoiseImageDataset
from Generator.NN import NNDenoiser
from Generator.add_noise import add_noise
from Generator.positional_encoding import positional_encoding
import csv
import os
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch


transform = T.Compose([
    T.Resize(128),
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3) 
])

dataset = CustomDataset("data/", transform=transform, add_noise=add_noise)

dataset = CustomNoiseImageDataset(dataset)

#indices = np.random.choice(len(dataset), size=5000, replace=False)
#small_dataset = Subset(dataset, indices)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=1
)
val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=1
)

log_file = "artifacts/pos/training_log_nn.csv"
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"   # still OK, but slow
model = NNDenoiser().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-6
)
loss_fn = nn.MSELoss()

num_epochs = 500
positional_encoded = positional_encoding(10000, 4096).to(device)
best_val = float("inf")
patience = 5
counter = 0
idle_device = "cpu"
if idle_device:
    to_idle = lambda x: x.to(idle_device)
else:
    to_idle = lambda x: x

for epoch in range(1,num_epochs+1):
    model.train()
    train_loss = 0
    for images, t, noise, t_int in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        images = images.to(device)
        t = t.to(device)
        noise = noise.to(device)
        t_int = t_int.to(device)
        B, C, H, W = images.shape

        t_i = t_int.flatten()
        t_i = positional_encoded[t_i,:].view(B, -1)
        optimizer.zero_grad()
        pred_grad = model(images,t_i)
        loss = ((pred_grad - (noise - images))**2).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)

    # validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images,t,noise, t_int in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            images = images.to(device)
            t = t.to(device)
            noise = noise.to(device)
            t_int = t_int.to(device)
            B, C, H, W = images.shape

            t_i = t_int.flatten()
            t_i = positional_encoded[t_i,:].view(B, -1)

            pred_grad = model(images,t_i)
            loss = ((pred_grad - (noise - images))**2).mean()
            val_loss += loss.item() * images.size(0)

    val_loss /= len(val_loader.dataset)
    scheduler.step(val_loss)
    

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    # log CSV
    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_loss])

    # save checkpoint 
    torch.save(model.state_dict(), f"artifacts/pos/model_{epoch}.pt")
    
    # early stopping
    if val_loss < best_val:
        best_val = val_loss
        counter = 0
        torch.save(model.state_dict(), "artifacts/pos/best.pt")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

