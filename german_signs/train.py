import torch
import torch.nn as nn

import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

from open_images import readTrafficSigns
from models import FCN, ResNet, DenseNet


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    trainX, trainY = readTrafficSigns('/data/home/n.kostin/datasets/german_signs/GTSRB/Final_Training/Images')
    criterion = torch.nn.CrossEntropyLoss()
    model = DenseNet(layers_config=(6, 12, 24, 16), bn_size=4, growth_rate=32, device=device)  # ResNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)

    print("[INFO] Training")
    print(device)
    model.train(True)
    batch_size = 256
    losses = []
    episodes = 6000

    for episode in range(episodes + 1):
        batchidx = np.random.randint(0, trainX.shape[0], batch_size)
        batchX = trainX[batchidx].to(device)
        batchY = trainY[batchidx].to(device)
        pred_logits = model(batchX)
        loss = criterion(pred_logits, batchY)
        losses.append(loss.cpu().data.numpy())

        loss.backward()
        opt.step()
        opt.zero_grad()
        if episode % 100 == 0 and episode != 0:
            print(f"-> {episode} episode, loss = {losses[-1]}")
        if episode > 3000:
            opt = torch.optim.Adam(model.parameters(), lr=0.00001)
        if episode % 1000 == 0 and episode != 0:
            torch.save(model.state_dict(), './densenet_1.pt')
