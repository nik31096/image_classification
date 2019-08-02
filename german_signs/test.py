import torch
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
from PIL import Image
from models import ResNet

from pprint import pprint
import time
from sys import exit
from collections import Counter

from sklearn.metrics import f1_score, classification_report as c_r


def read_test(test_dir):
    resizer = transforms.Compose([
        transforms.Resize((50, 50))
    ])
    with open(f'{test_dir}/Test.csv') as f:
        data = f.read().split('\n')

    images, labels = [], []
    for row in data[1:-1]:
        image = np.array(resizer(Image.open(test_dir + '/' + row.split(',')[-1].split('/')[1])))
        images.append(torch.FloatTensor(image.reshape(image.shape[2], image.shape[0], image.shape[1])))
        labels.append(int(row.split(',')[-2]))

    return torch.stack(images), torch.LongTensor(labels)


X, Y = read_test('/data/home/n.kostin/datasets/german_signs/test')

model = ResNet()
model.load_state_dict(torch.load('./resnet_model_adam_0005_128_2k_adam_00005_2k_gamma_04_a_15_b_6.pt'))

# pprint([param.nelement() for param in model.parameters()])  # TODO: add model parameters print

print("[INFO] Testing")
model.eval()
start = time.time()
preds = list(torch.argmax(F.softmax(model(torch.FloatTensor(X)), dim=1), dim=1).cpu().data.numpy())
print(f"Network inference time is {time.time() - start}")
print("F1-score micro: ", f1_score(Y, preds, average='micro'))
print("F1-score macro: ", f1_score(Y, preds, average='macro'))

print(c_r(Y, preds, target_names=[f"class {str(i)}" for i in range(1, 44)]))

