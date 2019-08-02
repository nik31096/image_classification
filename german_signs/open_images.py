# import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from cv2 import LUT, convertScaleAbs
from torchvision import transforms


def gamma_adjust(img, gamma):
    lut = np.zeros((1, 256), dtype=np.int)
    for i in range(256):
        lut[0, i] = np.clip(np.power(i / 255, gamma)*255, a_min=0, a_max=255)

    out = LUT(img, lut)

    return out


def readTrafficSigns(rootpath):
    resizer = transforms.Compose([
        transforms.Resize((50, 50))
    ])

    images, labels = [], []
    shapes = []
    # loop over all 43 classes
    for c in range(0, 43):
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        with open(prefix + 'GT-' + format(c, '05d') + '.csv') as f:
            data = f.read().split('\n')[1:-1]
        # loop over all images in current annotations file
        for row_ in data:
            # TODO: make use of ROI data from csv file
            row = row_.split(';')
            image = np.array(resizer(Image.open(prefix + row[0])))
            if int(row[7]) == 27:
                img = convertScaleAbs(image, alpha=1.5, beta=6)
                images.append(torch.FloatTensor(img.reshape(img.shape[2], img.shape[0], img.shape[1])))
                labels.append(int(row[7]))
            img = gamma_adjust(image, 0.4) # convertScaleAbs(image, alpha=1.5, beta=6)
            images.append(torch.FloatTensor(img.reshape(img.shape[2], img.shape[0], img.shape[1])))
            labels.append(int(row[7]))
            shapes.append(image.size)
            images.append(torch.FloatTensor(image.reshape(image.shape[2], image.shape[0], image.shape[1])))
            labels.append(int(row[7]))

    return torch.stack(images), torch.LongTensor(labels)


if __name__ == '__main__':
    path2train_data = '../../../datasets/german_signs/GTSRB/Final_Training/Images'
    trainImages, trainLabels = readTrafficSigns(path2train_data)

    print(trainImages.shape, trainLabels.shape)
