# -*- coding: utf-8 -*-
"""pytorch_4.1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b8ogU6hSUhBUcsLEGlKvN-BSaA-1JJlu
"""

from torchvision import datasets, transforms, utils

from torch.utils import data

import matplotlib.pyplot as plt
import numpy as np

transform  = transforms.Compose([transforms.ToTensor()]) #텐서로 바꿔줌

trainset = datasets.FashionMNIST(
    root = './.data/',
    train = True,
    download = True,
    transform = transform
)

testset = datasets.FashionMNIST(
    root = './.data/',
    train = False,
    download = True,
    transform = transform
)

batch_size = 16

train_loader = data.DataLoader(
    dataset = trainset,
    batch_size = batch_size
)

test_loader = data.DataLoader(
    dataset = testset,
    batch_size = batch_size
)

dataiter = iter(train_loader)
images, labels = next(dataiter)

img = utils.make_grid(images, padding = 0) #padding에 따라 사진의 너비와 높이가 달라짐, 아마도 사진마다 padding을 추가하는거 같음
npimg = img.numpy() #img는 파이토시 텐서이기 때문에 numpy 행렬로 바꿔줌
plt.figure(figsize=(10,7))
plt.imshow(np.transpose(npimg,(1,2,0)))
plt.show()

print(labels)

CLASSES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

for label in labels :
  index = label.item() #label은 tensor(9), label.item()하면 9가 나옴
  print(CLASSES[index])

idx = 1
item_img = images[idx]
item_npimg = item_img.squeeze().numpy() #맷플롯립에서 이용이 가능한 넘파이 행렬로 만듬
plt.title(CLASSES[labels[idx].item()])
plt.imshow(item_npimg, cmap='gray')
plt.show()

