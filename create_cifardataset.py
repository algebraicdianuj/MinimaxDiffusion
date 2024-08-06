import torch
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import rotate as scipyrotate
from torchvision import datasets, transforms
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
im_size = (32, 32)
num_classes = 10
num_subclasses = 100
batch_size = 128
im_batch= 1000
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
transform = transforms.Compose([transforms.ToTensor(),
                                #  transforms.Normalize(mean=mean, std=std)
                                 ])
dst_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform) # no augmentation
train_loader = torch.utils.data.DataLoader(dst_train, batch_size=batch_size, shuffle=True)


original_class_dict={0: 'airplane',
                     1: 'automobile',
                     2: 'bird',
                     3: 'cat',
                     4: 'deer',
                     5: 'dog',
                     6: 'frog',
                     7: 'horse',
                     8: 'ship',
                     9: 'truck'}


class_dict = {'airplane': 'n02687172',
                'automobile': 'n04037443',
                'bird': 'n01534433',
                'cat': 'n02123045',
                'deer': 'n02415577',
                'dog': 'n02107683',
                'frog': 'n01644373',
                'horse': 'n02398521',
                'ship': 'n02692877',
                'truck': 'n03345487'}


''' organize the real dataset '''
images_all = []
labels_all = []
indices_class = [[] for c in range(num_classes)]



images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
labels_all = [dst_train[i][1] for i in range(len(dst_train))]
for i, lab in enumerate(labels_all):
    indices_class[lab].append(i)
images_all = torch.cat(images_all, dim=0).to(device)
labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)


def get_images(c, n): # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return images_all[idx_shuffle]


if os.path.exists('cifar10_dataset'):
    os.system('rm -r cifar10_dataset')
os.makedirs('cifar10_dataset', exist_ok=True)


for c in range(num_classes):
    img_batch_c = get_images(c, im_batch)
    original_class=original_class_dict[c]
    modified_class=class_dict[original_class]
    os.makedirs(f'cifar10_dataset/{modified_class}', exist_ok=True)
    for k in range(im_batch):
        torchvision.utils.save_image(img_batch_c[k], f'cifar10_dataset/{modified_class}/img_{k}.png')


fold_names=os.listdir('cifar10_dataset')

with open('misc/cifar10.txt', 'w') as f:
    for item in fold_names:
        f.write(f"{item}\n")

    
