import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
import numpy as np
from GoogleNet import GoogLeNet
from cifar import load_data

_ ,_ , test_loader, _ = load_data()

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Đường dẫn đến file chứa model_state_dict đã lưu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = ".\model_round_3.pth"
model = GoogLeNet().to(device)
# Nạp model_state_dict vào mô hình mới
model.load_state_dict(torch.load(model_path))
model.eval()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
'''
dataiter = iter(test_loader)
images, labels = next(dataiter)

print(images.shape, labels.shape)
# show images
imshow(torchvision.utils.make_grid(images))
'''

#x = torch.randn(1,3,32,32)
#y = model(x)
#print(torch.max(y,1)[1])
#print(y)
batch_size = 4
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    #n_class_correct = [0 for i in range(10)]
    #n_class_samples = [0 for i in range(10)]
    #for i, images, labels in (tqdm(test_loader)):
    for i, (images, labels) in enumerate (tqdm(test_loader)):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        '''for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
'''

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    #for i in range(10):
       # acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        #print(f'Accuracy of {classes[i]}: {acc} %')