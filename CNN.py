#Buris L.H
#https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
import os
import torch
import argparse
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch Pré_Treined')
parser.add_argument("--dataset", default="", type=str, required=True,
                    help="informação.")
parser.add_argument("--kfold", default="", type=str, required=False,
                    help="K-fold id.")
parser.add_argument("--train", required=True,
                    type=str, help="Path to data parent directory.")
parser.add_argument("--test", required=True,
                    type=str, help="Path to data parent directory.")
parser.add_argument('--batch_size',
                    default=32, type=int, help='batch_size')
parser.add_argument('--epoch', default=200,
                    type=int, help='you need in the epoch')
parser.add_argument('--n_classe', default=3, type=int, help='you need')
parser.add_argument('--model', default="ResNet50",
                    type=str, help='model type (default: ResNet50)')
parser.add_argument('--num-workers', type=int,
                    default=2, help='number of workers')
parser.add_argument("--input_size", default=256,
                    type=int, help="input size img.")


args = parser.parse_args()

name = "CNN_"+args.dataset+"_"+args.model
save_dir = "results"
seed = name+"_"+args.dataset+"_"+args.kfold
print('seed==>',seed)

writer = SummaryWriter()

result_model = list()
result_model.append("SEED::  "+str(seed)+ "\n")
result_model.append("============================= \n")

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((args.input_size,args.input_size)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((args.input_size,args.input_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset_xl = datasets.ImageFolder(os.path.abspath(args.train), transform=transform_train)
train_loader_xl = torch.utils.data.DataLoader(trainset_xl,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=0)

testset = datasets.ImageFolder(os.path.abspath(args.test), transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)



print('==> Building model..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#net = models.resnet50(pretrained=True)
#net = torchvision.models.vgg16(pretrained=True)
net = torchvision.models.vgg19(pretrained=True)
net.classifier[6] = nn.Linear(4096,args.n_classe)

#modules = list(list_net)
#modules.append(nn.Flatten())
#modules.append(nn.Linear(4096,args.n_classe))
#net = nn.Sequential(*modules)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion_cnn = nn.CrossEntropyLoss()
optimizer_cnn = optim.Adam(net.parameters(), lr=0.0001)

def save_model(model):
    torch.save(model.state_dict(), save_dir + "/"+seed+"model.pt")

def CreateDir(path):
        try:
                os.mkdir(path)
        except OSError as error:
                print(error)


CreateDir(save_dir)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader_xl):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_cnn.zero_grad()
        outputs = net(inputs)
        loss = criterion_cnn(outputs, targets)
        loss.backward()
        optimizer_cnn.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        writer.add_scalar('Training/ACC_',100.*correct/total, (epoch*len(train_loader_xl.dataset)/args.batch_size)+batch_idx)
        writer.add_scalar('Training/loss_',train_loss/(batch_idx+1),(epoch*len(train_loader_xl.dataset)/args.batch_size)+batch_idx)
    print('\n %d',correct/total*100)
    writer.add_scalar('Training/ACC',correct/total*100, epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    print("ACC_test",acc)
    return acc

def main():
  for epoch in range(0, args.epoch):
    train(epoch)
  acc = test(epoch)

  save_model(net)

  result_model.append("-Arquitetura::  "+args.model+ "\n")
  result_model.append("-ACC_Test::  "+str(acc)+ "\n")

  arquivo = open(name+".txt", "a")
  arquivo.writelines(result_model)
  arquivo.close()

if __name__ == '__main__':
    main()
