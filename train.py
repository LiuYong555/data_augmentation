# run train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8

import pdb
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from util.misc import CSVLogger
from util.cutout import Cutout
from util.mixup import mixup_data, mixup_criterion
from util.cutmix import rand_bbox

from model.resnet import ResNet18
from model.wide_resnet import WideResNet
import warnings

model_options = ['resnet18', 'wideresnet']
dataset_options = ['cifar10', 'cifar100', 'svhn']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar100',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
### cutout相关参数
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
### mixup相关参数
parser.add_argument('--mixup', action='store_true', default=False,
                    help='apply mixup')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
### cutmix相关参数
parser.add_argument('--cutmix',action='store_true',default=False,
                    help='apply cutmix')
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    test_id = args.dataset + '_' + args.model
    if args.cutout:
        test_id = test_id + '_' + 'cutout'
    elif args.mixup:
        test_id = test_id + '_' + 'mixup'
    elif args.cutmix:
        test_id = test_id + '_' + 'cutmix'
    else:
        pass


    print(args)


    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_transform = transforms.Compose([])
    if args.data_augmentation:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))


    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])



    num_classes = 100
    train_dataset = datasets.CIFAR100(root='data/',
                                        train=True,
                                        transform=train_transform,
                                        download=True)

    test_dataset = datasets.CIFAR100(root='data/',
                                        train=False,
                                        transform=test_transform,
                                        download=True)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)

    if args.model == 'resnet18':
        cnn = ResNet18(num_classes=num_classes)


    cnn = cnn.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                    momentum=0.9, nesterov=True, weight_decay=5e-4)


    scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

    filename = 'logs/' + test_id + '.csv'
    csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)

    def train(loader):
        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()

            cnn.zero_grad()
            pred = cnn(images)

            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()
            cnn_optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total

            progress_bar.set_postfix(
                xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
                acc='%.3f' % accuracy)

        return accuracy

    def train_mixup(loader):
        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.
        train_loss = 0.
        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()


            images, labels_a, labels_b, lam = mixup_data(images, labels, args.alpha, args.cuda)
            images, labels_a, labels_b = map(Variable, (images, labels_a, labels_b))
            pred = cnn(images)
            xentropy_loss = mixup_criterion(criterion, pred, labels_a, labels_b, lam)
            #train_loss += xentropy_loss.data[0]
            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (lam * predicted.eq(labels_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(labels_b.data).cpu().sum().float())

            cnn_optimizer.zero_grad()
            xentropy_loss.backward()
            cnn_optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            #pred = torch.max(pred.data, 1)[1]
            #total += labels.size(0)
            #correct += (pred == labels.data).sum().item()
            accuracy = correct / total

            progress_bar.set_postfix(
                xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
                acc='%.3f' % accuracy)

        return accuracy

    def train_cutmix(loader):
        cnn.train()
        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()

            r = np.random.rand(1)
            if args.beta > 0 and r < args.cutmix_prob:
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                labels_a = labels
                labels_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                # compute output
                pred = cnn(images)
                loss = criterion(pred, labels_a) * lam + criterion(pred, labels_b) * (1. - lam)
            else:
                # compute output
                pred = cnn(images)
                loss = criterion(pred, labels)

            cnn_optimizer.zero_grad()
            loss.backward()
            cnn_optimizer.step()

            xentropy_loss_avg += loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total

            progress_bar.set_postfix(
                xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
                acc='%.3f' % accuracy)

        return accuracy


    def test(loader):
        cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
        correct = 0.
        total = 0.
        for images, labels in loader:
            images = images.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                pred = cnn(images)

            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        val_acc = correct / total
        cnn.train()
        return val_acc


    for epoch in range(args.epochs):

        if args.cutmix:
            accuracy = train_cutmix(train_loader)
        elif args.mixup:
            accuracy = train_mixup(train_loader)
        else:
            accuracy = train(train_loader)

        '''
        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()

            if args.mixup:
                images, labels_a, labels_b, lam = mixup_data(images, labels, args.alpha, args.cuda)
                images, labels_a, labels_b = map(Variable, (images, labels_a, labels_b))
                pred = cnn(images)
                xentropy_loss = mixup_criterion(criterion, pred, labels_a, labels_b, lam)
                xentropy_loss.backward()
                cnn_optimizer.step()

            else:

                cnn.zero_grad()
                pred = cnn(images)

                xentropy_loss = criterion(pred, labels)
                xentropy_loss.backward()
                cnn_optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total

            progress_bar.set_postfix(
                xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
                acc='%.3f' % accuracy)
        '''
        test_acc = test(test_loader)
        tqdm.write('test_acc: %.3f' % (test_acc))

        scheduler.step(epoch)  # Use this line for PyTorch <1.4
        # scheduler.step()     # Use this line for PyTorch >=1.4

        row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
        csv_logger.writerow(row)

    torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
    csv_logger.close()
