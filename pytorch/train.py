import os
import sys
import pathlib
import time
from datetime import datetime
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as T

from config import parse_arguments
from models.resnet import resnet50



def train(args, train_loader, test_loader, net, device, writer, log_dir, checkpoint_dir):
    step = 0
    net.train()

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total = 0
        correct = 0
        train_loss = 0

        for imgs, labels in iter(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = net(imgs)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            acc = 100.*correct/total

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.print_freq == 0 and step != 0:
                print('Epoch: {:2d}, Step: {:5d}, Loss: {:5f}, Acc: {:5f}'.format(epoch, step, loss.item(), acc))
                writer.add_scalar('Train loss', loss, step)
                writer.add_scalar('Train acc', acc, step)
            step += 1


        if epoch %2 == 0 and epoch != 0:
            test(args, test_loader, net, device, writer, log_dir, checkpoint_dir, step)
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, str(step)) + '.pth')

def test(args, data_loader, net, device, writer, log_dir, checkpoint_dir, step):
    net.eval()

    correct = 0
    total = 0
    acc = 0

    with torch.no_grad():
        for imgs, labels in iter(data_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = net(imgs)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        acc = 100.*correct/total
        print('[*] Test acc: {:5f}'.format(acc))
        writer.add_scalar('Test acc', acc, step)

def main(args):
    ### Device check 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    num_gpus = torch.cuda.device_count()
    print('[*] GPU {} is detected.'.format(args.gpu_idx))

    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
        

    ### Path setting
    today = str(datetime.today()).split(' ')[0] + '_' + str(time.strftime('%H%M%S'))
    folder_name = '{}_{}'.format(today, args.log)

    ### Make directory
    log_dir = os.path.join(args.log_dir, folder_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, folder_name)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print('[*] log directory : {}'.format(log_dir))
    print('[*] checkpoint directory : {}'.format(checkpoint_dir))

    ### Logfile
    f = open(os.path.join(log_dir, 'args.txt'), 'w')
    f.write(str(args))
    f.close()

    print('[*] prepare datasets & dataloader...')
    train_transform = T.Compose([
        T.Resize(48),
        T.CenterCrop(32),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    test_transform = T.Compose([
        T.Resize(32),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    train_datasets = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_datasets = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)

    print('[*] build network...')
    net = resnet50(num_classes=100)
    if num_gpus > 1 and device == 'cuda':
        net = nn.DataParallel(net)
    net = net.to(device)


    print('[*] start training')
    writer = SummaryWriter(log_dir)
    train(args, train_loader, test_loader, net, device, writer, log_dir, checkpoint_dir)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
