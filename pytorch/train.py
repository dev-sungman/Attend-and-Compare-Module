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
from models.resnet import resnet50, ACMBlock


def train(args, train_loader, valid_loader, net, device, writer, log_dir, checkpoint_dir):
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
            loss = criterion(logits, labels)
            _, preds = logits.max(1)

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            acc = 100.*correct/total

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.print_freq == 0 and step != 0:
                print('Epoch: {:2d}, Step: {:5d}, Loss: {:5f}, Acc: {:5f}'.format(epoch, step, loss.item, acc))
                writer.add_scalar('Train loss', loss, step)
                writer.add_scalar('Train acc', acc, step)
            step += 1

        valid(args, valid_loader, net, writer, log_dir, checkpoint_dir, step)

        if epoch %2 == 0 and epoch != 0:
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, str(step)) + '.pth')

def valid(args, data_loader, net, device, writer, step):
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
            loss = criterion(logits, labels)
            _, preds = logits.max(1)

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        acc = 100.*correct/total
        print('[*] Validation acc: {:5f}'.format(acc))
        writer.add_scalar('Validation acc', acc, step)

def test(args, data_loader, net, device, writer, step):
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
            loss = criterion(logits, labels)
            _, preds = logits.max(1)

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        acc = 100.*correct/total
        print('[*] Test acc: {:5f}'.format(acc))
        writer.add_scalar('Test acc', acc, step)

def main(args):

