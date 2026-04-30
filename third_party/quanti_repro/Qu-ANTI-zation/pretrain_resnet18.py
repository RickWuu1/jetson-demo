"""
Pre-train ResNet-18 on CIFAR-10 (clean model).
This produces the base model required by backdoor_w_lossfn.py.

Usage:
    python pretrain_resnet18.py --epochs 100 --lr 0.0001

Output:
    models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth
"""
import os
import json
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from utils.datasets import load_dataset
from utils.networks import load_network
from utils.optimizers import load_lossfn
from utils.learner import train, valid


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_loader, valid_loader = load_dataset('cifar10', args.batch_size, normalize=True, kwargs=kwargs)

    net = load_network('cifar10', 'ResNet18', 10)
    if use_cuda:
        net.cuda()
    print(f' : ResNet18 on {"cuda" if use_cuda else "cpu"}')

    criterion = load_lossfn('cross-entropy')
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 80], gamma=0.1)

    best_acc = 0.0
    save_dir = 'models/cifar10/train'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'ResNet18_norm_128_200_Adam-Multi.pth')

    for epoch in range(1, args.epochs + 1):
        train(epoch, net, train_loader, criterion, scheduler, optimizer, use_cuda=use_cuda)
        acc, loss = valid(epoch, net, valid_loader, criterion, use_cuda=use_cuda)

        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), save_path)
            print(f'  -> Best acc {acc:.2f}%, saved to {save_path}')

    print(f'\nDone. Best clean acc: {best_acc:.2f}%')
    print(f'Model saved: {save_path}')

    # Save summary
    with open('models/cifar10/train/pretrain_summary.json', 'w') as f:
        json.dump({'best_acc': best_acc, 'epochs': args.epochs, 'lr': args.lr}, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    run(args)
