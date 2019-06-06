# adapted from: https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from utils import get_data_loaders


class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = nnf.relu(self.conv1(x))
        x = nnf.max_pool2d(x, 2, 2)
        x = nnf.relu(self.conv2(x))
        x = nnf.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = nnf.relu(self.fc1(x))
        x = self.fc2(x)
        return nnf.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nnf.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with pt.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nnf.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Pypt MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=60)
    parser.add_argument('--fashion', action='store_true', default=False)
    parser.add_argument('--save-dir', type=str, default=None)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and pt.cuda.is_available()

    pt.manual_seed(args.seed)

    device = pt.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = get_data_loaders(use_cuda, args, fashion=args.fashion)

    model = MnistCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if args.save_dir is not None:
        pt.save(model.state_dict(), "softmax_mnist_cnn{}.pt".format('_fashion' if args.fashion else ''))


if __name__ == '__main__':
    main()
