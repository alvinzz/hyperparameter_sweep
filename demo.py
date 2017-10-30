import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from sweep import Sweeper

class mnistModel(object):
    def __init__(self, params):
        # parse args
        self.batch_size = params['batch_size']
        self.dropout = params['dropout']
        self.num_hidden = params['num_hidden']
        self.activation = params['activation']
        self.learning_rate = params['learning_rate']
        if params['optimizer'] == 'sgd':
            self.optimizer_fn = optim.SGD
        elif params['optimizer'] == 'adam'
            self.optimizer_fn = optim.Adam
        elif params['optimizer'] == 'rms':
            self.optimizer_fn = optim.RMSprop

        # load in data
        cuda_args = {'num_workers': 1, 'pin_memory': True}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.batch_size, shuffle=True, **cuda_args)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.batch_size, shuffle=True, **cuda_args)

        # create NN according to params
        class Net(nn.Module):
            def __init__(self, dropout, num_hidden, activation):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
                self.conv3_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)
                self.dropout = dropout
                self.num_hidden = num_hidden
                self.activation = activation

            def forward(self, x):
                x = self.activation(F.max_pool2d(self.conv1(x), 2))
                if self.dropout:
                    x = self.activation(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
                else:
                    x = self.activation(F.max_pool2d(self.conv2(x), 2))
                if num_hidden == 3:
                    if self.dropout:
                        x = self.activation(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
                    else:
                        x = self.activation(F.max_pool2d(self.conv3(x), 2))
                x = x.view(-1, 320)
                x = self.activation(self.fc1(x))
                if self.dropout:
                    x = F.dropout(x, training=self.training)
                x = self.fc2(x)
                return F.log_softmax(x)
        self.model = Net().cuda()
        self.optimizer = self.optimizer_fn(model.parameters(), lr=self.learning_rate)

    def train(self):
        # 10 epochs
        for _ in range(10):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

    def eval(self):
        model.eval()
        test_loss = 0
        for data, target in self.test_loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        test_loss /= len(test_loader.dataset)
        return test_loss

if __name__ == '__main__':
    param_dict = {
                'batch_size': {'type': 'discrete', 'range': [16, 32, 64, 128]},
                'dropout': {'type': 'discrete', 'range': [True, False]},
                'num_hidden': {'type': 'discrete', 'range': [2, 3]},
                'optimizer': {'type': 'discrete', 'range': ['rms', 'sgd', 'adam']},
                'activation': {'type': 'discrete', 'range': [F.relu, F.tanh, F.sigmoid]},
                'learning_rate': {'type': 'continuous', 'range': [1e-5, 1e-1], 'sweep_type': 'exp', 'sweep_num': 5},
            }
    s = Sweeper(param_dict, mnistModel)
    s.random_sweep(num_iters=2)
