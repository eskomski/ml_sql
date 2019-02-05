import sqlite3

from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_tensor

data = MNIST('data', train=True, transform=to_tensor, download=True)
train_size = int(len(data)*0.7)
split = random_split(data, [train_size, len(data) - train_size])

train = DataLoader(split[0], batch_size=4096)
dev = DataLoader(split[1], batch_size=4096)

class DNN(nn.Module):
    def __init__(self, d, c, layers=[128, 64]):
        super(DNN, self).__init__()
        self.layers = nn.ModuleList()

        prev = d
        for l in layers:
            self.layers += [nn.Linear(prev, l)]
            prev = l
        self.layers += [nn.Linear(prev, c)]

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x).clamp(min=0)
        return self.layers[-1](x)

def run_exp(layers, lr, conn=None, exp_id=None):
    if conn and exp_id is None:
        print('error: you forgot the experiment id. incrementing bozo counter...')
        return

    model = DNN(28*28, 10, layers)

    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    if conn: c = conn.cursor()

    for e in range(5):
        model.train()
        train_loss = 0.0; train_acc = 0

        for batch in train:
            y_pred = model(batch[0].view(-1, 28*28))

            loss = criterion(y_pred, batch[1])
            train_loss += loss.item()
            train_acc += (y_pred.argmax(1) == batch[1]).sum().item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        train_loss /= len(train)
        train_acc /= train_size

        model.eval()
        dev_loss = 0.0; dev_acc = 0

        for batch in dev:
            y_pred = model(batch[0].view(-1, 28*28))
            dev_loss += criterion(y_pred, batch[1]).item()
            dev_acc += (y_pred.argmax(1) == batch[1]).sum().item()

        dev_loss /= len(dev)
        dev_acc /= (len(data) - train_size)

        print('epoch {}: train loss={:.4f} acc={:.4f}, dev loss={:.4f} acc={:.4f}'
                .format(e, train_loss, train_acc, dev_loss, dev_acc))

        if conn:
            c.execute('insert into results values (?, ?, ?, ?)', (exp_id, e, dev_loss, dev_acc))
            conn.commit()

