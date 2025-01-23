import os
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# # Check if CUDA is available
# print(torch.cuda.is_available())


n_epochs = 7
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.9
log_interval = 10
method="Adam"

model_name = "model_checkpoint.pth"

# Fix randomness
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Data Loaders
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

# Define Network
class Net(nn.Module):
    def __init__(self):
        #V1 CODE
        # super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

        #V2 CODE
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        #V1
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)

        #V2
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

network = Net()

# Optimization Setup
#V3 Learning rate reduced from 0.01 to 0.001
def get_optimizer(network, method, learning_rate, momentum):
    if method == "Adam":
        return optim.Adam(network.parameters(), lr=learning_rate)
    elif method == "SGD":
        return optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    elif method == "RMSprop":
        return optim.RMSprop(network.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer method: {method}")

optimizer = get_optimizer(network, method, learning_rate, momentum)

# Model Saving/Loading
save_path = './results'
os.makedirs(save_path, exist_ok=True)

def save_model(network, optimizer, path):
    torch.save({'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, path)

def load_model(network, optimizer, path):
    checkpoint = torch.load(path, weights_only=True)
    network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model loaded successfully from {path}")

# Training and Testing Functions
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                  f" ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            train_losses.append(loss.item())
            train_counter.append((batch_idx * batch_size_train) + ((epoch - 1) * len(train_loader.dataset)))

            # Save checkpoint with default or custom name
            save_model(network, optimizer, os.path.join(save_path, model_name))

def test(num_runs=1):
    network.eval()
    total_loss = 0
    total_correct = 0
    for _ in range(num_runs):
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()
        total_loss += test_loss / len(test_loader.dataset)
        total_correct += correct / len(test_loader.dataset)
    avg_loss = total_loss / num_runs
    avg_accuracy = total_correct / num_runs
    print(f"\nTest set (averaged over {num_runs} runs): Avg. loss: {avg_loss:.4f}, "
          f"Accuracy: {avg_accuracy * 100:.2f}%\n")
    return avg_loss, avg_accuracy

# Choose Mode
mode = input("Enter 'train' to train a new model or 'test' to evaluate a pre-trained model: ").strip().lower()

if mode == "train":
    custom_name = input("Enter a name for saving the model (or press Enter for default): ").strip()
    model_name = custom_name if custom_name else 'model_checkpoint.pth'
    model_path = os.path.join(save_path, model_name)

    t_out = (0,0)

    test()  # Evaluate before training
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        t_out = test()

    f = open("results/"+model_name+".txt", "w")
    f.write(model_name+"\n")
    f.write("avg_loss, avg_accuracy " + "\n")
    f.write(' '.join(str(val) for val in t_out)+"\n\n")
    f.write("number of epchos "+str(+n_epochs)+"\n")
    f.write("batch_size_train " +str( batch_size_train) + "\n")
    f.write("batch_size_test " + str(batch_size_test) + "\n")
    f.write("learning_rate " +str( learning_rate) + "\n")
    f.write("momentum " +str( momentum )+ "\n")
    f.write("log_interval " + str(log_interval )+ "\n")
    f.write("log_interval " + method + "\n\n")


    for name, module in network.named_modules():
        f.write(str(name)+" "+str(module) +"\n")


    f.close()

    print(f"Model saved as {model_path}")

elif mode == "test":
    custom_name = input("Enter the name of the model to load (or press Enter for default): ").strip()
    model_name = custom_name if custom_name else 'model_checkpoint.pth'
    model_path = os.path.join(save_path, model_name)

    if os.path.exists(model_path):
        load_model(network, optimizer, model_path)
        test(num_runs=10)  # Run test 10 times and average results
    else:
        print(f"No model found at {model_path}. Please train a model first.")
else:
    print("Invalid input. Please enter 'train' or 'test'.")