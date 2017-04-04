import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable


# Define a class that specifies the network architecture
class Net(nn.Module):

	# Constructor
	def __init__(self):
		# Call the constructor of the parent class
		super(Net, self).__init__()

		# Define various modules that will be used in specifying the net arch

		# Conv1 (input channels, output channels, filter dims)
		self.conv1 = nn.Conv2d(1, 6, 5)
		# Conv2
		self.conv2 = nn.Conv2d(6, 16, 5)
		# FC1 (input dim, output dim)
		self.fc1 = nn.Linear(16*5*5, 120)
		# FC2
		self.fc2 = nn.Linear(120, 84)
		# FC3
		self.fc3 = nn.Linear(84, 10)

	# Forward pass
	def forward(self, x):

		# Specify the actual network architecture here

		# Max pooling with stride = 2, and a 2 x 2 filter window
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x

	# Helper function
	def num_flat_features(self, x):

		# All dimensions, except the batch dimension
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


# Initialize the network
net = Net()
params = list(net.parameters())

# Create the optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# Input image
ip = Variable(torch.randn(1, 1, 32, 32))
# out = net(ip)

# A dummy target
target = Variable(torch.range(1, 10))

# Loss function
criterion = nn.MSELoss()
# loss = criterion(out, target)
# print 'loss:', loss

# # Zero the gradient buffers of all parameters
# net.zero_grad()
# # Backprop (with random gradients)
# out.backward(torch.randn(1, 10))

# In the training loop
optimizer.zero_grad()
out = net(ip)
loss = criterion(out, target)
loss.backward()
optimizer.step()
