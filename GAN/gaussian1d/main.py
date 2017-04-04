"""
A very simple GAN example - 1 D Gaussian
"""

# General modules
import os

# NumPy
import numpy as np

# Torch-specific modules
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable


"""
True data distribution
"""
class RealDataDistribution:

	# Constructor (initialize the parameters of the Gaussian)
	def __init__(self, mu = 0.0, sigma = 1.0):
		self.mu = mu
		self.sigma = sigma

	# Draw 'n' samples from the distribution
	def sample(self, n = 1):
		return torch.Tensor(np.random.normal(loc = self.mu, scale = self.sigma, size = (n,1)))


"""
Latent variable distribution (Generator noise distribution)
Referred to as z in GAN papers
"""
class GeneratorNoiseDistribution:

	# Constructor (take in the parameters of a uniform distribution)
	def __init__(self, low = 0, high = 1):
		self.low = low
		self.high = high

	# Draw 'n' samples from the distribution
	def sample(self, n = 1):
		return torch.Tensor(np.random.uniform(low = self.low, high = self.high, size = (n,1)))


"""
Generator network
"""
class GeneratorNetwork(nn.Module):

	# Constructor
	def __init__(self, inputSize = 1, hiddenSize = 50, outputSize = 1):
		# Call the constructor of the parent class
		super(GeneratorNetwork, self).__init__()
		# Define various layers that will be used to construct the network
		self.map1 = nn.Linear(inputSize, hiddenSize)
		self.map2 = nn.Linear(hiddenSize, hiddenSize)
		self.map3 = nn.Linear(hiddenSize, outputSize)

	# Specify the forward pass
	def forward(self, input):
		input = func.elu(self.map1(input))
		input = func.sigmoid(self.map2(input))
		return self.map3(input)


"""
Discriminator network
"""
class DiscriminatorNetwork(nn.Module):

	# Constructor
	def __init__(self, inputSize = 1, hiddenSize = 50, outputSize = 1):
		# Call the constructor of the parent class
		super(DiscriminatorNetwork, self).__init__()
		# Define various layers that will be used to construct the network
		self.map1 = nn.Linear(inputSize, hiddenSize)
		self.map2 = nn.Linear(hiddenSize, hiddenSize)
		self.map3 = nn.Linear(hiddenSize, outputSize)

	# Specify the forward pass
	def forward(self, input):
		input = func.elu(self.map1(input))
		input = func.elu(self.map2(input))
		return func.sigmoid(self.map3(input))


"""
Main module
"""
if __name__ == '__main__':

	"""
	Various parameters used in training
	"""
	
	# Batch size for the discriminator network
	batchSize = 100
	# Number of epochs for which the GAN is to be trained
	numEpochs = 25000
	# Number of steps (within each epoch) for which the discriminator is to be finetuned
	numDiscriminatorSteps = 1
	# Learning rate (for the optimizer) (a default choice is 0.002)
	learningRate = 0.002
	# Beta1 (for the Adam optimizer) (a default choice is 0.5)
	beta1 = 0.5


	"""
	Initialization
	"""

	# Initialize the true distribution
	realDist = RealDataDistribution()

	# Initialize the generator noise distribution
	generatorNoiseDist = GeneratorNoiseDistribution()

	# Initialize the generator network
	generator = GeneratorNetwork(inputSize = 1, hiddenSize = 25, outputSize = 1)

	# Initialize the discriminator network
	discriminator = DiscriminatorNetwork(inputSize = 1, hiddenSize = 25, outputSize = 1)

	# Specify the criterion used for training
	criterion = nn.BCELoss()

	# Setup optimizers for the generator and the discriminator
	optimizerGenerator = optim.Adam(generator.parameters(), lr = learningRate, betas = (beta1, 0.999))
	optimizerDiscriminator = optim.Adam(discriminator.parameters(), lr = learningRate, betas = (beta1, 0.999))


	"""
	Main training loop
	"""

	# For each epoch
	for i in range(numEpochs):

		# First, train the discriminator to distinguish among real and fake samples
		# This is carried out for a few steps (batches)
		for k in range(numDiscriminatorSteps):

			# Clear the accumulated discriminator gradients
			discriminator.zero_grad()

			# Draw samples from the true distribution
			trueSamples = Variable(realDist.sample(batchSize))
			# All these samples are true (label is 1)
			labels = Variable(torch.ones(batchSize,1))
			# Pass this through the discriminator
			discriminatorTrueOutput = discriminator(trueSamples)
			# Apply the BCE criterion
			discriminatorTrueLoss = criterion(discriminatorTrueOutput, labels)
			# Accumulate gradients (but do NOT update the network parameters right away)
			discriminatorTrueLoss.backward()

			# Draw samples from the latent distribution (for input to the generator)
			generatorNoise = Variable(generatorNoiseDist.sample(batchSize))
			# Pass this through the generator
			generatorOutput = generator(generatorNoise)
			# All these samples are fake (label is 0)
			labels = Variable(torch.zeros(batchSize,1))
			# Pass this through the discriminator ('detach' it so that the generator params
			# do not get updates)
			discriminatorFakeOutput = discriminator(generatorOutput.detach())
			# Apply the BCE criterion
			discriminatorFakeLoss = criterion(discriminatorFakeOutput, labels)
			# Accumulate the gradients
			discriminatorFakeLoss.backward()

			# Update the discriminator network parameters using the accumulated gradients
			discriminatorLoss = discriminatorTrueLoss + discriminatorFakeLoss
			optimizerDiscriminator.step()


		# Now train the generator to fool the discriminator

		# Clear the accumulated generator gradients
		generator.zero_grad()
		# Draw samples from the latent distribution (for input to the generator)
		generatorNoise = Variable(generatorNoiseDist.sample(batchSize))
		# Pass this through the generator
		generatorOutput = generator(generatorNoise)
		# Pass this through the discriminator
		discriminatorOutput = discriminator(generatorOutput)
		# For training, the generator assumes that all these samples have been classified 
		# as 'real' by the discriminator (label is 0)
		labels = Variable(torch.ones(batchSize, 1))
		# Apply the BCE criterion
		generatorLoss = criterion(discriminatorOutput, labels)
		# Accumulate the gradients
		generatorLoss.backward()

		# Update the generator netwrok parameters using the accumulate gradients
		optimizerGenerator.step()

		# Print status to stdout
		if i % 100 == 0:
			# print 'Epoch:', i, 'Generator Error:', generatorLoss.data.storage().tolist()[0], \
			# 'Discriminator True Error:', discriminatorTrueLoss.data.storage().tolist()[0], \
			# 'Discriminator Fake Error:', discriminatorFakeLoss.data.storage().tolist()[0]
			print 'Epoch:', i, 'Mean:', torch.mean(generatorOutput.data), \
			'Std:', torch.std(generatorOutput.data)
