# import basic library
import random
import argparse 
import numpy as np 
import robotWorld as World

# import torch library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# create parser here
parser = argparse.ArgumentParser(description="Reinforcement learning Abilene dataset")
parser.add_argument("--memory", type = int, default = 2000, help = "memory size")
parser.add_argument("--epochs", type = int, default = 4, help = "number of epochs")
parser.add_argument("--batch_size", type = int, default = 64, help = "batch size")
parser.add_argument("--lr", type = float, default = 1e-6, help = "learning rate")
parser.add_argument("--weight_decay", type = list, default = [0, 1e-4], help = "regularization parameter")
parser.add_argument("--models", type = str, default = 'Mobilenet', help = "Enter Mobilenet, Resnet18, Resnet34, Resnet50")
opt = parser.parse_args()

class Memory(): 

	def __init__(self, size): 

		self.count = 0
		self.buffer = []
		self.size = size

	def add(self, xp): 

		if len(self.buffer) < self.size: 
			self.buffer.append(xp)
		else: 
			self.buffer[self.count] = xp

		self.count = (self.count+1)%self.size

	def clear(self): 

		self.buffer = []

	def randomSample(self, batch_size): 

		taille = batch_size if batch_size < len(self.buffer) else len(self.buffer)
		sample = random.sample(self.buffer, taille)
		return sample, taille

def discountReward(r, gamma = 0.99): 

	discounted = np.zeros_like(r)
	current = 0

	for i in reversed(range(r.shape[0])): 
		current = current*gamma + r[i]
		discounted[i] = current
	return discounted


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#						     Model Creation 
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


hiddenSize = 100
model = nn.Sequential(nn.Linear(observation_space, hiddenSize), nn.ReLU(), nn.Linear(hiddenSize, action_space))

successiveActions = 10 # number of frames before choosing new action given perceptions

load = True
if load: 
	print('Loading previous model....')
	model = torch.load('catcherDQN.robot')
	print('Model loaded !')

adam = optim.Adam(model.parameters(), 1e-3)

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#						    Learning Loop
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

epochs = opt.epochs
batch_size = opt.batch
memory = Memory(opt.memory)


epsi_max = 0.9
epsi_min = 0.1
epsi_decay = 5000

informationFreq = 100
successes = 0
success_history = []

for epoch in range(epochs):

	complete = False
	steps = 0
	reward =  0
	ep_history = []

	s = env.reset()

	while not complete: 

		# Choosing action via epsilon greedy methods 

		if steps % successiveActions == 0:
			epsi = epsi_min + (epsi_max - epsi_min)*np.exp(-epoch*1./epsi_decay)
			exploration = np.random.random()

			if exploration < epsi: 
				action = env.randomAction()
			else: 
				sTensor = Variable(torch.Tensor(s).type(torch.FloatTensor), requires_grad = False)
				result = model(sTensor)
				action = np.argmax((result.data.numpy()).reshape(-1))

		# Getting observation as a result of interaction and saving transition 

		newState, r, complete, success = env.step(action)

		ep_history.append([s, action, r, newState])
		reward += r
		s = newState
		steps += 1

		if complete: 
			successes += success

			ep_history = np.array(ep_history)
			ep_history[:,2] = discountReward(ep_history[:,2])

			# Adding episodes to memory

			for ep in ep_history: 
				memory.add(ep)
			

			# Sampling mini_batch from memory

			mini_batch,trueSize = memory.randomSample(batch_size)

			sH = np.zeros((trueSize, observation_space))
			rH = np.zeros((trueSize))
			aH = np.zeros((trueSize))
			nsH = np.zeros_like(sH)

			for ex in range(trueSize): 
				sH[ex,:] = mini_batch[ex][0]
				aH[ex] = mini_batch[ex][1]
				rH[ex] = mini_batch[ex][2]
				nsH[ex,:] = mini_batch[ex][3]


			indexes = np.arange(len(mini_batch))*action_space
			for it, ind in enumerate(indexes): 
				indexes[it] += aH.reshape(-1)[it]

			# Value iteration 

			shTensor = Variable(torch.from_numpy(sH).type(torch.FloatTensor))
			qValues = model.forward(shTensor)

			selectedActions = Variable(torch.from_numpy(indexes).type(torch.LongTensor))
			qValues = qValues.view(-1).index_select(dim = 0, index = selectedActions)

			nextStateTensor = Variable(torch.from_numpy(nsH).type(torch.FloatTensor), volatile = True)
			nextQValues = model.forward(nextStateTensor).max(1)[0]
			nextQValues.volatile = False

			rewardTensor = Variable(torch.from_numpy(rH).type(torch.FloatTensor))
	
			expected = rewardTensor + 0.9*nextQValues

			loss_fn = nn.MSELoss()
			loss = loss_fn(qValues, expected)
			adam.zero_grad()
			loss.backward()
			adam.step()


			# Book keeping

			if epoch%informationFreq == 0: 

				text = 'It: {} / {} | Success {} / {} | Explo: {} '.format(epoch,epochs,successes, informationFreq, epsi)
				print(text)
				success_history.append(successes)
				successes = 0
				torch.save(model, 'catcherDQN.robot')
