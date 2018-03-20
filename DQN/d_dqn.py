import tensorflow as tf
import numpy as np

# np.random.seed(1)
# tf.set_random_seed(1)

""" 
This code is a modified version of code at https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
It has been modified to suit our needs, incorporate some new algorithms and tune parameters easily
"""

sDQN = "DQN"
sDOUBLE_DQN = "Double_DQN"
sPRIORITIZED_DQN = "PrioritizedReplayDQN"
sDOUBLE_PRIORITIZED_DQN = "DoublePrioritizedReplayDQN"

class DQN:
	def __init__(self, statedim, actiondim, gamma = 0.99, epsmin = 0.5, epsmax = 0.95, epsstep = 0.0001, 
	tupleMemoryCount = 50000, tupleBatchSize= 32, replacetargetineps = 4000, optimizer = tf.train.AdamOptimizer, lr = 0.0001,
	hiddenlayers = 2, hiddenunits = 100, dqnType = sDQN):

		self.statedim = statedim #dimension of the input state space
		self.actiondim = actiondim #dimension of the action space
		self.gamma = gamma #discount factor in reward calculation
		self.eps = epsmin #init epsilon with min allowed value of epsilon
		self.epsmax = epsmax #maximum exploitation allowed
		self.epsstep = epsstep #the increase in epsilon per step
		self.tupleMemoryCount = tupleMemoryCount #the total allowed tuples in memory for experience replay
		self.replacetargetineps = replacetargetineps #the number of steps after which target params are updated with DQN network params
		self.optimizer = optimizer #the optimizer to be used to calculate gradient
		self.lr = lr #learning rate in backpropagation
		self.hiddenlayers = hiddenlayers #the number of hidden layers in the network
		self.hiddenunits = hiddenunits #the number of hidden units per layer
		self.dqnType = dqnType #the dqn type to be used
		self.tupleBatchSize = tupleBatchSize #the number of tuples in each batch in a backprop
		self.timesteps = 0 #keeps track of the time steps to be used
		self.currMemCount = 0 #keeps track of the number of tuples seen so far

		self.prioritized = False #false by default
		self.doubleDqn = False #false by default
		
		if(dqnType==sPRIORITIZED_DQN):
			self.prioritized = True
			self.tupleMemory = Memory(self.tupleMemoryCount)
		elif(dqnType == sDOUBLE_DQN):
			self.doubleDqn = True
			self.tupleMemory = np.zeros((self.tupleMemoryCount, self.statedim*2 + 1 + 1)) # <s, a, r, s'>
		elif(dqnType == sDOUBLE_PRIORITIZED_DQN):
			self.prioritized = True
			self.doubleDqn = True
			self.tupleMemory = Memory(self.tupleMemoryCount)
		else:
			self.tupleMemory = np.zeros((self.tupleMemoryCount, self.statedim*2 + 1 + 1)) # <s, a, r, s'>

		self.updateParams()
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())


	def initLayers(self, inp, context, weightsinit, biasinit):
		for i in range(self.hiddenlayers):
			if(i==0):
				with tf.variable_scope('l1'):
					w1 = tf.get_variable('w1', [self.statedim, self.hiddenunits], initializer=weightsinit, collections=context)
					b1 = tf.get_variable('b1', [1, self.hiddenunits], initializer=biasinit, collections=context)
					out = tf.nn.relu(tf.matmul(inp, w1) + b1)
					inp = out
			else:
				currl = str(i+1)
				with tf.variable_scope('l'+currl):
					# print('l'+currl)
					w = tf.get_variable('w'+currl, [self.hiddenunits, self.hiddenunits], initializer=weightsinit, collections=context)
					b = tf.get_variable('b'+currl, [1, self.hiddenunits], initializer=biasinit, collections=context)
					out = tf.nn.relu(tf.matmul(inp, w) + b)
					inp = out
		currl = str(self.hiddenlayers+1)
		# print(currl)
		with tf.variable_scope('l'+currl):
			w = tf.get_variable('w'+str(currl), [self.hiddenunits, self.actiondim], initializer=weightsinit, collections=context)
			b = tf.get_variable('b'+str(currl), [1, self.actiondim], initializer=biasinit, collections=context)
			out = tf.matmul(inp, w) + b

		return out


	def updateParams(self):
		self.s = tf.placeholder(tf.float32, [None, self.statedim], name='s')
		self.sprime = tf.placeholder(tf.float32, [None, self.statedim], name='sprime')
		self.target = tf.placeholder(tf.float32, [None, self.actiondim], name='target')
		if(self.prioritized):
			self.lossWeight = tf.placeholder(tf.float32, [None, 1], name='lossWeight')

		with tf.variable_scope('dqnnet'):
			inp, context, weightsinit, biasinit = self.s, ["dqnnetparams", tf.GraphKeys.GLOBAL_VARIABLES], \
			tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)
			self.qval = self.initLayers(inp, context, weightsinit, biasinit)
		with tf.variable_scope('lossandoptim'):
			if(self.prioritized):
				self.loss = tf.reduce_mean(self.lossWeight*tf.squared_difference(self.qval, self.target))
				self.errorForSumTree = tf.reduce_sum(tf.abs(self.qval - self.target), axis=1) #sum across all actions
			else:
				self.loss = tf.reduce_mean(tf.squared_difference(self.qval, self.target))
			self.runoptim = self.optimizer().minimize(self.loss)
		with tf.variable_scope('targetnet'):
			inp, context, weightsinit, biasinit = self.sprime, ["targetnetparams", tf.GraphKeys.GLOBAL_VARIABLES], \
			tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)
			self.targetnetqval = self.initLayers(inp, context, weightsinit, biasinit)

	def selectAction(self, currState):
		currState = currState[np.newaxis, :]
		currActionVal = self.sess.run(self.qval, feed_dict={self.s : currState})
		maxAction = np.argmax(currActionVal)
		maxActionVal = np.max(currActionVal)

		if np.random.uniform() > self.eps:
			maxAction = np.random.randint(0, self.actiondim)
			maxActionVal = currActionVal[0, maxAction]

		return maxAction, maxActionVal

	def addTupleToMemory(self, s, a, r, sprime):
		currTuple = np.hstack((s, a, r, sprime))
		if(self.prioritized):
			self.tupleMemory.addTuple(currTuple)
		else:
			memAccIndex = self.currMemCount % self.tupleMemoryCount
			self.tupleMemory[memAccIndex,:] = currTuple
			self.currMemCount += 1

	def backprop(self):
		#Replace target vars every replacetargetineps timesteps
		if(self.timesteps%self.replacetargetineps == 0):
			dqnparams = tf.get_collection("dqnnetparams")
			targparams = tf.get_collection("targetnetparams")
			for dqnparam, tparam in zip(dqnparams, targparams):
				tf.assign(tparam, dqnparam)
		self.timesteps += 1

		if(self.prioritized):
			treePosns, tupleBatch, lossWeight = self.tupleMemory.getBatchSample(self.tupleBatchSize)
		else:
			if(self.currMemCount > self.tupleMemoryCount):
				randIndex = np.random.choice(self.tupleMemoryCount, self.tupleBatchSize)
			else:
				randIndex = np.random.choice(self.currMemCount, self.tupleBatchSize)
			# print("Random indexes: " ,randIndex.shape)
			tupleBatch = self.tupleMemory[randIndex, :]
		# print("Chosen tuples: ", tupleBatch.shape)

		currValS, targetValSPrime = self.sess.run([self.qval, self.targetnetqval],
			feed_dict={self.s : tupleBatch[:, :self.statedim],
						self.sprime : tupleBatch[:, -self.statedim:]})

		currTarget = currValS.copy()
		tupleBatchIndexList = np.arange(self.tupleBatchSize, dtype=np.int32)
		tupleBatchActionIndex = tupleBatch[:, self.statedim].astype(int)
		tupleBatchRewards = tupleBatch[:, self.statedim+1]
		
		if(self.doubleDqn):
			maxActions = np.argmax(currValS, axis=1)
			targetExpectedReward = targetValSPrime[tupleBatchIndexList, maxActions]
		else:
			targetExpectedReward = np.max(targetValSPrime, axis=1)
		currTarget[tupleBatchIndexList, tupleBatchActionIndex] = tupleBatchRewards + self.gamma*targetExpectedReward

		if(self.prioritized):
			runningoptim, currCost, errs = self.sess.run([self.runoptim, self.loss, 
				self.errorForSumTree], feed_dict={self.s : tupleBatch[:, :self.statedim],
													self.target : currTarget,
													self.lossWeight : lossWeight})
			self.tupleMemory.updateBatchPriority(treePosns, errs)
		else:
		#backprop the error:
			runningoptim, currCost = self.sess.run([self.runoptim, self.loss],
				feed_dict={self.s : tupleBatch[:, :self.statedim],
							self.target : currTarget})

		if(self.eps+self.epsstep<=self.epsmax):
			self.eps += self.epsstep
		else:
			self.eps = self.epsmax


class SumTree(object):
	def __init__(self, treeCapacity):
		self.dataPosn = 0
		self.treeCapacity = treeCapacity
		self.treeValues = np.zeros(2*treeCapacity - 1) #A complete binary tree of n leaves has 2n-1 nodes in total
		self.actualData = np.zeros(treeCapacity, dtype=object) #To store the actual tuple data

	def add(self, data, priority):
		treePosn = self.dataPosn + self.treeCapacity - 1
		self.actualData[self.dataPosn] = data #update the data inn actual data array
		self.update(treePosn, priority) #update the priority in the sumtree to accomodate to new data
		self.dataPosn += 1
		if self.dataPosn >= self.treeCapacity: #reset data position to fill new data in round robin fashion
			self.dataPosn = 0

	def update(self, treePosn, priority):
		changeInPriority = priority - self.treeValues[treePosn]
		self.treeValues[treePosn] = priority
		while treePosn!=0: #Propagate the change until the root
			treePosn = (treePosn-1) // 2
			self.treeValues[treePosn] += changeInPriority

	def getLeafTuple(self, val):
		parentPosn = 0
		while True:
			leftChildPosn = 2*parentPosn + 1
			rightChildPosn = leftChildPosn + 1
			if leftChildPosn >= len(self.treeValues):
				leafTuplePosn = parentPosn
				break
			else:
				if val <= self.treeValues[leftChildPosn]:
					parentPosn = leftChildPosn
				else:
					val -= self.treeValues[leftChildPosn]
					parentPosn = rightChildPosn
		return leafTuplePosn, self.treeValues[leafTuplePosn], self.actualData[leafTuplePosn-self.treeCapacity+1]

	def totalPriority(self):
		return self.treeValues[0]
		
class Memory():
	def  __init__(self, tupleMemoryCount):
		self.minErr = 0.01 #Avoids zero probability for zero error tuples
		self.alpha = 0.6 # Used to convert error to priority
		self.beta = 0.4 # Used to weigh the loss according to priority
		self.betaStep = 0.001
		self.errMax = 1.0
		self.sumTree = SumTree(tupleMemoryCount)

	def addTuple(self, tupleData):
		defaultPriority = np.max(self.sumTree.treeValues[-self.sumTree.treeCapacity:]) #Assigning max priority of existing leaf nodes as the default priority
		if defaultPriority==0:
			defaultPriority = self.errMax
		self.sumTree.add(tupleData, defaultPriority)

	def updateBatchPriority(self, treePosn, updatedErrors):
		updatedErrors += self.minErr # To avoid 0 errors
		clippedErrors = np.minimum(updatedErrors, self.errMax)
		newPriority = np.power(clippedErrors, self.alpha)
		for t, p in zip(treePosn, newPriority):
			self.sumTree.update(t, p)

	def getBatchSample(self, n):
		batchPosn = np.empty((n,), dtype=np.int32)
		batchTuples = np.empty((n, len(self.sumTree.actualData[0])))
		lossWeight = np.empty((n,1))

		segmentPriority = self.sumTree.totalPriority()/n
		self.beta = np.min([1.0, self.beta + self.betaStep])
		minPriority = np.min(self.sumTree.treeValues[-self.sumTree.treeCapacity:])

		for i in range(n):
			l, r = segmentPriority*i, segmentPriority*(i+1)
			val = np.random.uniform(l, r)
			leafPosn, priority, tupleData = self.sumTree.getLeafTuple(val)
			lossWeight[i,0] = np.power(priority/minPriority, -self.beta)
			batchPosn[i], batchTuples[i,:] = leafPosn, tupleData

		return batchPosn, batchTuples, lossWeight