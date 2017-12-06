from collections import defaultdict
import operator
import math
import time
import numpy as np

class UCT():
	def __init__(self, MDP, heuristic=UCT.random_heuristic):
		self.MDP = MDP
		self.heuristic = heuristic
		self.T = set()
		self.N = defaultdict(dict)
		self.Q = defaultdict(dict)
		self.c = 0.1
		self.time_limit = 10

	def random_heuristic(self, s):
		actions = self.MDP.getActions(s)
		return np.random.choice(actions)

	def select_action(self, s, d):
		self.T = set()
		self.N = defaultdict(dict)
		self.Q = defaultdict(dict)
		start = time.time()
		while time.time() - start < self.time_limit:
			simulate(s,d)
		return max(self.Q[s].iteritems(), key=operator.itemgetter(1))[0]

	def simulate(self, s, d):
		if d == 0:
			return 0 # Can change to heuristic evaluation function
		if s not in T:
			for a in MDP.actions(s):
				self.N[s][a] = 0.0
				self.Q[s][a] = 0.0
			T.add(s)
			return rollout(s,d)
		valaction = []
		for a, q_sa in self.Q[s].iteritems():
			n_sa = self.N[s][a]
			n_s = sum(self.N[s][ap] for ap in self.N[s]))
			valaction.append((q_sa + self.c * math.sqrt(math.log(n_s) / n_sa), a))
		a = max(valaction)[1]
		(sp, r) = self.MDP.simulate_action(s,a)
		q = r + self.MDP.discount()*simulate(sp,d-1)
		self.N[s][a] += 1
		self.Q[s][a] += (q - self.Q[s][a])/self.N[s][a]
		return q

	def rollout(self, s, d):
		if d == 0:
			return 0 # Can change to heuristic evaluation function
		a = self.heuristic(state)
		(sp, r) = self.MDP.simulate_action(s,a)
		return r + self.MDP.discount() * rollout(sp, d-1)

