from collections import defaultdict
import operator
import math
import time
import numpy as np
from copy import deepcopy

class UCT():
	def __init__(self, MDP, heuristic):
		self.MDP = MDP
		self.heuristic = self.random_heuristic
		self.T = set()
		self.N = defaultdict(dict)
		self.Q = defaultdict(dict)
		self.c = 0.1
		self.time_limit = 0.1
		self.curr_player = 0

	def random_heuristic(self, s):
		actions = self.MDP.actions(s)
		return np.random.choice(actions)

	def select_action(self, s, d, play_random=False):
		self.curr_player = s.curr_player
		if play_random:
			return self.random_heuristic(s)
		if not s.is_attack():
			return self.heuristic(s)
		# print "selecting action"
		self.T = set()
		self.N = defaultdict(dict)
		self.Q = defaultdict(dict)
		start = time.time()
		iters = 0 
		while time.time() - start < self.time_limit:
			iters += 1
			self.simulate(s,d)
		# print "select action completed {} simulations".format(iters)
		return max(self.Q[s].iteritems(), key=operator.itemgetter(1))[0]

	def simulate(self, s, d):
		if d == 0:
			return 0 # Can change to heuristic evaluation function
		if s.is_end():
			return 0

		if s not in self.T:
			# print s.to_string()
			for a in self.MDP.actions(s):
				# print a.to_string()
				self.N[s][a] = 1.0
				self.Q[s][a] = 0.0
			self.T.add(s)
			return self.rollout(s,d)
		
		valaction = []
		
		for a, q_sa in self.Q[s].iteritems():
			n_sa = self.N[s][a]
			n_s = sum(self.N[s][ap] for ap in self.N[s])
			explore_term = math.sqrt(math.log(n_s) / n_sa)
			valaction.append((q_sa + self.c * explore_term, a))
		if self.curr_player == s.curr_player:
			a = max(valaction)[1]
		else:
			a = min(valaction)[1]
		sp, r = self.get_sp_r(s, a)
		q = r + self.MDP.discount()*self.simulate(sp,d-1)
		self.N[s][a] += 1
		self.Q[s][a] += (q - self.Q[s][a])/self.N[s][a]
		return q

	def get_sp_r(self, s, a):
		sp, r_list = self.MDP.simulate_action(s,a)
		r = r_list[self.curr_player]
		return sp, r

	def rollout(self, s, d):
		# print "rollout"
		# print d
		# print s
		
		if d == 0 or s.is_end():
			return 0 # Can change to heuristic evaluation function
		a = self.heuristic(s)
		sp, r = self.get_sp_r(s, a)
		return r + self.MDP.discount() * self.rollout(sp, d-1)

