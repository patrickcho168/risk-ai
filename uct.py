from collections import defaultdict
import operator
import math
import time
import numpy as np
import random
from copy import deepcopy

class UCT():
	def __init__(self, MDP, heuristic_player, c):
		self.MDP = MDP
		self.heuristic_player = heuristic_player
		self.T = set()
		self.N = defaultdict(dict)
		self.Q = defaultdict(dict)
		self.c = c
		self.time_limit = 0.5
		self.curr_player = 0

	def random_heuristic(self, s):
		actions = self.MDP.actions(s)
		return np.random.choice(actions)

	def flush_data(self):
		self.T = set()
		self.N = defaultdict(dict)
		self.Q = defaultdict(dict)

	def select_action(self, s, d):
		self.curr_player = s.curr_player
		if not s.is_attack():
			return self.heuristic_player.getAction(s)
		start = time.time()
		iters = 0
		phi_s = self.MDP.state_to_feature_vect(s, self.curr_player)
		while time.time() - start < self.time_limit:
			iters += 1
			self.simulate(s, phi_s, d)
		# print "select action completed {} simulations".format(iters)
		return max(self.Q[phi_s].iteritems(), key=operator.itemgetter(1))[0]

	def simulate(self, s, phi_s, d):
		if d == 0:
			return self.MDP.evaluate(s, self.curr_player)
		if s.is_end():
			return 0

		if phi_s not in self.T:
			for a in self.MDP.actions(s):
				self.N[phi_s][a] = 1.0
				self.Q[phi_s][a] = 0.0
			self.T.add(phi_s)
			return self.rollout(s,d)
		
		valaction = []
		
		for a, q_sa in self.Q[phi_s].iteritems():
			n_sa = self.N[phi_s][a]
			n_s = sum(self.N[phi_s][ap] for ap in self.N[phi_s])
			explore_term = math.sqrt(math.log(n_s) / n_sa)
			valaction.append((q_sa + self.c * explore_term, a))
		if self.curr_player == s.curr_player:
			a = max(valaction)[1]
		else:
			a = min(valaction)[1]
		sp, r = self.get_sp_r(s, a)
		phi_sp = self.MDP.state_to_feature_vect(sp, self.curr_player)
		q = r + self.MDP.discount()*self.simulate(sp, phi_sp, d-1)
		self.N[phi_s][a] += 1
		self.Q[phi_s][a] += (q - self.Q[phi_s][a])/self.N[phi_s][a]
		return q

	def get_sp_r(self, s, a):
		sp, r_list = self.MDP.simulate_action(s,a)
		r = r_list[self.curr_player]
		return sp, r

	def rollout(self, s, d):
		if d == 0:
			return self.MDP.evaluate(s, self.curr_player)
		if s.is_end():
			return 0
		if random.random() < 0.1: #explore a bit in rollout
			a = random.choice(self.MDP.actions(s))
		else:
			a = self.heuristic_player.getAction(s)
		sp, r = self.get_sp_r(s, a)
		return r + self.MDP.discount() * self.rollout(sp, d-1)

