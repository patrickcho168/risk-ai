from collections import defaultdict
import operator
import math
import time
import numpy as np

class UCT():
	def __init__(self, MDP, heuristic):
		self.MDP = MDP
		self.heuristic = self.random_heuristic
		self.T = set()
		self.N = defaultdict(dict)
		self.Q = defaultdict(dict)
		self.c = 0.1
		self.time_limit = 1

	def random_heuristic(self, s):
		actions = self.MDP.actions(s)
		return np.random.choice(actions)

	def select_action(self, s, d):
		if not s.is_attack():
			return self.heuristic(s)
		self.T = set()
		self.N = defaultdict(dict)
		self.Q = defaultdict(dict)
		start = time.time()
		iters = 0 
		while time.time() - start < self.time_limit:
			iters += 1
			self.simulate(s,d)
		print "select action completed {} simulations".format(iters)
		return max(self.Q[s].iteritems(), key=operator.itemgetter(1))[0]

	def simulate(self, s, d):
		if d == 0:
			return 0 # Can change to heuristic evaluation function
		if s not in self.T:
			for a in self.MDP.actions(s):
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
		assert len(valaction) > 0
		a = max(valaction)[1]
		sp, r = self.get_sp_r(s, a)

		q = r + self.MDP.discount()*self.simulate(sp,d-1)
		cached_qs = self.Q[s]
		cached_ns = self.N[s]
		try: 
			self.N[s][a] += 1
		except:
			print "Key error"
			print valaction
			print self.N[s]
			print self.Q[s]
			print cached_qs
			print cached_ns
			print s.to_string()
			print a.to_string()
			raise ValueError("empty")
		self.Q[s][a] += (q - self.Q[s][a])/self.N[s][a]
		return q

	def get_sp_r(self, s, a):
		# print s.to_string()
		# print a.to_string()
		sp, r_list = self.MDP.simulate_action(s,a)
		r = r_list[s.curr_player]
		return sp, r

	def rollout(self, s, d):
		if d == 0 or s.is_end():
			return 0 # Can change to heuristic evaluation function
		a = self.heuristic(s)
		sp, r = self.get_sp_r(s, a)
		return r + self.MDP.discount() * self.rollout(sp, d-1)

