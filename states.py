state_setup = 'SETUP'
state_place = 'PLACE'
state_attack = 'ATTACK'
state_fortify = 'FORTIFY'
state_end = 'END'

class State():
	def __init__(self, game_phase, country_mapping, curr_player):
		self.game_phase = game_phase
		self.country_mapping = country_mapping
		self.curr_player = curr_player

	def to_string(self):
		to_ret = "\nSTATE:\n"
		to_ret += str(self.game_phase) + '\n'
		to_ret += str(self.country_mapping) + '\n'
		to_ret += "Current player: {}".format(self.curr_player) + '\n'
		return to_ret

	def __hash__(self):
		return hash(self.to_string())

	def is_setup(self):
		return self.game_phase == state_setup

	def is_place(self):
		return self.game_phase == state_place

	def is_attack(self):
		return self.game_phase == state_attack

	def is_fortify(self):
		return self.game_phase == state_fortify

	def is_end(self):
		return self.game_phase == state_end

class Setup_State(State):
	def __init__(self, country_mapping, curr_player, troops_to_place):
		State.__init__(self, state_setup, country_mapping, curr_player)
		self.troops_to_place = troops_to_place

	def to_string(self):
		to_ret = State.to_string(self)
		to_ret += "Troops to place: {}\n".format(self.troops_to_place)
		return to_ret

class Place_State(State):
	def __init__(self, country_mapping, curr_player, troops_to_place):
		State.__init__(self, state_place, country_mapping, curr_player)
		self.troops_to_place = troops_to_place

	def to_string(self):
		to_ret = State.to_string(self)
		to_ret += "Troops to place: {}\n".format(self.troops_to_place)
		return to_ret

class Attack_State(State):
	def __init__(self, country_mapping, curr_player):
		State.__init__(self, state_attack, country_mapping, curr_player)

class Fortify_State(State):
	def __init__(self, country_mapping, curr_player):
		State.__init__(self, state_fortify, country_mapping, curr_player)

class End_State(State):
	def __init__(self, country_mapping, curr_player):
		State.__init__(self, state_end, country_mapping, curr_player)
