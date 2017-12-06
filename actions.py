action_place_troops = 'PLACE_TROOPS'
action_attack_country = 'ATTACK_COUNTRY'
action_move_troops = 'MOVE_TROOPS'
action_end_turn = 'END_TURN'

class Action():
	def __init__(self, action_type):
		self.action_type = action_type

	def to_string(self):
		to_ret = "\nACTION:\n"
		to_ret += str(self.action_type) + '\n'
		return to_ret

	def __hash__(self):
		return hash(self.to_string())

	def is_attack(self):
		return self.action_type == action_attack_country

	def is_place(self):
		return self.action_type == action_place_troops

	def is_fortify(self):
		return self.action_type == action_move_troops

	def is_end(self):
		return self.action_type == action_end_turn

class Attack_Action(Action):
	def __init__(self, from_country, to_country):
		Action.__init__(self, action_attack_country)
		self.from_country = from_country
		self.to_country = to_country

	def to_string(self):
		to_ret = Action.to_string(self)
		to_ret += "{} -> {}\n".format(self.from_country, self.to_country)
		return to_ret

class Place_Action(Action):
	def __init__(self, country, num_troops):
		Action.__init__(self, action_place_troops)
		self.country = country
		self.num_troops = num_troops

	def to_string(self):
		to_ret = Action.to_string(self)
		to_ret += "Place {} troops at {}\n".format(self.num_troops, self.country)
		return to_ret

class Fortify_Action(Action):
	def __init__(self, from_country, to_country, num_troops):
		Action.__init__(self, action_move_troops)
		self.from_country = from_country
		self.to_country = to_country
		self.num_troops = num_troops

	def to_string(self):
		to_ret = Action.to_string(self)
		to_ret += "{} -> {}\n".format(self.from_country, self.to_country)
		to_ret += "Num troops {}\n".format(self.num_troops)
 		return to_ret

class End_Action(Action):
	def __init__(self):
		Action.__init__(self, action_end_turn)