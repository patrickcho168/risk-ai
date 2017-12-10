import random
import math
from collections import defaultdict

class QLearningAlgorithm():
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0
        self.SARS_buffer = []
        self.SARS_buffer_max_size = 10
        self.SARS_buffer_bias = 0.5

    def flush_data(self):
        self.SARS_buffer = []
        self.weights = defaultdict(float)

    # Return the Q function associated with the weights and features
    def getQ(self, state, action, playerNumber):
        score = 0
        for f, v in self.featureExtractor(state, action, playerNumber):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state, do_explore=True):
        
        def pick_random_best(score_action_pairs):
            best_score = max(score_action_pairs)[0]
            # print best_score
            candidates = [p for p in score_action_pairs if p[0] == best_score]
            return random.choice(candidates)[1]

        self.numIters += 1
        if do_explore and random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return pick_random_best([(self.getQ(state, action, state.curr_player), action) for action in self.actions(state)])

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 0.1
        # return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        if newState is None:
            return

        if not state.is_attack():
            return

        #add some sars to the buffer
        if newState.is_end() or random.random() < self.SARS_buffer_bias:
            self.SARS_buffer.append((state, action, reward, newState))

        if len(self.SARS_buffer) == self.SARS_buffer_max_size:
            self.SARS_buffer = self.SARS_buffer[::-1] #reverse the buffer
            for state, action, reward, newState in self.SARS_buffer:
                next_player = newState.curr_player
                for player in range(len(reward)):
                    Qhat = self.getQ(state, action, player)
                    if newState.is_end():
                        Vhat = 0
                    else:
                        if player == next_player:
                            Vhat = max([self.getQ(newState, act, next_player) for act in self.actions(newState)])
                        else:
                            Vhat = -max([self.getQ(newState, act, next_player) for act in self.actions(newState)])

                    features = self.featureExtractor(state, action, player)
                    for f, v in features:
                        self.weights[f] = self.weights[f] - self.getStepSize() * (Qhat - (reward[player] + self.discount * Vhat)) * v
            self.SARS_buffer = []
            return max(self.weights.values())
