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

    # Return the Q function associated with the weights and features
    def getQ(self, state, action, playerNumber):
        score = 0
        for f, v in self.featureExtractor(state, action, playerNumber):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        gameState, playerNumber, countryMap, additionalParameter = state
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action, playerNumber), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        if newState is None:
            return
        playerNum = state[1]
        Qhat = self.getQ(state, action, playerNum)
        Vhat = max([self.getQ(newState, act, playerNum) for act in self.actions(newState)])
        features = self.featureExtractor(state, action, playerNum)
        for f, v in features:
            self.weights[f] = self.weights[f] - self.getStepSize() * (Qhat - (reward[playerNum] + self.discount * Vhat)) * v