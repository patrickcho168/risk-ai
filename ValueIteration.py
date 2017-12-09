import collections

class ValueIteration():
    '''
    Solve the MDP using value iteration.  Your solve() method must set
    - self.V to the dictionary mapping states to optimal values
    - self.pi to the dictionary mapping states to an optimal action
    Note: epsilon is the error tolerance: you should stop value iteration when
    all of the values change by less than epsilon.
    The ValueIteration class is a subclass of util.MDPAlgorithm (see util.py).
    '''
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        def computeQ(mdp, V, state, action):
            # Return Q(state, action) based on V(state).
            try:
                return sum(prob * (reward[state.curr_player] + mdp.discount() * \
                            V[newState] if newState.curr_player == state.curr_player else -V[newState]) \
                            for newState, prob, reward in mdp.succAndProbReward(state, action))
            except:
                for newState, prob, reward in mdp.succAndProbReward(state, action):
                    print newState
                raise ValueError('No State Found')

        def computeOptimalPolicy(mdp, V):
            # Return the optimal policy given the values V.
            pi = {}
            for state in mdp.states:
                pi[state] = max((computeQ(mdp, V, state, action), action) for action in mdp.actions(state))[1]
            return pi

        V = collections.defaultdict(float)  # state -> value of state
        numIters = 0
        while True:
            print "Running Iteration : {}".format(numIters)
            newV = {}
            numStates = 0
            for state in mdp.states:
                numStates += 1
                if numStates % 1000 == 0:
                    print "Done {}/{} states".format(numStates, len(mdp.states))
                if state.is_end():
                    newV[state] = 0
                    continue
                # This evaluates to zero for end states, which have no available actions (by definition)
                newV[state] = max(computeQ(mdp, V, state, action) for action in mdp.actions(state))
            numIters += 1
            maxError = max(abs(V[state] - newV[state]) for state in mdp.states)
            print "Max Error: {}".format(maxError)
            if maxError < epsilon:
                V = newV
                break
            V = newV

        # Compute the optimal policy now
        pi = computeOptimalPolicy(mdp, V)
        print "ValueIteration: %d iterations" % numIters
        self.pi = pi
        self.V = V