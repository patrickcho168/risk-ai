import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pickle
from mapGen import ClassicWorldMap
from riskMDP import RiskMDP
from QLearning import QLearningAlgorithm

def simulate(mdp, rl, numTrials=10, maxIterations=1000000, verbose=False,
             sort=False, showTime=0.5, show=False, do_explore=True, random_players=[]):
    def sample(probs):
        return np.random.choice(len(probs), p=probs)


    if verbose:
        plt.ion()
        plt.show()
    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        if trial % 50 == 0:
            print "Trial Number: %s" %trial
        state = mdp.startState()
        sequence = [state]
        totalDiscount = 1
        totalReward = []
        for player in range(mdp.numberOfPlayers):
            totalReward.append(0)
        for iterationNumber in range(maxIterations):
            turn = state[1]
            if random_players and turn in random_players:
                 action = rl.getAction(state, do_explore, play_random=True)
            else:
                action = rl.getAction(state, do_explore)
            if verbose and action[0]=='ATTACK_COUNTRY':
                print "-----%s-----" %iterationNumber
                print "Player Number: %s" %state[1]
                print "State: " + str(state)
                print "Action: " + str(action)
                if show:
                    mdp.drawState(state, show=True, showTime=showTime)
            transitions = mdp.succAndProbReward(state, action)
            if sort: transitions = sorted(transitions)
            if len(transitions) == 0:
                rl.incorporateFeedback(state, action, 0, None)
                break

            # Choose a random transition
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)

            rl.incorporateFeedback(state, action, reward, newState)
            for player in range(mdp.numberOfPlayers):
                totalReward[player] += totalDiscount * reward[player]
            totalDiscount *= mdp.discount()
            state = newState

        ### Save Weights
        with open('weights.pkl', 'wb') as f:
            pickle.dump(rl.weights, f, pickle.HIGHEST_PROTOCOL)
        # if verbose:
        #     print "Trial %d (totalReward = %s): %s" % (trial, totalReward, sequence)
        totalRewards.append(totalReward)
    return totalRewards

if __name__ == "__main__":
    # worldMap = ClassicWorldMap("classicWorldMap.csv", "classicWorldMapCoordinates.csv")
    # worldMap = ClassicWorldMap("smallWorldMap.csv", "smallWorldMapCoordinates.csv")
    worldMap = ClassicWorldMap("mediumWorldMap.csv", "mediumWorldMapCoordinates.csv")
    numberOfPlayers = 2
    mdp = RiskMDP(worldMap, 2, verbose=True)
    rl = QLearningAlgorithm(mdp.actions, mdp.discount(), mdp.smartFeatures)

    num_trails = 1000

    rewards = simulate(mdp, rl, numTrials=num_trails, verbose=False)

    player0Rewards = 0
    player1Rewards = 0
    player0RewardsSequence = []
    p0_wins = 0
    for reward in rewards:
        if reward[0] > 0:
            p0_wins += 1
        player0Rewards += reward[0]
        player1Rewards += reward[1]
        player0RewardsSequence.append(reward[0])

    # print player0RewardsSequence
    print "Player 0 Total Reward: %s" %player0Rewards
    print "Player 1 Total Reward: %s" %player1Rewards
    print "Player 0 win rate: {}".format(p0_wins/float(len(player0RewardsSequence)))

    rewards = simulate(mdp, rl, numTrials=num_trails, verbose=False, do_explore=False, random_players=[1])

    player0Rewards = 0
    player1Rewards = 0
    p0_wins = 0

    player0RewardsSequence = []
    for reward in rewards:
        if reward[0] > 0:
            p0_wins += 1
        player0Rewards += reward[0]
        player1Rewards += reward[1]
        player0RewardsSequence.append(reward[0])

    # print player0RewardsSequence
    print "Player 0 Total Reward: %s" %player0Rewards
    print "Player 1 Total Reward: %s" %player1Rewards
    print "Player 0 win rate: {}".format(p0_wins/float(len(player0RewardsSequence)))

    rewards = simulate(mdp, rl, numTrials=num_trails, verbose=False, do_explore=False, random_players=[0,1])

    player0Rewards = 0
    player1Rewards = 0
    p0_wins = 0
    player0RewardsSequence = []
    for reward in rewards:
        if reward[0] > 0:
            p0_wins += 1
        player0Rewards += reward[0]
        player1Rewards += reward[1]
        player0RewardsSequence.append(reward[0])

    # print player0RewardsSequence
    print "Player 0 Total Reward: %s" %player0Rewards
    print "Player 1 Total Reward: %s" %player1Rewards
    print "Player 0 win rate: {}".format(p0_wins/float(len(player0RewardsSequence)))

    # rewards = simulate(mdp, rl, numTrials=1, verbose=True, show=False, do_explore=False, player1_random=True)