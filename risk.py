import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pickle
from mapGen import ClassicWorldMap
from riskMDP import RiskMDP
from QLearning import QLearningAlgorithm
from HeuristicPlayer import HeuristicPlayer

player_qlearning = 'QLearning'
player_random = 'Random'
player_heuristic = 'Heuristic'

def simulate(mdp, rl, hp, numTrials=10, maxIterations=1000000, verbose=False,
             sort=False, showTime=0.5, show=False, do_explore=True, players=[]):
    def sample(probs):
        return np.random.choice(len(probs), p=probs)

    if verbose:
        plt.ion()
        plt.show()
    totalRewards = []  # The rewards we get on each trial
    weight_max = []
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
            if state.is_end():
                break
            curr_player = state.curr_player
            if state.is_place():
                action = hp.getAction(state)
            elif not state.is_setup() and players[curr_player] == player_heuristic:
                action = hp.getAction(state)
            elif not state.is_attack() or players[curr_player] == player_random:
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
            
            newState, reward = mdp.simulate_action(state, action)
            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)

            highest_weight = rl.incorporateFeedback(state, action, reward, newState)
            if highest_weight:
                weight_max.append(highest_weight)
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
    plt.plot(xrange(len(weight_max)), weight_max)
    plt.show()
    return totalRewards

if __name__ == "__main__":
    # worldMap = ClassicWorldMap("classicWorldMap.csv", "classicWorldMapCoordinates.csv")
    # worldMap = ClassicWorldMap("smallWorldMap.csv", "smallWorldMapCoordinates.csv")
    worldMap = ClassicWorldMap("mediumWorldMap.csv", "mediumWorldMapCoordinates.csv")
    numberOfPlayers = 2
    mdp = RiskMDP(worldMap, 2, verbose=True)
    rl = QLearningAlgorithm(mdp.actions, mdp.discount(), mdp.smartFeatures)
    hp = HeuristicPlayer(worldMap, mdp)

    num_trails = 500

    players = [player_qlearning, player_qlearning]
    rewards = simulate(mdp, rl, hp, numTrials=num_trails, verbose=False, players=players)
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
    print players
    print "Player 0 Total Reward: %s" %player0Rewards
    print "Player 1 Total Reward: %s" %player1Rewards
    print "Player 0 win rate: {}".format(p0_wins/float(len(player0RewardsSequence)))

    players = [player_qlearning, player_heuristic]
    rewards = simulate(mdp, rl, hp, numTrials=num_trails, verbose=False, do_explore=False, players=players)
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
    print players
    print "Player 0 Total Reward: %s" %player0Rewards
    print "Player 1 Total Reward: %s" %player1Rewards
    print "Player 0 win rate: {}".format(p0_wins/float(len(player0RewardsSequence)))

    players = [player_heuristic, player_qlearning]
    rewards = simulate(mdp, rl, hp, numTrials=num_trails, verbose=False, do_explore=False, players=players)
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
    print players
    print "Player 0 Total Reward: %s" %player0Rewards
    print "Player 1 Total Reward: %s" %player1Rewards
    print "Player 0 win rate: {}".format(p0_wins/float(len(player0RewardsSequence)))

    players = [player_heuristic, player_heuristic]
    rewards = simulate(mdp, rl, hp, numTrials=num_trails, verbose=False, do_explore=False, players=players)
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
    print players
    print "Player 0 Total Reward: %s" %player0Rewards
    print "Player 1 Total Reward: %s" %player1Rewards
    print "Player 0 win rate: {}".format(p0_wins/float(len(player0RewardsSequence)))

    # rewards = simulate(mdp, rl, numTrials=1, verbose=True, show=False, do_explore=False, player1_random=True)
