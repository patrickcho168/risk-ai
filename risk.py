import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pickle
from mapGen import ClassicWorldMap
from riskMDP import RiskMDP
from QLearning import QLearningAlgorithm
from uct import *
from HeuristicPlayer import HeuristicPlayer
import random

player_qlearning = 'QLearning'
player_random = 'Random'
player_heuristic = 'Heuristic'
player_uct = 'UCT'

def simulate(mdp, rl, hp, numTrials=10, maxIterations=1000000, verbose=False,
             sort=False, showTime=0.5, show=False, do_explore=True, players=[]):
    def sample(probs):
        return np.random.choice(len(probs), p=probs)

    def play_random(state):
        return random.choice(mdp.actions(state))

    if verbose:
        plt.ion()
        plt.show()
    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        if trial % 5 == 0:
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
            if players[curr_player] == player_qlearning:
                if state.is_attack():
                    action = rl.getAction(state, do_explore)
                else:
                    action = hp.getAction(state)
            elif players[curr_player] == player_random:
                action = play_random(state)
            elif players[curr_player] == player_heuristic:
                action = hp.getAction(state)
            elif players[curr_player] == player_uct:
                if state.is_attack():
                    action = rl.select_action(state, d=5)
                else:
                    action = hp.getAction(state)

            else:
                raise ValueError("Unsupported AI {}")
            if verbose and action.is_attack():
                print "-----%s-----" %iterationNumber
                print "Player Number: %s" %state.curr_player
                print "State: " + state.to_string()
                print "Action: " + action.to_string()
                if show:
                    mdp.drawState(state, show=True, showTime=showTime)
            
            newState, reward = mdp.simulate_action(state, action)
            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)

          
            for player in range(mdp.numberOfPlayers):
                totalReward[player] += totalDiscount * reward[player]
            totalDiscount *= mdp.discount()
            state = newState

        # ### Save Weights
        # with open('weights.pkl', 'wb') as f:
        #     pickle.dump(rl.weights, f, pickle.HIGHEST_PROTOCOL)
        # if verbose:
        #     print "Trial %d (totalReward = %s): %s" % (trial, totalReward, sequence)
        totalRewards.append(totalReward)

    return totalRewards

if __name__ == "__main__":
    #worldMap = ClassicWorldMap("classicWorldMap.csv", "classicWorldMapCoordinates.csv")
    # worldMap = ClassicWorldMap("smallWorldMap.csv", "smallWorldMapCoordinates.csv")
    worldMap = ClassicWorldMap("mediumWorldMap.csv", "mediumWorldMapCoordinates.csv")
    numberOfPlayers = 2
    mdp = RiskMDP(worldMap, 2, verbose=True)
    rl = QLearningAlgorithm(mdp.actions, mdp.discount(), mdp.smartFeatures)
    hp = HeuristicPlayer(worldMap, mdp)
    uct = UCT(mdp, hp)

    num_trails = 100


    players = [player_uct, player_uct]
    #rewards = simulate(mdp, rl, hp, numTrials=num_trails, verbose=False, players=players)
    rewards = simulate(mdp, uct, hp, numTrials=num_trails, verbose=False, players=players)
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

    players = [player_uct, player_heuristic]
    rewards = simulate(mdp, uct, hp, numTrials=num_trails, verbose=False, do_explore=False, players=players)
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

    players = [player_heuristic, player_uct]
    rewards = simulate(mdp, uct, hp, numTrials=num_trails, verbose=False, do_explore=False, players=players)
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

    # players = [player_heuristic, player_heuristic]
    # rewards = simulate(mdp, uct, hp, numTrials=num_trails, verbose=False, do_explore=False, players=players)
    # player0Rewards = 0
    # player1Rewards = 0
    # p0_wins = 0
    # player0RewardsSequence = []
    # for reward in rewards:
    #     if reward[0] > 0:
    #         p0_wins += 1
    #     player0Rewards += reward[0]
    #     player1Rewards += reward[1]
    #     player0RewardsSequence.append(reward[0])
    # # print player0RewardsSequence
    # print players
    # print "Player 0 Total Reward: %s" %player0Rewards
    # print "Player 1 Total Reward: %s" %player1Rewards
    # print "Player 0 win rate: {}".format(p0_wins/float(len(player0RewardsSequence)))
