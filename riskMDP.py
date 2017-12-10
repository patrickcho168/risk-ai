import copy
import numpy as np
from actions import *
from states import *

class RiskMDP:
    def __init__(self, board_map, numberOfPlayers, verbose=False):
        self.board_map = board_map
        assert(numberOfPlayers >= 2 and numberOfPlayers <= 6)
        self.numberOfPlayers = numberOfPlayers
        self.verbose = verbose
        self.attackReward = 1
        self.winRewardFactor = 100
        self.noGameReward = [0] * self.numberOfPlayers
        self.fortify_max_counter = 5
        self.fortify_counter = 0
      
        # Initialize Attack Memo for a single round of dice throw
        # Refer to http://web.mit.edu/sp.268/www/risk.pdf
        self.attackMemo = {}
        self.attackMemo[(3,2)] = {}
        self.attackMemo[(3,2)][(0,2)] = 0.372
        self.attackMemo[(3,2)][(1,1)] = 0.336
        self.attackMemo[(3,2)][(2,0)] = 0.292 # Sum to 1
        self.attackMemo[(2,2)] = {}
        self.attackMemo[(2,2)][(0,2)] = 0.228
        self.attackMemo[(2,2)][(1,1)] = 0.324
        self.attackMemo[(2,2)][(2,0)] = 0.448 # Sum to 1
        self.attackMemo[(1,2)] = {}
        self.attackMemo[(1,2)][(0,1)] = 0.255
        self.attackMemo[(1,2)][(1,0)] = 0.745 # Sum to 1
        self.attackMemo[(3,1)] = {}
        self.attackMemo[(3,1)][(0,1)] = 0.660
        self.attackMemo[(3,1)][(1,0)] = 0.340 # Sum to 1
        self.attackMemo[(2,1)] = {}
        self.attackMemo[(2,1)][(0,1)] = 0.579
        self.attackMemo[(2,1)][(1,0)] = 0.421 # Sum to 1
        self.attackMemo[(1,1)] = {}
        self.attackMemo[(1,1)][(0,1)] = 0.417
        self.attackMemo[(1,1)][(1,0)] = 0.583 # Sum to 1

        # Memoize battles
        self.battleMemo = {}

    ###### START HELPER FUNCITONS ######

    def getEmptyCountries(self, state):
        numberOfCountries = self.board_map.numberOfCountries()
        listOfEmptyCountries = []
        for i in range(numberOfCountries):
            if i not in state.country_mapping:
                listOfEmptyCountries.append(i)
        return listOfEmptyCountries

    def getPlayerCountries(self, state):

        listOfPlayerCountries = []
        for country, countryState in state.country_mapping.iteritems():
            countryPlayer, countryTroops = countryState
            if countryPlayer == state.curr_player:
                listOfPlayerCountries.append(country)
        return listOfPlayerCountries

    def getNextPlayer(self, curr_player):
        return (curr_player + 1)%self.numberOfPlayers

    def ownsContinent(self, curr_player, country_mapping, continent):
        countries = self.board_map.getContinentCountries(continent)
        if not country_mapping:
            return False
        for country in countries:
            if country not in country_mapping or country_mapping[country][0] != curr_player:
                return False
        return True

    def getTroopCount(self, state):
        assert state.is_place()
        # Add number of countries owned / 3 floored
        troopCount = 0
        numberOfPlayerCountries = len(self.getPlayerCountries(state))
        troopCount += numberOfPlayerCountries / 3
        # Add based on continents owned
        numberOfContinents = self.board_map.numberOfContinents()
        for continent in range(numberOfContinents):
            if self.ownsContinent(state.curr_player, state.country_mapping, continent):
                troopCount += self.board_map.getContinentReward(continent)
        return max(3, troopCount)

    def getPossibleMoveTroops(self, state):
        assert state.is_fortify()
        possible_actions = [End_Action()]
        if self.fortify_counter <= self.fortify_max_counter:
            playerCountries = self.getPlayerCountries(state)
            for from_country in playerCountries:
                for to_country in self.reachable(state.country_mapping, from_country):
                        num_troops = state.country_mapping[from_country][1]-1
                        if num_troops > 0:
                            possible_actions.append(Fortify_Action(from_country, to_country, num_troops))
        return possible_actions

    def reachable(self, country_mapping, from_country):
        curr_player = country_mapping[from_country][0]

        # DFS
        countryList = [from_country]
        visited = set([from_country])
        while len(countryList) > 0:
            nextCountry = countryList.pop()
            allNeighbors = self.board_map.bordered(nextCountry)
            for neighbor in allNeighbors:
                if country_mapping[neighbor][0] == curr_player and neighbor not in visited:
                    visited.add(neighbor)
                    countryList.append(neighbor)
        visited.remove(from_country)
        return visited

    def simulate_action(self, state, action):
        def sample(probs):
            return np.random.choice(len(probs), p=probs)

        transitions = self.succAndProbReward(state, action)
        i = sample([prob for newState, prob, reward in transitions])
        newState, prob, reward = transitions[i]
        return newState, reward

    def simulateAttack(self, state, action):
      
        country_mapping = state.country_mapping
        attackingCountry = action.from_country
        defendingCountry = action.to_country
        defendingPlayer = country_mapping[defendingCountry][0]
        results = []

        # Cannot attack own country
        assert(country_mapping[attackingCountry][0] != country_mapping[defendingCountry][0])       
        numberOfAttackingTroops = country_mapping[attackingCountry][1]
        numberOfDefendingTroops = country_mapping[defendingCountry][1]

        # Check Memo for Same Battle
        if (numberOfAttackingTroops, numberOfDefendingTroops) not in self.battleMemo:
            self.getAttackProbability(numberOfAttackingTroops, numberOfDefendingTroops)

        for result, probability in self.battleMemo[(numberOfAttackingTroops, numberOfDefendingTroops)].iteritems():
            attackersLeft = result[0]
            defendersLeft = result[1]
            newcountry_mapping = copy.deepcopy(country_mapping)
            # Attackers Won!
            if defendersLeft == 0:
                newcountry_mapping[defendingCountry] = (country_mapping[attackingCountry][0], attackersLeft - 1) # TAKE OVER TERRITORY
                newcountry_mapping[attackingCountry] = (country_mapping[attackingCountry][0], 1)
                attack_success = True
            # Attackers Lost!
            elif attackersLeft == 1:
                newcountry_mapping[defendingCountry] = (country_mapping[defendingCountry][0], defendersLeft)
                newcountry_mapping[attackingCountry] = (country_mapping[attackingCountry][0], attackersLeft)
                attack_success = False
            else:
                raise ValueError("Either Attacker Loses or Defender Loses!")
            new_state = Attack_State(newcountry_mapping, state.curr_player)
            winner = self.getWinner(new_state)
            if winner == -1: # No winner
                if attack_success:
                    attackSuccessReward = []
                    for i in range(self.numberOfPlayers):
                        if i == state.curr_player:
                            attackSuccessReward.append(self.attackReward)
                        elif i == defendingPlayer:
                            attackSuccessReward.append(-0.9*self.attackReward) #not as heavy a penalty for losing a state
                        else:
                            attackSuccessReward.append(0)
                    results.append((new_state, probability, attackSuccessReward))
                else:
                    results.append((new_state, probability, self.noGameReward))
            else:
                assert(winner == state.curr_player)
                endGameReward = []
                for i in range(self.numberOfPlayers):
                    if i != winner:
                        endGameReward.append(-1*self.winRewardFactor)
                    else:
                        endGameReward.append((self.numberOfPlayers-1)*self.winRewardFactor)
                new_state = End_State(newcountry_mapping, state.curr_player)
                results.append((new_state, probability, endGameReward))
        return results
    
    def getAttackProbability(self, attackers, defenders):
        if (attackers, defenders) in self.battleMemo:
            return self.battleMemo[(attackers, defenders)]
        numberOfAttackerDice = min(3, attackers-1)
        numberOfDefenderDice = min(2, defenders)
        finalResult = {}
        for result, probability in self.attackMemo[(numberOfAttackerDice, numberOfDefenderDice)].iteritems():
            nextAttackers = attackers - result[0]
            nextDefenders = defenders - result[1]
            if nextDefenders == 0:
                nextResult = {(nextAttackers, nextDefenders): 1}
            elif nextAttackers == 1:
                nextResult = {(nextAttackers, nextDefenders): 1}
            else:
                nextResult = self.getAttackProbability(nextAttackers, nextDefenders)
            for aResult, aProbability in nextResult.iteritems():
                finalResult[aResult] = finalResult.get(aResult, 0) + aProbability * probability
        self.battleMemo[(attackers, defenders)] = finalResult
        return finalResult

    # Returns -1 if no winner yet
    # Else returns curr_player of winner
    def getWinner(self, state):
        country_mapping = state.country_mapping
        winningPlayer = country_mapping[0][0]
        for country, countryState in country_mapping.iteritems():
            countryPlayer, countryTroops = countryState
            if winningPlayer != countryPlayer:
                return -1
        return winningPlayer

    ###### END HELPER FUNCITONS ######

    def startState(self):
        firstPlayer = 0
        country_mapping = {}
        if self.numberOfPlayers == 2:
            starting_troops = 40
        elif self.numberOfPlayers == 3:
            starting_troops = 35
        elif self.numberOfPlayers == 4:
            starting_troops = 30
        elif self.numberOfPlayers == 5:
            starting_troops = 25
        elif self.numberOfPlayers == 6:
            starting_troops = 20
        else:
            raise ValueError("2-6 Players Only")
        return Setup_State(country_mapping, firstPlayer, starting_troops)

    def actions(self, state):
        listOfActions = []

        # Setup State. Player gets to place troops in any empty countries.
        # When no more empty countries, place in own territory.
        if state.is_setup():
            troops_to_place = state.troops_to_place
            emptyCountries = self.getEmptyCountries(state)
            assert(troops_to_place > 0)
            if len(emptyCountries) > 0:
                for country in emptyCountries:
                    listOfActions.append(Place_Action(country, 1))
            else:
                playerCountries = self.getPlayerCountries(state)
                for country in playerCountries:
                    listOfActions.append(Place_Action(country, 1))
            return listOfActions
        
        # Place Troops State. 
        # Player gets to place troops in any territory he owns.
        elif state.is_place():
            troops_to_place = state.troops_to_place
            assert troops_to_place > 0
            playerCountries = self.getPlayerCountries(state)
            for country in playerCountries:
                listOfActions.append(Place_Action(country, 1))
            return listOfActions
        
        # Attack State.
        # Player gets to end turn or attack any adjacent country.
        elif state.is_attack():
            country_mapping = state.country_mapping
            curr_player = state.curr_player
            listOfActions = [End_Action()]
            playerCountries = self.getPlayerCountries(state)
            for country in playerCountries:
                assert country_mapping[country][0] == curr_player
                # Must have more than one troop to attack!
                if country_mapping[country][1] > 1:
                    allNeighbors = self.board_map.bordered(country)
                    for neighborCountry in allNeighbors:
                        # Cannot attack yourself!
                        if country_mapping[neighborCountry][0] != curr_player:
                            listOfActions.append(Attack_Action(country, neighborCountry))
            return listOfActions
        
        # Fortify State.
        # Move from one country to the next to decide whether or not to move troops
        elif state.is_fortify():
            return self.getPossibleMoveTroops(state)
        
        elif state.is_end():
            raise ValueError("Game has ended")
            # return ['Celebrate!']

        else:
            raise ValueError("Impossible Game State")

    def succAndProbReward(self, state, action):
        results = []
        noGameReward = self.noGameReward
        attackReward = self.attackReward
        winRewardFactor = self.winRewardFactor
        newcountry_mapping = copy.deepcopy(state.country_mapping)
        curr_player = state.curr_player

        if state.is_end():
            raise ValueError("Cannot take actions in end state")
               
        elif state.is_setup():
            assert action.is_place()
            country = action.country
            num_troops = action.num_troops

            # In setup state
            if country not in newcountry_mapping:
                newcountry_mapping[country] = (curr_player, num_troops)
            else:
                newcountry_mapping[country] = (curr_player, newcountry_mapping[country][1] + num_troops)
            nextPlayer = self.getNextPlayer(curr_player)
            # One round has ended!
            if nextPlayer == 0:
                # More troops to place
                if state.troops_to_place > num_troops:
                    next_state = Setup_State(newcountry_mapping, nextPlayer, state.troops_to_place-num_troops)
                    results.append((next_state, 1, noGameReward))
                # Let the game begin
                else:
                    next_state = Place_State(newcountry_mapping, nextPlayer, 0)
                    troopCount = self.getTroopCount(next_state)
                    next_state = Place_State(newcountry_mapping, nextPlayer, troopCount)
                    results.append((next_state, 1, noGameReward))
            # The round is continuing
            else:
                next_state = Setup_State(newcountry_mapping, nextPlayer, state.troops_to_place)
                results.append((next_state, 1, noGameReward))
        
        # In place state
        elif state.is_place():
            assert action.is_place()
            country = action.country
            num_troops = action.num_troops
            
            newcountry_mapping[country] = (curr_player, newcountry_mapping[country][1] + num_troops)
            # More troops to place later
            if state.troops_to_place > num_troops:
                next_state = Place_State(newcountry_mapping, curr_player, state.troops_to_place-num_troops)
                results.append((next_state, 1, noGameReward))
            # Move on to next game state
            else:
                next_state = Attack_State(newcountry_mapping, curr_player)
                results.append((next_state, 1, noGameReward))
            
        elif state.is_attack():
            if action.is_end():
                next_state = Fortify_State(newcountry_mapping, curr_player)
                results.append((next_state, 1, noGameReward))
            else:
                return self.simulateAttack(state, action)
        
        elif state.is_fortify():
            if action.is_end():
                self.fortify_counter = 0
                next_state = Place_State(newcountry_mapping, self.getNextPlayer(curr_player), 0)
                troopCount = self.getTroopCount(next_state)
                next_state = Place_State(newcountry_mapping, self.getNextPlayer(curr_player), troopCount)
                results.append((next_state, 1, noGameReward))
            else:
                self.fortify_counter += 1
                from_country = action.from_country
                to_country = action.to_country
                num_troops = action.num_troops


                newcountry_mapping[from_country] = (newcountry_mapping[from_country][0], newcountry_mapping[from_country][1] - num_troops)
                newcountry_mapping[to_country] = (newcountry_mapping[to_country][0], newcountry_mapping[to_country][1] + num_troops)

                next_state = Fortify_State(newcountry_mapping, curr_player) #can only fortify once
                results.append((next_state, 1, noGameReward))
        
        else:
            raise ValueError("Impossible State Type: %s" %state)
        
        return results

    def discount(self):
        return 1

    def drawState(self, state, show=True, showTime=0.05):
        self.board_map.drawState(state.country_mapping, show, showTime)

    def featureExtractor(self, state, action, playerNum):
        gameState, curr_player, country_mapping, additionalParameter = state
        features = []
        for country, countryState in country_mapping.iteritems():
            countryPlayer = countryState[0]
            countryTroops = countryState[1]
            if countryPlayer == playerNum:
                features.append(1+countryTroops/5)
            else:
                features.append(-(1+countryTroops/5))
        featureKey = (tuple(features), action)
        featureValue = 1
        return [(featureKey, featureValue)]

    #TODO: add features for-
    #number of clusters of countries
    #border size / opp neighbor size
    #border troop total / opp neighbor border troop total
    #proportion of excess troops on border
    def smartFeatures(self, state, action, playerNum):
        curr_player = state.curr_player
        country_mapping = state.country_mapping
        features = []
        my_country_states = []
        opp_country_states = []
        ownership_feature = []
        my_troop_count_list = []
        num_my_troops = 0
        num_opp_troops = 0
        for country, countryState in country_mapping.iteritems():
            countryPlayer = countryState[0]
            countryTroops = countryState[1]
            if countryPlayer == playerNum:
                my_country_states.append(countryState)
                num_my_troops += countryTroops
                my_troop_count_list.append(countryTroops)
                ownership_feature.append(1)
            else:
                opp_country_states.append(countryState)
                num_opp_troops += countryTroops
                ownership_feature.append(-1)

        num_my_countries = len(my_country_states)
        num_opp_countries = len(opp_country_states)

        my_continent_bonus = 0
        opp_continent_bonus = 0 
        total_continent_count = self.board_map.numberOfContinents()
        for continent in range(total_continent_count):
            if self.ownsContinent(playerNum, country_mapping, continent):
                my_continent_bonus += self.board_map.getContinentReward(continent)
            for opp in range(self.numberOfPlayers):
                if opp != playerNum:
                    if self.ownsContinent(opp, country_mapping, continent):
                        opp_continent_bonus += self.board_map.getContinentReward(continent)

        def safe_ratio(x, y, base):
            return min([(base+x)/(base+y), 5])

        features.append((tuple([safe_ratio(num_my_troops, num_opp_troops, 1), 'my_troop_ratio']), 1))
        features.append((tuple([safe_ratio(num_opp_troops, num_my_troops, 1), 'opp_troop_ratio']), 1))

        features.append((((num_my_troops-num_opp_troops)/5, 'troop_diff'), 1))

        features.append((tuple([safe_ratio(num_my_countries, num_opp_countries, 1), 'my_country_ratio']), 1))
        features.append((tuple([safe_ratio(num_opp_countries, num_my_countries, 1), 'opp_country_ratio']), 1))

        features.append((((num_my_countries-num_opp_countries)/2, 'country_diff'), 1))

        features.append((tuple([safe_ratio(my_continent_bonus, opp_continent_bonus, 1), 'my_continent_ratio']), 1))
        features.append((tuple([safe_ratio(opp_continent_bonus, my_continent_bonus, 1), 'opp_continent_ratio']), 1))

        features.append((((my_continent_bonus-opp_continent_bonus)/2, 'cont_diff'), 1))

        turn_indicator = -1
        if playerNum==curr_player:
            turn_indicator = 1
        features.append(('turn', turn_indicator))
        
        attack_indicator = -1*turn_indicator
        if action.is_attack():
            attack_indicator *= -1
        features.append(('is_attack', attack_indicator))

        return features

    def evaluate(self, state, playerNum):
        curr_player = state.curr_player
        country_mapping = state.country_mapping
        features = []
        my_country_states = []
        opp_country_states = []
        ownership_feature = []
        my_troop_count_list = []
        num_my_troops = 0
        num_opp_troops = 0
        for country, countryState in country_mapping.iteritems():
            countryPlayer = countryState[0]
            countryTroops = countryState[1]
            if countryPlayer == playerNum:
                my_country_states.append(countryState)
                num_my_troops += countryTroops
                my_troop_count_list.append(countryTroops)
                ownership_feature.append(1)
            else:
                opp_country_states.append(countryState)
                num_opp_troops += countryTroops
                ownership_feature.append(-1)

        num_my_countries = len(my_country_states)
        num_opp_countries = len(opp_country_states)

        my_continent_bonus = 0
        opp_continent_bonus = 0 
        total_continent_count = self.board_map.numberOfContinents()
        for continent in range(total_continent_count):
            if self.ownsContinent(playerNum, country_mapping, continent):
                my_continent_bonus += self.board_map.getContinentReward(continent)
            for opp in range(self.numberOfPlayers):
                if opp != playerNum:
                    if self.ownsContinent(opp, country_mapping, continent):
                        opp_continent_bonus += self.board_map.getContinentReward(continent)

        def safe_ratio(x, y, base):
            return min([(base+x)/(base+y), 5])

        my_troop_ratio = safe_ratio(num_my_troops, num_opp_troops, 1)
        opp_troop_ratio = safe_ratio(num_opp_troops, num_my_troops, 1)
        troop_diff = num_my_troops-num_opp_troops
        
        my_country_ratio = safe_ratio(num_my_countries, num_opp_countries, 1)
        opp_country_ratio = safe_ratio(num_opp_countries, num_my_countries, 1)
        country_diff = (num_my_countries-num_opp_countries)

        my_continent_ratio = safe_ratio(my_continent_bonus, opp_continent_bonus, 1)
        opp_continent_ratio = safe_ratio(opp_continent_bonus, my_continent_bonus, 1)
        cont_diff = my_continent_bonus-opp_continent_bonus

        turn_indicator = -1
        if playerNum==curr_player:
            turn_indicator = 1

        evaluation = 0
        evaluation += troop_diff
        evaluation += 10 * my_troop_ratio
        evaluation -= 10 * opp_troop_ratio
        evaluation += 5 * country_diff
        evaluation += 40 * my_country_ratio
        evaluation -= 40 * opp_continent_ratio
        evaluation += cont_diff
        evaluation += 20 * my_continent_ratio
        evaluation -= 20 * opp_continent_ratio
        evaluation += 3 * turn_indicator

        return min(evaluation, 0.95 * self.winRewardFactor)

    def state_to_feature_vect(self, state, playerNum):
        if state.is_end():
            return ["END"]
        curr_player = state.curr_player
        country_mapping = state.country_mapping
        features = []
        my_country_states = []
        opp_country_states = []
        ownership_feature = []
        my_troop_count_list = []
        num_my_troops = 0
        num_opp_troops = 0
        for country, countryState in country_mapping.iteritems():
            countryPlayer = countryState[0]
            countryTroops = countryState[1]
            if countryPlayer == playerNum:
                my_country_states.append(countryState)
                num_my_troops += countryTroops
                my_troop_count_list.append(countryTroops)
                ownership_feature.append(1)
            else:
                opp_country_states.append(countryState)
                num_opp_troops += countryTroops
                ownership_feature.append(-1)

        num_my_countries = len(my_country_states)
        num_opp_countries = len(opp_country_states)

        my_continent_bonus = 0
        opp_continent_bonus = 0 
        total_continent_count = self.board_map.numberOfContinents()
        for continent in range(total_continent_count):
            if self.ownsContinent(playerNum, country_mapping, continent):
                my_continent_bonus += self.board_map.getContinentReward(continent)
            for opp in range(self.numberOfPlayers):
                if opp != playerNum:
                    if self.ownsContinent(opp, country_mapping, continent):
                        opp_continent_bonus += self.board_map.getContinentReward(continent)

        def safe_ratio(x, y, base):
            return min([(base+x)/(base+y), 5])

        my_troop_ratio = safe_ratio(num_my_troops, num_opp_troops, 1)
        opp_troop_ratio = safe_ratio(num_opp_troops, num_my_troops, 1)
        troop_diff = (num_my_troops-num_opp_troops) / 5
        
        my_country_ratio = safe_ratio(num_my_countries, num_opp_countries, 1)
        opp_country_ratio = safe_ratio(num_opp_countries, num_my_countries, 1)
        country_diff = (num_my_countries-num_opp_countries) / 2

        my_continent_ratio = safe_ratio(my_continent_bonus, opp_continent_bonus, 1)
        opp_continent_ratio = safe_ratio(opp_continent_bonus, my_continent_bonus, 1)
        cont_diff = (my_continent_bonus-opp_continent_bonus) / 2

        turn_indicator = -1
        if playerNum==curr_player:
            turn_indicator = 1

        game_phase = state.game_phase
        valid_actions = sorted(self.actions(state))

        features = tuple([my_troop_ratio, opp_troop_ratio, troop_diff, my_country_ratio, opp_country_ratio, \
                        country_diff, my_continent_ratio, opp_continent_ratio, cont_diff, turn_indicator, \
                        game_phase, tuple(valid_actions)])
        
        return features