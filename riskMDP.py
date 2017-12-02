import copy
import numpy as np

class RiskActions:
    def __init__(self):
        self.placeTroops = 'PLACE_TROOPS'
        self.attackCountry = 'ATTACK_COUNTRY'
        self.moveTroops = 'MOVE_TROOPS'
        self.endTurn = 'END_TURN'

class RiskStates:
    def __init__(self):
        self.setup = 'SETUP'
        self.place = 'PLACE'
        self.attack = 'ATTACK'
        self.fortify = 'FORTIFY'
        self.end = 'END'

class RiskMDP:
    def __init__(self, worldMap, numberOfPlayers, verbose=False):
        self.worldMap = worldMap
        assert(numberOfPlayers >= 2 and numberOfPlayers <= 6)
        self.numberOfPlayers = numberOfPlayers
        self.verbose = verbose

        # Setup Struct of Risk Game States
        self.gameStates = RiskStates()

        # Setup Struct of Risk Game Actions
        self.gameActions = RiskActions()
       
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
        countryMap = state[2]
        numberOfCountries = self.worldMap.numberOfCountries()
        listOfEmptyCountries = []
        for i in range(numberOfCountries):
            if i not in countryMap:
                listOfEmptyCountries.append(i)
        return listOfEmptyCountries

    def getPlayerCountries(self, state):
        countryMap = state[2]
        playerNum = state[1]
        listOfPlayerCountries = []
        for country, countryState in countryMap.iteritems():
            countryPlayer, countryTroops = countryState
            if countryPlayer == playerNum:
                listOfPlayerCountries.append(country)
        return listOfPlayerCountries

    def getNextPlayer(self, playerNum):
        return (playerNum + 1)%self.numberOfPlayers

    def ownsContinent(self, playerNum, countryMap, continent):
        countries = self.worldMap.getContinentCountries(continent)
        # print playerNum, countryMap, continent, countries
        if not countryMap:
            return False
        for country in countries:
            if country not in countryMap or countryMap[country][0] != playerNum:
                return False
        return True

    def getTroopCount(self, state):
        gameState, playerNumber, countryMap, additionalParameter = state
        assert(gameState == self.gameStates.place)

        # Add number of countries owned / 3 floored
        troopCount = 0
        numberOfPlayerCountries = len(self.getPlayerCountries(state))
        troopCount += numberOfPlayerCountries / 3



        # Add based on continents owned
        numberOfContinents = self.worldMap.numberOfContinents()
        for continent in range(numberOfContinents):
            if self.ownsContinent(playerNumber, countryMap, continent):
                troopCount += self.worldMap.getContinentReward(continent)
        return max(3, troopCount)

    def getPossibleMoveTroops(self, state):
        gameState, playerNumber, countryMap, additionalParameter = state
        playerCountries = self.getPlayerCountries(state)
        possibleMoves = []
        for country1 in playerCountries:
            for country2 in playerCountries:
                if self.canFortify(countryMap, country1, country2):
                    possibleMoves.append((country1, country2))
        return possibleMoves

    def canFortify(self, countryMap, country1, country2):
        # Same player owns the 2 countries
        assert(countryMap[country1][0] == countryMap[country2][0])
        playerNumber = countryMap[country1][0]

        # DFS
        countryList = [country1]
        visited = {country1: 1}
        while len(countryList) > 0:
            nextCountry = countryList.pop()
            allNeighbors = self.worldMap.bordered(nextCountry)
            for neighbor in allNeighbors:
                if neighbor == country2:
                    return True
                if countryMap[neighbor][0] == playerNumber and neighbor not in visited:
                    visited[neighbor] = 1
                    countryList.append(neighbor)
        return False

    def simulateAttack(self, attackingCountry, defendingCountry, state):
        gameState, playerNumber, countryMap, additionalParameter = state
        
        # Cannot attack own country
        assert(countryMap[attackingCountry][0] != countryMap[defendingCountry][0])       
        numberOfAttackingTroops = countryMap[attackingCountry][1]
        numberOfDefendingTroops = countryMap[defendingCountry][1]
        countryMapToProbabilityMapping = []

        # Check Memo for Same Battle
        if (numberOfAttackingTroops, numberOfDefendingTroops) not in self.battleMemo:
            self.getAttackProbability(numberOfAttackingTroops, numberOfDefendingTroops)

        for result, probability in self.battleMemo[(numberOfAttackingTroops, numberOfDefendingTroops)].iteritems():
            attackersLeft = result[0]
            defendersLeft = result[1]
            newCountryMap = copy.deepcopy(countryMap)
            # Attackers Won!
            if defendersLeft == 0:
                newCountryMap[defendingCountry] = (countryMap[attackingCountry][0], attackersLeft - 1) # TAKE OVER TERRITORY
                newCountryMap[attackingCountry] = (countryMap[attackingCountry][0], 1)
                countryMapToProbabilityMapping.append((newCountryMap, probability))
                attack_success = True
            # Attackers Lost!
            elif attackersLeft == 1:
                newCountryMap[defendingCountry] = (countryMap[defendingCountry][0], defendersLeft)
                newCountryMap[attackingCountry] = (countryMap[attackingCountry][0], attackersLeft)
                countryMapToProbabilityMapping.append((newCountryMap, probability))
                attack_success = False
            else:
                raise ValueError("Either Attacker Loses or Defender Loses!")
        return countryMapToProbabilityMapping, attack_success

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
    # Else returns playerNumber of winner
    def getWinner(self, state):
        gameState, playerNumber, countryMap, additionalParameter = state
        winningPlayer = countryMap[0][0]
        for country, countryState in countryMap.iteritems():
            countryPlayer, countryTroops = countryState
            if winningPlayer != countryPlayer:
                return -1
        return winningPlayer

    ###### END HELPER FUNCITONS ######

    def startState(self):
        firstPlayer = 0
        countryMap = {}
        if self.numberOfPlayers == 2:
            starting_troops = 40
            # countryMap[0] = (0, 1)
            # countryMap[1] = (0, 25)
            # countryMap[2] = (0, 1)
            # countryMap[6] = (0, 1)
            # countryMap[3] = (1, 1)
            # countryMap[4] = (1, 25)
            # countryMap[5] = (1, 1)
            # countryMap[7] = (1, 1)
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
        return (self.gameStates.setup, firstPlayer, countryMap, starting_troops)

    def actions(self, state):
        gameState, playerNumber, countryMap, additionalParameter = state
        listOfActions = []

        # Setup State. Player gets to place troops in any empty countries.
        # When no more empty countries, place in own territory.
        if gameState == self.gameStates.setup:
            emptyCountries = self.getEmptyCountries(state)
            assert(additionalParameter > 0)
            if len(emptyCountries) > 0:
                for country in emptyCountries:
                    listOfActions.append((self.gameActions.placeTroops, country, 1))
            else:
                playerCountries = self.getPlayerCountries(state)
                for country in playerCountries:
                    listOfActions.append((self.gameActions.placeTroops, country, 1))
            return listOfActions
        
        # Place Troops State. 
        # Player gets to place troops in any territory he owns.
        elif gameState == self.gameStates.place:
            assert(additionalParameter > 0)
            playerCountries = self.getPlayerCountries(state)
            for country in playerCountries:
                listOfActions.append((self.gameActions.placeTroops, country, 1))
            return listOfActions
        
        # Attack State.
        # Player gets to end turn or attack any adjacent country.
        elif gameState == self.gameStates.attack:
            listOfActions = [(self.gameActions.endTurn,)]
            playerCountries = self.getPlayerCountries(state)
            for country in playerCountries:
                assert countryMap[country][0] == playerNumber
                # Must have more than one troop to attack!
                if countryMap[country][1] > 1:
                    allNeighbors = self.worldMap.bordered(country)
                    for neighborCountry in allNeighbors:
                        # Cannot attack yourself!
                        if countryMap[neighborCountry][0] != playerNumber:
                            listOfActions.append((self.gameActions.attackCountry, country, neighborCountry))
            return listOfActions
        
        # Fortify State.
        # Move from one country to the next to decide whether or not to move troops
        elif gameState == self.gameStates.fortify:
            countryPairs = additionalParameter[0]
            countryMove = additionalParameter[1]
            countryPair = countryPairs[countryMove]
            fromCountry = countryPair[0]

            # Must leave at least 1 troop behind
            for i in range(countryMap[fromCountry][1]):
                listOfActions.append((self.gameActions.moveTroops, i))
            return listOfActions
        
        elif gameState == self.gameStates.end:
            return ['Celebrate!']

        else:
            raise ValueError("Impossible Game State")

    def succAndProbReward(self, state, action):
        gameState, playerNumber, countryMap, additionalParameter = state
        actionType = action[0]
        results = []
        noGameReward = [0] * self.numberOfPlayers
        attackReward = 0
        winRewardFactor = 100
        if gameState == self.gameStates.end:
            return results
               
        elif actionType == self.gameActions.placeTroops:
            countryNum = action[1]
            numberOfTroops = action[2]
            
            # In setup state
            if gameState == self.gameStates.setup:
                if countryNum not in countryMap:
                    countryMap[countryNum] = (playerNumber, numberOfTroops)
                else:
                    countryMap[countryNum] = (playerNumber, countryMap[countryNum][1] + numberOfTroops)
                nextPlayer = self.getNextPlayer(playerNumber)
                # One round has ended!
                if nextPlayer == 0:
                    # More troops to place
                    if additionalParameter > numberOfTroops:
                        results.append(((self.gameStates.setup, nextPlayer, countryMap, additionalParameter-numberOfTroops), 1, noGameReward))
                    # Let the game begin
                    else:
                        nextState = (self.gameStates.place, nextPlayer, countryMap, 0)
                        troopCount = self.getTroopCount(nextState)
                        results.append(((self.gameStates.place, nextPlayer, countryMap, troopCount), 1, noGameReward))
                # The round is continuing
                else:
                    results.append(((self.gameStates.setup, nextPlayer, countryMap, additionalParameter), 1, noGameReward))
            
            # In place state
            elif gameState == self.gameStates.place:
                countryMap[countryNum] = (playerNumber, countryMap[countryNum][1] + numberOfTroops)
                # More troops to place later
                if additionalParameter > numberOfTroops:
                    results.append(((self.gameStates.place, playerNumber, countryMap, additionalParameter-numberOfTroops), 1, noGameReward))
                # Move on to next game state
                else:
                    results.append(((self.gameStates.attack, playerNumber, countryMap, None), 1, noGameReward))
            else:
                raise ValueError("Impossible Game State While Placing Troops")
        
        elif actionType == self.gameActions.attackCountry:
            attackingCountry = action[1]
            defendingCountry = action[2]
            assert countryMap[attackingCountry][0] == playerNumber
            assert countryMap[defendingCountry][0] != playerNumber
            defendingPlayer = countryMap[defendingCountry][0]
            countryMapProbability, attack_success = self.simulateAttack(attackingCountry, defendingCountry, state)
            for oneResult in countryMapProbability:
                countryMap, probability = oneResult
                newState = (self.gameStates.attack, playerNumber, countryMap, None)
                winner = self.getWinner(newState)
                if winner == -1: # No winner
                    if attack_success:
                        attackSuccessReward = []
                        for i in range(self.numberOfPlayers):
                            if i == playerNumber:
                                attackSuccessReward.append(attackReward)
                            elif i == defendingPlayer:
                                attackSuccessReward.append(-0.9*attackReward) #not as heavy a penalty for losing a state
                            else:
                                attackSuccessReward.append(0)
                        results.append((newState, probability, attackSuccessReward))
                    else:
                        results.append((newState, probability, noGameReward))
                else:
                    assert(winner == playerNumber)
                    endGameReward = []
                    for i in range(self.numberOfPlayers):
                        if i != winner:
                            endGameReward.append(-1*winRewardFactor)
                        else:
                            endGameReward.append((self.numberOfPlayers-1)*winRewardFactor)
                    newState = (self.gameStates.end, playerNumber, countryMap, None)
                    results.append((newState, probability, endGameReward))
        
        elif actionType == self.gameActions.moveTroops:
            countryPairs = additionalParameter[0]
            countryMove = additionalParameter[1]
            countryPair = countryPairs[countryMove]
            fromCountry = countryPair[0]
            toCountry = countryPair[1]
            troopsMoved = action[1]
            countryMap[fromCountry] = (countryMap[fromCountry][0], countryMap[fromCountry][1] - troopsMoved)
            countryMap[toCountry] = (countryMap[toCountry][0], countryMap[toCountry][1] + troopsMoved)

            # No More Chance To Move Troops
            if countryMove >= len(countryPairs) - 1:
                nextPlayer = self.getNextPlayer(playerNumber)
                nextState = (self.gameStates.place, nextPlayer, countryMap, None)
                troopCount = self.getTroopCount(nextState)
                results.append(((self.gameStates.place, nextPlayer, countryMap, troopCount), 1, noGameReward))
            else:
                results.append(((self.gameStates.fortify, playerNumber, countryMap, [countryPairs, countryMove+1]), 1, noGameReward))

        elif actionType == self.gameActions.endTurn:
            if gameState == self.gameStates.attack:
                possibleMoveTroops = self.getPossibleMoveTroops(state)
                if len(possibleMoveTroops) > 0:
                    results.append(((self.gameStates.fortify, playerNumber, countryMap, [possibleMoveTroops, 0]), 1, noGameReward))
                else:
                    nextPlayer = self.getNextPlayer(playerNumber)
                    nextState = (self.gameStates.place, nextPlayer, countryMap, None)
                    troopCount = self.getTroopCount(nextState)
                    results.append(((self.gameStates.place, nextPlayer, countryMap, troopCount), 1, noGameReward))
            else:
                raise ValueError("Impossible Game State While Ending Turn")
        
        else:
            raise ValueError("Impossible Action Type: %s" %action)
        
        return results

    def discount(self):
        return 1

    def drawState(self, state, show=True, showTime=0.05):
        countryMap = state[2]
        self.worldMap.drawState(countryMap, show, showTime)

    def featureExtractor(self, state, action, playerNum):
        gameState, playerNumber, countryMap, additionalParameter = state
        features = []
        for country, countryState in countryMap.iteritems():
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
        gameState, playerNumber, countryMap, additionalParameter = state
        features = []
        my_country_states = []
        opp_country_states = []
        ownership_feature = []
        my_troop_count_list = []
        num_my_troops = 0
        num_opp_troops = 0
        for country, countryState in countryMap.iteritems():
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
        total_continent_count = self.worldMap.numberOfContinents()
        for continent in range(total_continent_count):
            if self.ownsContinent(playerNum, countryMap, continent):
                my_continent_bonus += self.worldMap.getContinentReward(continent)
            for opp in range(self.numberOfPlayers):
                if opp != playerNum:
                    if self.ownsContinent(opp, countryMap, continent):
                        opp_continent_bonus += self.worldMap.getContinentReward(continent)

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
        if playerNum==playerNumber:
            turn_indicator = 1
        features.append(('turn', turn_indicator))
        features.append((('action_type', action[0]), turn_indicator))

        return features
