import copy


class RiskActions:
    def __init__(self):
        self.tradeCards = 'TRADE_CARDS'
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
            ownContinent = True
            countries = self.worldMap.getContinentCountries(continent)
            for country in countries:
                if countryMap[country][0] != playerNumber:
                    ownContinent = False
                    break
            if ownContinent:
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
            # Attackers Lost!
            elif attackersLeft == 1:
                newCountryMap[defendingCountry] = (countryMap[defendingCountry][0], defendersLeft)
                newCountryMap[attackingCountry] = (countryMap[attackingCountry][0], attackersLeft)
                countryMapToProbabilityMapping.append((newCountryMap, probability))
            else:
                raise ValueError("Either Attacker Loses or Defender Loses!")
        return countryMapToProbabilityMapping

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
            additionalParameter = 40
        elif self.numberOfPlayers == 3:
            additionalParameter = 35
        elif self.numberOfPlayers == 4:
            additionalParameter = 30
        elif self.numberOfPlayers == 5:
            additionalParameter = 25
        elif self.numberOfPlayers == 6:
            additionalParameter = 20
        else:
            raise ValueError("2-6 Players Only")
        return (self.gameStates.setup, firstPlayer, countryMap, additionalParameter)

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
        if gameState == self.gameStates.end:
            return results
        elif actionType == self.gameActions.tradeCards:
            # Disallow trading of cards for now
            results.append(((self.gameStates.place, playerNumber, countryMap, additionalParameter), 1, noGameReward))
        
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
            countryMapProbability = self.simulateAttack(attackingCountry, defendingCountry, state)
            for oneResult in countryMapProbability:
                countryMap, probability = oneResult
                newState = (self.gameStates.attack, playerNumber, countryMap, None)
                winner = self.getWinner(newState)
                if winner == -1: # No winner
                    results.append((newState, probability, noGameReward))
                else:
                    assert(winner == playerNumber)
                    endGameReward = []
                    for i in range(self.numberOfPlayers):
                        if i != winner:
                            endGameReward.append(-1)
                        else:
                            endGameReward.append(self.numberOfPlayers-1)
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
                features.append(countryTroops/5)
            else:
                features.append(-countryTroops/5)
        featureKey = (tuple(features), action)
        featureValue = 1
        return [(featureKey, featureValue)]