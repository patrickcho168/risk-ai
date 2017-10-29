import networkx as nx
import csv
import matplotlib.pyplot as plt
import numpy as np

class WorldMap:
    def numberOfCountries(self): raise NotImplementedError("Override me")
    def numberOfContinents(self): raise NotImplementedError("Override me")
    def numberOfBorders(self): raise NotImplementedError("Override me")
    def isBordered(self, country1, country2): raise NotImplementedError("Override me")
    def getContinent(self, country): raise NotImplementedError("Override me")
    def getContinentReward(self, continent): raise NotImplementedError("Override me")
    def drawMap(self): raise NotImplementedError("Override me")

class ClassicWorldMap(WorldMap):
    def __init__(self):
        self.worldMap = nx.Graph()
        self.countryToContinentMapping = {}
        self.continentToScoreMapping = {}
        self.pos = {}

        CLASSIC_WORLD_MAP_CSV = "classicWorldMap.csv"
        edgeCheck = {}
        with open(CLASSIC_WORLD_MAP_CSV, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                countryNum = int(row[0])
                continentNum = int(row[1])
                continentScore = int(row[2])
                self.countryToContinentMapping[countryNum] = continentNum
                self.continentToScoreMapping[continentNum] = continentScore
                for idx in range(3, len(row)):
                    self.worldMap.add_edge(countryNum, int(row[idx]))
                    countryA = min(countryNum, int(row[idx]))
                    countryB = max(countryNum, int(row[idx]))
                    if (countryA, countryB) in edgeCheck:
                        del edgeCheck[(countryA, countryB)]
                    else:
                        edgeCheck[(countryA, countryB)] = 1

        # Get Coordinates Of Node
        minX = 0
        maxX = 0
        minY = 0
        maxY = 0
        CLASSIC_WORLD_MAP_COORDINATES_CSV = "classicWorldMapCoordinates.csv"
        with open(CLASSIC_WORLD_MAP_COORDINATES_CSV, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                X = float(row[1])
                Y = float(row[2])
                self.pos[int(row[0])] = np.array([X, Y])
                minX = min(minX, X)
                maxX = max(maxX, X)
                minY = min(minY, Y)
                maxY = max(maxY, Y)

        for node, coordinates in self.pos.iteritems():
            self.pos[node][0] = (coordinates[0] - minX) * 2 * (maxX-minX) - 1
            self.pos[node][1] = 1 - (coordinates[1] - minY) * 2 * (maxY-minY)

        assert(len(edgeCheck) == 0)


    def numberOfCountries(self):
        return self.worldMap.number_of_nodes()

    def numberOfContinents(self):
        return len(self.continentToScoreMapping)

    def numberOfBorders(self):
        return self.worldMap.number_of_edges()

    def isBordered(self, country1, country2):
        return self.worldMap.has_edge(country1, country2)

    def getContinent(self, country):
        return self.countryToContinentMapping[country]

    def getContinentReward(self, continent):
        return self.continentToScoreMapping[continent]

    def drawMap(self, show=True, showTime=0.05):
        numberOfContinents = self.numberOfContinents()
        values = [self.countryToContinentMapping[node] * 1.0 / numberOfContinents for node in self.worldMap.nodes()]
        nx.draw(self.worldMap, self.pos, cmap=plt.get_cmap('jet'), node_color=values)
        if show:
            plt.show()
        else:
            plt.pause(showTime)

class RiskMDP:
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

if __name__ == "__main__":
    worldMap = ClassicWorldMap()
    worldMap.drawMap(show=False)


