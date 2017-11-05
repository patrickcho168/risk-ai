import csv
import networkx as nx
from collections import defaultdict
import numpy as np

class WorldMap:
    def numberOfCountries(self): raise NotImplementedError("Override me")
    def numberOfContinents(self): raise NotImplementedError("Override me")
    def numberOfBorders(self): raise NotImplementedError("Override me")
    def isBordered(self, country1, country2): raise NotImplementedError("Override me")
    def bordered(self, country): raise NotImplementedError("Override me")
    def getContinent(self, country): raise NotImplementedError("Override me")
    def getContinentReward(self, continent): raise NotImplementedError("Override me")
    def getContinentCountries(self, continent): raise NotImplementedError("Override me")
    def drawMap(self): raise NotImplementedError("Override me")

class ClassicWorldMap(WorldMap):
    def __init__(self, worldMapFile, worldMapCoordinatesFile):
        self.worldMap = nx.Graph()
        self.countryToContinentMapping = {}
        self.continentToCountriesMapping = defaultdict(list)
        self.continentToScoreMapping = {}
        self.pos = {}
        edgeCheck = set()

        with open(worldMapFile, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                countryNum = int(row[0])
                continentNum = int(row[1])
                continentScore = int(row[2])
                self.countryToContinentMapping[countryNum] = continentNum
                self.continentToScoreMapping[continentNum] = continentScore
                self.continentToCountriesMapping[continentNum].append(countryNum)
                for idx in range(3, len(row)):
                    self.worldMap.add_edge(countryNum, int(row[idx]))
                    countryA = min(countryNum, int(row[idx]))
                    countryB = max(countryNum, int(row[idx]))
                    if (countryA, countryB) in edgeCheck:
                        edgeCheck.remove((countryA, countryB))
                    else:
                        edgeCheck.add((countryA, countryB))
        assert(len(edgeCheck) == 0)

        # Get Coordinates Of Node
        minX = 0
        maxX = 0
        minY = 0
        maxY = 0
        with open(worldMapCoordinatesFile, 'rb') as f:
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


    def numberOfCountries(self):
        return self.worldMap.number_of_nodes()

    def numberOfContinents(self):
        return len(self.continentToScoreMapping)

    def numberOfBorders(self):
        return self.worldMap.number_of_edges()

    def isBordered(self, country1, country2):
        return self.worldMap.has_edge(country1, country2)

    def bordered(self, country):
        return nx.all_neighbors(self.worldMap, country)

    def getContinent(self, country):
        return self.countryToContinentMapping[country]

    def getContinentReward(self, continent):
        return self.continentToScoreMapping[continent]

    def getContinentCountries(self, continent):
        return self.continentToCountriesMapping[continent]

    def drawMap(self, show=True, showTime=0.05):
        numberOfContinents = self.numberOfContinents()
        values = [self.countryToContinentMapping[node] * 1.0 / numberOfContinents for node in self.worldMap.nodes()]
        nx.draw(self.worldMap, self.pos, cmap=plt.get_cmap('jet'), node_color=values)
        if show:
            plt.show()
        else:
            plt.pause(showTime)

    def drawState(self, countryMap, show=True, showTime=0.05):
        numberOfContinents = self.numberOfContinents()
        values = [self.countryToContinentMapping[node] * 1.0 / numberOfContinents for node in self.worldMap.nodes()]
        nx.draw_networkx(self.worldMap, self.pos, cmap=plt.get_cmap('jet'), node_color=values, node_size=500, alpha=0.5, hold=False, with_labels=False)
        # Maximum 6 players
        colors = ['k', 'r', 'b', 'g', 'y', 'c']
        for idx in range(len(colors)):
            labels = {}
            for country, countryState in countryMap.iteritems():
                countryPlayer = countryState[0]
                countryTroops = countryState[1]
                if countryPlayer == idx:
                    labels[country] = countryTroops
            nx.draw_networkx_labels(self.worldMap, self.pos, labels=labels, font_size=12, font_color=colors[idx])
        if show:
            plt.show()
        else:
            plt.pause(showTime)
            plt.clf()
            plt.close()