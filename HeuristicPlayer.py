import random
import math
from collections import defaultdict
from actions import *

class HeuristicPlayer():
    def __init__(self, board, mdp):
        self.board = board
        self.mdp = mdp
    
    def getAction(self, state):
        
        easiest_cont, volatile_cont = self.getTargetContinent(state)
        
        #valid_actions = self.actions(state)
        if state.is_setup():
            pass #TODO
        elif state.is_place():
            #fortify attacking of easiest continent
            target_cont_country = self.getContWeakestBorder(state, easiest_cont)
            if target_cont_country <= -1:
                #reinforce weakness in own continent or boost footholds in others
                target_cont_country = self.getContWeakestBorder(state, volatile_cont)
            return Place_Action(target_cont_country, 1)
        elif state.is_attack():
            #attack from strongest to weakest, starting with trying to attack in the easiest continent
            if easiest_cont < -1:
                easiest_cont = None
            attack_from = self.getContStrongestOwnCountry(state, easiest_cont)
            if attack_from < 0:
                easiest_cont = None
                attack_from = self.getContStrongestOwnCountry(state, easiest_cont)
            if attack_from < 0:
                return End_Action() #no possible moves e.g. all countries 1 troop only
            attack_to = self.getCountryWeakestContOpp(state, attack_from, easiest_cont)
            return Attack_Action(attack_from, attack_to)
        elif state.is_fortify():
            fort_from, fort_troops = self.getSafeCountry(state)
            if fort_from is not None:
                #same logic as for choosing place action
                fort_to = self.getContWeakestBorder(state, easiest_cont)
                if fort_to <= -1:
                    fort_to = self.getContWeakestBorder(state, volatile_cont)
                #print fort_from, fort_to, fort_troops, state.country_mapping
                return Fortify_Action(fort_from, fort_to, fort_troops - 1)
            return End_Action()
        elif state.is_end():
            return ['Celebrate!']
        else:
            raise ValueError("Impossible Game State")
    
    # Find the most vulnerable continents for this player, which means both which is the easiest to attack (based on having a good foothold) or is in the most need of reinforcements (weak foothold or own a continent but low troops)
    def getTargetContinent(self, state):
        easiest_cont = -1
        easiest_cont_ratio = -1
        volatile_cont = -1
        volatile_cont_diff = float("inf")
        for continent in range(self.board.numberOfContinents()):
            foothold = False
            self_troops = 0
            opp_troops = 0
            for country in self.board.getContinentCountries(continent):
                owner, troops = state.country_mapping[country]
                if owner == state.curr_player:
                    self_troops += troops
                    foothold = True
                else:
                    opp_troops += troops
            if foothold:
                if opp_troops > 0:
                    cont_ratio = float(self_troops) / opp_troops
                    if cont_ratio > easiest_cont_ratio:
                        easiest_cont = continent
                        easiest_cont_ratio = cont_ratio
                    easiest_cont = continent
                cont_diff = abs(self_troops - opp_troops)
                if cont_diff < volatile_cont_diff:
                    volatile_cont = continent
                    volatile_cont_diff = cont_diff
            else: #handle cases where cocntinent lines are clearly split
                if easiest_cont < 0:
                    easiest_cont = continent
                if easiest_cont < 0:
                    volatile_cont = continent
        return (easiest_cont, volatile_cont)
    
    # For a continent where the player has a foothold or owns the whole continent, find the weakest border if any, otherwise return -1 (no foothold or whole continent is enclosed)
    def getContWeakestBorder(self, state, continent):
        weakest = -1
        weakest_ratio = float("inf")
        for country in self.board.getContinentCountries(continent):
            if state.curr_player == state.country_mapping[country][0]:
                opp_troops = self.getBorderOppTroops(state, country)
                if opp_troops > 0:
                    troop_ratio = float(state.country_mapping[country][1]) / opp_troops
                    if troop_ratio < weakest_ratio:
                        weakest = country
                        weakest_ratio = troop_ratio
        return weakest
    
    # Find the current player's strongest country which has a neighbouring enemy. May optionally be limited to within a particular continent
    def getContStrongestOwnCountry(self, state, target_continent=None, min_troops=2):
        strongest_country = -1
        strongest_country_troops = min_troops-1
        if target_continent is None:
            target_continent = range(self.board.numberOfContinents())
        else:
            target_continent = [target_continent]
        for continent in target_continent:
            for country in self.board.getContinentCountries(continent):
                if state.curr_player == state.country_mapping[country][0] and self.getBorderOppTroops(state, country) > 0 and state.country_mapping[country][1] > strongest_country_troops:
                    strongest_country = country
                    strongest_country_troops = state.country_mapping[country][1]
        return strongest_country
        
    # For a country, find its weakest opponent. Optionally limited to a target continent
    def getCountryWeakestContOpp(self, state, country, target_continent=None):
        if target_continent is not None:
            continentCountries = self.board.getContinentCountries(target_continent)
        weakest_opp = -1
        weakest_troops = float("inf")
        for border in self.board.bordered(country):
            if state.country_mapping[border][0] != state.country_mapping[country][0]:
                if target_continent is None or border in continentCountries:
                    if state.country_mapping[border][1] < weakest_troops:
                        weakest_troops = state.country_mapping[border][1]
                        weakest_opp = border
        return weakest_opp
    
    def getBorderOppTroops(self, state, country):
        opp_troops = 0
        for border in self.board.bordered(country):
            if state.country_mapping[country][0] != state.country_mapping[border][0]:
                opp_troops += state.country_mapping[border][1]
        return opp_troops
    
    # Get first safe (no enemy neightbours) country and troop count with >1 troops
    def getSafeCountry(self, state):
        for country in state.country_mapping:
            if state.country_mapping[country][0] == state.curr_player and self.getBorderOppTroops(state, country) == 0 and state.country_mapping[country][1] > 1:
                return (country, state.country_mapping[country][1])
        return (None, None)
