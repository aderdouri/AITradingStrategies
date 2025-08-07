import gym 
import numpy as np
import pandas as pd
import re, os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import random
from itertools import product

row = 3
col = 3
out = row*col

class tictactoe():
    def __init__(self,p="1",r="goal"):
        self.board = np.array(["**"]*out).reshape(row,col)
        self.r = r
        self.end = False
        self.move = None
        self.wins = None
        if p == "1":
            self.p1 = "X"
            self.id = 1
            self.p2 = "O"
        else:
            self.p1="O"
            self.id = 2
            self.p2 = "X"
        self.state = {'**':0,'X':1,'O':2} 
        self.boardpos = {'**':2,'X':1,'O':0} 
        
        pos = self.positions()
        self.transform1d = {position:i for (i,position) in enumerate(pos)}
        self.transform2d = {i:position for (i,position) in enumerate(pos)}
        
        self.initial = self.boardtostate()
        board_out = [list(range(row)) for _ in range(out)]
        states = set(product(*board_out))
        
        for s in states:
            if s.count(0)%2 == 1 and s.count(1)== s.count(2):
                p1state = s
            if s.count(0)%2 == 0 and s.count(1) == s.count(2)+1:
                p2state = s
                
        if p == "1":
            self.s1 = p1state
        else:
            self.s1 = p2state

        
    def reset(self):
        self.board = np.array(["**"]*out).reshape(row,col)
        self.end = False
        self.move = None
        self.wins = None
        self.initial = self.boardtostate()
    
    def display(self):
        print(pd.DataFrame(self.board))
        
        
    def boardtostate(self):
        return tuple([self.state[x] for x in np.ravel(self.board)])
        
    @staticmethod
    def actions(state):
        return [i for i,x  in enumerate(state) if x ==0]
        
    def endgame(self):
        if not np.any(self.board=='**') :
            self.end = True  
        return self.end
    

    def positions(self):
        x,y = np.where(self.board=='**')
        a=[]
        for x,y in zip(x,y):
            a.append((x,y))
        return a
    
    def winning_shot(self,p):
        if np.all(self.board[0,:] == p):
            self.wins = "r1"  #row
        elif np.all(self.board[1,:] == p): 
            self.wins = "r2" #row
        elif np.all(self.board[2,:] == p):
            self.wins = "r3"
        elif np.all(self.board[:,0] == p):
            self.wins =  "c1"
        elif np.all(self.board[:,1] == p):
            self.wins = "c2"
        elif np.all(self.board[:,2] == p):
            self.wins = "c3"
        elif np.all(self.board.diagonal()== p):
            self.wins = "d1"
        elif  np.all(np.fliplr(self.board).diagonal()== p):
            self.wins = "d2"
        else:
            return False
        return True
    
    def p1_turn(self,position):
        assert position[0]>=0 and position[0]<=2 and position[1]>=0 and position[1]<=2 , "Row, Column can be [0, 1, 2] only"
        assert self.board[position] == "**" , f"Cell is already filled with {self.board[position]}"
        assert np.any(self.board == '**') , "The board is full"
        assert self.winning_shot(self.p1) == False and self.winning_shot(self.p2)== False , "End the game now"
        self.board[position] = self.p1
       
        
        p1_wins = self.winning_shot(self.p1)
        print(p1_wins)
        p2_wins = self.winning_shot(self.p2)
        
        
        if self.r == "goal":
            if p1_wins:
                self.end = True
                return 1
            elif p2_wins:
                self.end = True
                return -1
            else:
                return 0
            
        elif self.r == "penalty":
            if p1_wins:
                self.end = True
                return 0
            elif p2_wins:
                self.end = True
                return -10
            else:
                return -1
            
    def p2_turn(self,position):
        
        assert position[0]>=0 and position[0]<=2 and position[1]>=0 and position[1]<=2 , "choose wisely"
        assert self.board[position] == "**" , "move next"
        assert np.any(self.board == '**') , "game is full"
        assert self.winning_shot(self.p1) == False and self.winning_shot(self.p2)== False , "End the game now"
        self.board[position] = self.p2
        
        
    # asks user for input cell location to fill
    # useful for manual testing
    def ask_user_action(self, ):
        r, c = input('Make your move. Enter Row<space>Column:').split(' ')
        r = int(r) # row
        c = int(c) # col
        self.p1_turn((r, c))
        
    def take_action_agent(self, mode='random'):
        """
        make a move on board using an agent
        we assume agent is 'o'
        """
        if mode == 'random':
            self.take_random_action()
        else:
            # TODO implement making a move using rl_algo function
            pass
        
    def take_random_action(self, ):
        """
        randomly selects an empty position on board and makes a move
        for simplicity we assume random agent is 'o'
        """
        possible_locations = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == '**':
                    possible_locations.append((i, j))
        n = len(possible_locations)
        i = random.randrange(n)
        pos = possible_locations[i]
        print(pos)
        self.p2_turn(pos)
        