import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random

import matplotlib.pyplot as plt

import pickle
with open("./mystockstest.pkl", "rb") as f:
    d = pickle.load(f)
    
action_f = open('./numpy.txt', 'a')
profit_f = open('./profit.txt', 'a')


apl_open = d["ao"]
apl_close = d["ac"]
#msf_open = d["mo"]
#msf_close = d["mc"]


class StocksEnvAAPL(gym.Env):
    

    def __init__(self):
       
        self.low_state = np.zeros((15,))
        self.high_state = np.zeros((15,))+1000000

        self.viewer = None

       
        self.action_space = spaces.Discrete(3)    
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)        
        
   
        
        self.state = np.zeros(15)
        
        self.starting_cash = 2000
        self.buycount=0
        self.sellcount=0
        self.nothing=0
        self.nothingpseudo=0

        self.series_length = 100

        
        self.max_stride = 1
        self.stride = 1 # no longer varying it
        
        self.done = False
        self.reward = 0
        self.diversification_bonus = 100
        self.inaction_penalty = 0
        self.ps = []
        self.g_t = []
        self.action_set = []
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action_f.write((str(action) + ','))
        profit_sell = 0
        #print("\n previous state", " - " ,self.state[5]," - ",self.state[0], " - ",self.state[1], " - ",self.state[2])
        action = [action,1.]
        #print("\n previous state", " pf- " ,self.portfolio_value()," - ",self.state[0], " - ",self.state[1]," - ",self.state[2])
        cur_timestep = self.cur_timestep
        ts_left = 0 # self.series_length * self.stride - (cur_timestep - self.starting_point)
        retval = None
        cur_value = self.portfolio_value()
        gain = cur_value - self.starting_portfolio_value
        gain_avg = (apl_open[cur_timestep] - self.state[13]) * self.state[0]
        
           
        #print(self.starting_point + (100 * 1))
        if cur_timestep >= (self.starting_point + (160 * 1)):
            new_state = [self.state[0], self.state[1], self.next_opening_price(), \
                         *self.five_day_window(),self.state[4],self.next_open_price(self.state[0])]
            self.state = new_state
            bonus = 0.
            if self.state[0] > 0 :
                bonus = self.diversification_bonus 
            #self.g_t.append(self.reward)    
            #self.reward +=gain_avg
            total_prof = sum(self.ps)
            #print("\n ", gain_avg ," - ",total_prof," - ",self.buycount , " - " ,self.sellcount, "-" ,self.nothing,"- ",self.nothingpseudo) 
            profit_f.write(str(total_prof) + '\n')
            return np.array(new_state), gain_avg , True, { "msg": "done"}
        
        
        
        if action[0] == 1:
            if action[1] > self.state[0]:
                self.nothingpseudo+=1
                new_state = [self.state[0], self.state[1] ,self.next_opening_price(), \
                     *self.five_day_window(),self.state[13],self.next_open_price(self.state[0])]
                self.state = new_state
                self.reward += -100000
                retval = np.array(new_state), -100000 , False, { "msg": "nothing" }

            else:
                self.sellcount += 1
                apl_shares = self.state[0] - action[1]
                cash_gained = action[1] * apl_open[cur_timestep] * 0.9
                new_state = [apl_shares , self.state[1] + cash_gained, self.next_opening_price(), \
                       *self.five_day_window(),self.state[13],self.next_open_price(apl_shares)]
                
                self.state = new_state
                profit_sell = apl_open[cur_timestep] - self.state[13]
                self.ps.append(profit_sell)
                cur_value = self.portfolio_value()
                gain = cur_value - self.starting_portfolio_value
                self.reward += gain_avg
                retval = np.array(new_state),  gain_avg + (profit_sell * 100) , False, { "msg": "sold AAPL"}
        
        
        
        if action[0] == 2:
            self.nothing += 1
            new_state = [self.state[0], self.state[1] ,self.next_opening_price(), \
                     *self.five_day_window(),self.state[13],self.next_open_price(self.state[0])]
            self.state = new_state
            self.reward += gain_avg
            retval = np.array(new_state), gain_avg , False, { "msg": "nothing" }
        
        if action[0] == 0:
            
            if action[1] * apl_open[cur_timestep] > self.state[1]:
                new_state = [self.state[0], self.state[1], self.next_opening_price(), \
                         *self.five_day_window(),self.state[13],self.next_open_price(self.state[0])]
                self.state = new_state
                #self.reward += -100000
                self.nothingpseudo+=1
               # print("\nEpisode Terminating Bankrupt REWARD = " ,self.reward," - " ,self.buycount , " - " ,self.sellcount, "-" ,self.nothing ,"- ",self.nothingpseudo)
                
                retval = np.array(new_state),  -100000 ,False, { "msg": "bankrupted self"}
                
            else:
                self.buycount+=1
                apl_shares = self.state[0] + action[1]
                cash_spent = action[1] * apl_open[cur_timestep] * 1.1
                new_state = [apl_shares, self.state[1] - cash_spent, self.next_opening_price(), \
                        *self.five_day_window(),self.calcAvg(self.state[13],apl_open[cur_timestep]),self.next_open_price(apl_shares)]
                self.state = new_state
                cur_value = self.portfolio_value()
                gain = cur_value - self.starting_portfolio_value
                #self.reward += gain_avg
                retval = np.array(new_state), gain_avg, False, { "msg": "bought AAPL"}
                
        

                
        #print("\n action taken: ",action, " pf- " ,self.portfolio_value()," - ",self.state[0],  " - ",self.state[1])
        self.cur_timestep += 1 #self.stride

        return retval

    def reset(self):
        self.state = np.zeros(15)
        self.starting_cash = 2500
        self.cur_timestep = 10
        self.starting_point = self.cur_timestep
        self.state[0] = 10 # random.randint(20,100)
        self.state[1] = 1000 #random.randint(1000,5000)
        self.state[2] = apl_open[self.cur_timestep]
        self.starting_portfolio_value = self.portfolio_value_states()
        self.state[3:13] = self.five_day_window()
        self.state[13] = apl_open[self.cur_timestep]
        self.state[14] = self.starting_portfolio_value
        self.buycount=0
        self.sellcount=0
        self.nothing=0
        self.nothingpseudo=0
        self.done = False
        self.reward = 0
        self.ps = []
        action_f.write('\n')
        self.action_set = []
        
        return self.state

    
    def portfolio_value(self):
        return (self.state[0] * apl_close[self.cur_timestep])  + self.state[1]


    def portfolio_value_states(self):
        return (self.state[0] * apl_open[self.cur_timestep])  + self.state[1]
    
    def next_opening_price(self):
        step = self.cur_timestep + self.stride
        return apl_open[step]
        
    def next_open_price(self,apl_):
        step = self.cur_timestep + self.stride
        return (apl_ * apl_open[step])

    

    def five_day_window(self):
        step = self.cur_timestep
        if step < 5:
            return apl_open[0]
        apl5 = apl_open[step-5:step].mean()
        
        return apl_open[step-10:step]
    
    def calcAvg(self,prev,new):
        return ((prev*self.state[0])+new)/(self.state[0]+1)
    
    def giveShareRew(self):
        return 0 #self.state[0]/10
    
    def netprof(self):
        return  (apl_open[self.cur_timestep] - self.state[13]) * self.state[0]
    
    def render(self, mode='human'):
        print("")
      
        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
            
