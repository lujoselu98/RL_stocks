import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random

#to import the stocks data
import pickle
with open("./mystockss.pkl", "rb") as f:
    d = pickle.load(f)

#here the action sequence and profit of every episodes are stored    
action_f = open('./numpy.txt', 'a')
profit_f = open('./profit.txt', 'a')

#oopening and closing values of stock passed.
apl_open = d["dewa_open"]
apl_close = d["dewa_close"]


class StocksEnvAAPL(gym.Env):
    
#initialisation of state variables and action space
    def __init__(self):
       
        self.low_state = np.zeros((15,))
        self.high_state = np.zeros((15,))+1000000
        self.viewer = None
        self.action_space = spaces.Discrete(3)    
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)        
        
        self.state = np.zeros(15)
        

        self.series_length = 150

        
        self.max_stride = 5
        self.stride = self.max_stride
        
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

    # this the the step function which is used to take the action on the environment
    def step(self, action):
        action_f.write((str(action) + ',')) #To write the action taken to file
        profit_sell = 0
        #print("\n previous state", " - " ,self.state[5]," - ",self.state[0], " - ",self.state[1], " - ",self.state[2])
        action = [action,1.]
        #print("\n previous state", " pf- " ,self.portfolio_value()," - ",self.state[0], " - ",self.state[1]," - ",self.state[2])
        cur_timestep = self.cur_timestep
        cur_value = self.portfolio_value()
        gain = cur_value - self.starting_portfolio_value
        gain_avg = (apl_open[cur_timestep] - self.state[13]) * self.state[0]
        
           
        #to check if the timesteps exceed the data so as to end the episode        
        if cur_timestep >= self.starting_point + (self.series_length * self.stride):
            new_state = [self.state[0], self.state[1], self.next_opening_price(), \
                         *self.ten_day_window(),self.state[4],self.next_open_price(self.state[0])]
            self.state = new_state

            total_prof = sum(self.ps)
            #print("\n ", gain_avg ," - ",total_prof," - ",self.buycount , " - " ,self.sellcount, "-" ,self.nothing,"- ",self.nothingpseudo) 
            profit_f.write(str(total_prof) + '\n') #writes the total profit of episode to file
            return np.array(new_state), gain_avg , True, { "msg": "done"}
        
        
  #To sell
        if action[0] == 1:
            # to check if stocks are available to sell
            if action[1] > self.state[0]:
                self.nothingpseudo+=1
                new_state = [self.state[0], self.state[1] ,self.next_opening_price(), \
                     *self.ten_day_window(),self.state[13],self.next_open_price(self.state[0])]
                self.state = new_state
                
                retval = np.array(new_state), -100000 , False, { "msg": "nothing" }

            else:
                self.sellcount += 1
                apl_shares = self.state[0] - action[1]
                cash_gained = action[1] * apl_open[cur_timestep] * 0.9
                new_state = [apl_shares , self.state[1] + cash_gained, self.next_opening_price(), \
                       *self.ten_day_window(),self.state[13],self.next_open_price(apl_shares)]
                
                self.state = new_state
                profit_sell = apl_open[cur_timestep] - self.state[13]
                self.ps.append(profit_sell)
                cur_value = self.portfolio_value()
                gain = cur_value - self.starting_portfolio_value
                
                retval = np.array(new_state),  gain_avg + (profit_sell * 100) , False, { "msg": "sold AAPL"}
        
        
   # To do nothing/sit     
        if action[0] == 2:
            self.nothing += 1
            new_state = [self.state[0], self.state[1] ,self.next_opening_price(), \
                     *self.ten_day_window(),self.state[13],self.next_open_price(self.state[0])]
            self.state = new_state
            self.reward += gain_avg
            retval = np.array(new_state), gain_avg , False, { "msg": "nothing" }
   
   # To buy
        if action[0] == 0:
            # To check if cash is available to buy the stock
            if action[1] * apl_open[cur_timestep] > self.state[1]:
                new_state = [self.state[0], self.state[1], self.next_opening_price(), \
                         *self.ten_day_window(),self.state[13],self.next_open_price(self.state[0])]
                self.state = new_state
                self.nothingpseudo+=1
               # print("\nEpisode Terminating Bankrupt REWARD = " ,self.reward," - " ,self.buycount , " - " ,self.sellcount, "-" ,self.nothing ,"- ",self.nothingpseudo)
                
                retval = np.array(new_state),  -100000 ,False, { "msg": "bankrupted self"}
                
            else:
                self.buycount+=1
                apl_shares = self.state[0] + action[1]
                cash_spent = action[1] * apl_open[cur_timestep] * 1.1
                new_state = [apl_shares, self.state[1] - cash_spent, self.next_opening_price(), \
                        *self.ten_day_window(),self.calcAvg(self.state[13],apl_open[cur_timestep]),self.next_open_price(apl_shares)]
                self.state = new_state
                cur_value = self.portfolio_value()
                gain = cur_value - self.starting_portfolio_value
                
                retval = np.array(new_state), gain_avg, False, { "msg": "bought AAPL"}
                
                    
        #print("\n action taken: ",action, " pf- " ,self.portfolio_value()," - ",self.state[0],  " - ",self.state[1])
        self.cur_timestep += self.stride

        return retval
    
    
    #This is called before beginning of every new episode
    def reset(self):
        self.state = np.zeros(15)
        self.starting_cash = 1000
        self.cur_timestep = 10
        self.starting_point = self.cur_timestep
        self.state[0] = 10 
        self.state[1] = self.starting_cash
        self.state[2] = apl_open[self.cur_timestep]
        self.starting_portfolio_value = self.portfolio_value_states()
        self.state[3:13] = self.ten_day_window()
        self.state[13] = apl_open[self.cur_timestep]
        self.state[14] = self.starting_portfolio_value

        # These are used to keep a count of the actions
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

    #To calculate portfolio value w.r.t closing price
    def portfolio_value(self):
        return (self.state[0] * apl_close[self.cur_timestep])  + self.state[1]
    
    #To calculate portfolio value w.r.t opening price
    def portfolio_value_states(self):
        return (self.state[0] * apl_open[self.cur_timestep])  + self.state[1]
    
    #To return the next opening price    
    def next_opening_price(self):
        step = self.cur_timestep + self.stride
        return apl_open[step]
    
    #To return the latest investment in stocks    
    def next_open_price(self,apl_):
        step = self.cur_timestep + self.stride
        return (apl_ * apl_open[step])

    #To take the last 10 days opening value of the stock from data with respect to current time step
    def ten_day_window(self):
        step = self.cur_timestep 
        return apl_open[step-10:step]
    
    #To calculate the average buying price of the stocks
    def calcAvg(self,prev,new):
        return ((prev*self.state[0])+new)/(self.state[0]+1)
    
    
    def render(self, mode='human'):
        print("Render called")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
            
