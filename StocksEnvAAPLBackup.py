import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


import matplotlib.pyplot as plt

import pickle
with open("./aplmsfopencloseOG.pkl", "rb") as f:
	d = pickle.load(f)


apl_open = d["ao"]
apl_close = d["ac"]
msf_open = d["mo"]
msf_close = d["mc"]


class StocksEnvAAPL(gym.Env):
	

	def __init__(self):
		self.starting_shares_mean = 0
		self.randomize_shares_std = 0
		self.starting_cash_mean = 200
		self.randomize_cash_std = 0
		
		#-----
		self.low_state = np.zeros((5,))
		self.high_state = np.zeros((5,))+100000000

		self.viewer = None

		#self.action_space = spaces.Box(low=0, high=4,
		 #                              shape=(1,), dtype=np.float32)
		self.action_space = spaces.Discrete(3)    
		self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
											dtype=np.float32)        
		
		#===
		
		self.state = np.zeros(5)
		
		self.starting_cash = 2000

		self.series_length = 100
		self.starting_point = 1
		self.cur_timestep = self.starting_point
		
		self.state[0] = 70
		self.starting_portfolio_value = self.portfolio_value_states()
		self.state[1] = self.starting_cash
		self.state[2] = apl_open[self.cur_timestep]
		self.state[3] = self.starting_portfolio_value
		self.state[4] = self.five_day_window()
		
		self.max_stride = 5
		self.stride = self.max_stride # no longer varying it
		
		self.done = False
		self.reward = 0
		self.diversification_bonus = 1.
		self.inaction_penalty = 0

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):

		#print("\n previous state", " - " ,self.state[5]," - ",self.state[0], " - ",self.state[1], " - ",self.state[2])
		action = [action,1.]
		print("\n previous state", " pf- " ,self.state[3]," - ",self.state[0], " - ",self.state[1]," - ",self.state[2])
		cur_timestep = self.cur_timestep
		ts_left = self.series_length* self.stride - (cur_timestep - self.starting_point)
		retval = None
		cur_value = self.portfolio_value()
		gain = cur_value - self.starting_portfolio_value
		
		if cur_timestep >= self.starting_point + (self.series_length * self.stride):
			new_state = [self.state[0], self.state[1], self.next_opening_price(), \
						cur_value, self.five_day_window()]
			self.state = new_state
			bonus = 0.
			if self.state[0] > 0 :
				bonus = self.diversification_bonus
			print("\n  REWARD = {} Episode Terminating done  -- portfoliovalue is ", self.reward ,cur_value  ) 
			self.reward += bonus + gain*1000
			return np.array(new_state),  bonus + gain*1000, True, { "msg": "done"}
		
		if action[0] == 2:
			new_state = [self.state[0], self.state[1] ,self.next_opening_price(), \
					cur_value, self.five_day_window()]
			self.state = new_state
			self.reward += -self.inaction_penalty-ts_left +gain*1000
			retval = np.array(new_state),  -self.inaction_penalty-ts_left + gain*1000, False, { "msg": "nothing" }
		
		if action[0] == 0:
			if action[1] * apl_open[cur_timestep] > self.state[1]:
				new_state = [self.state[0], self.state[1], self.next_opening_price(), \
						cur_value, self.five_day_window()]
				self.state = new_state
				self.reward += -100000
				print("\nEpisode Terminating Bankrupt REWARD = " ,self.reward)
				
				retval = np.array(new_state), -100000, True, { "msg": "bankrupted self"}
				
			else:
				apl_shares = self.state[0] + action[1]
				cash_spent = action[1] * apl_open[cur_timestep] * 1.1
				new_state = [apl_shares, self.state[1] - cash_spent, self.next_opening_price(), \
					   self.next_open_price(apl_shares) + self.state[1] - cash_spent, self.five_day_window()]
				self.state = new_state
				cur_value = self.portfolio_value()
				gain = cur_value - self.starting_portfolio_value
				self.reward += -self.inaction_penalty-ts_left +gain*1000
				retval = np.array(new_state),  -self.inaction_penalty-ts_left +gain*1000 , False, { "msg": "bought AAPL"}
				
		

		if action[0] == 1:
			if action[1] > self.state[0]:
				new_state = [self.state[0],  self.state[1], self.next_opening_price(), \
						cur_value, self.five_day_window()]
				self.state = new_state
				self.reward += -100000
				print("\nEpisode Terminating soldmore  REWARD = ",self.reward)
				
				retval = np.array(new_state),  -100000, True, { "msg": "sold more than have"}
			else:
				apl_shares = self.state[0] - action[1]
				cash_gained = action[1] * apl_open[cur_timestep] * 0.9
				new_state = [apl_shares , self.state[1] + cash_gained, self.next_opening_price(), \
					   self.next_open_price(apl_shares) + self.state[1] + cash_gained, self.five_day_window()]
				self.state = new_state
				cur_value = self.portfolio_value()
				gain = cur_value - self.starting_portfolio_value
				self.reward += -self.inaction_penalty-ts_left +gain*1000
				retval = np.array(new_state),  -self.inaction_penalty-ts_left + gain*1000, False, { "msg": "sold AAPL"}
				

				
		print("\n action taken: ",action, " pf- " ,self.state[3]," - ",self.state[0],  " - ",self.state[1])
		self.cur_timestep += self.stride

		return retval

	def reset(self):
		self.state = np.zeros(5)
		self.starting_cash = 200
		self.cur_timestep = 1
		self.state[0] = 120
		self.state[1] = 4000
		self.state[2] = apl_open[self.cur_timestep]
		self.starting_portfolio_value = self.portfolio_value_states()
		self.state[3] = self.starting_portfolio_value
		self.state[4] = self.five_day_window()
		self.done = False
		self.reward = 0
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
		
		return apl5
	
	
	def render(self, mode='human'):
		print("Render called")
		
	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None
