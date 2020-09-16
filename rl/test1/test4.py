#%%
import gym
env = gym.make('SpaceInvaders-v0')
env.reset()

env.step(env.action_space.sample())
# env.render('human')