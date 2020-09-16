import gym
import cv2
# env = gym.make('SpaceInvaders-v0')
env = gym.make('Breakout-v4')

env.reset()
step = 0
# for _ in range(1000):
while True:
    #env.action_space.sample()
    img,reward,end,lives= env.step(step)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    lives = lives['ale.lives']
    print(reward,end,lives)
    # env.render('human')
    cv2.imshow('img', img)
    key = cv2.waitKey(0) & 0xFF
    # print(key)
    if key >= 48 and key <(48+6):
        step = key-48
    else:
        step = 0

    if end:
        break
    # step = 0
    # if key == ord('a'):
    #     step = 0
    # elif key == ord('d'):
    #     step = 5
    # else:
    #     step = 3



env.close()  # https://github.com/openai/gym/issues/893


