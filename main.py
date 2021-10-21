# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:58:20 2021

@author: black
"""

#alternative way to set it up, not using a func as ganspace WHICH DOES CAUSE A 
#PROBLEM IN COLAB AND NEEDS THE FOLLOWING LINES TO WORK:
#SOURCE: https://stackoverflow.com/questions/42249982/systemexit-2-error-when-calling-parse-args-within-ipython/42250328#42250328

#Multi-line comment: https://stackoverflow.com/questions/61563584/how-do-you-indent-in-google-colab

import argparse, os
import numpy as np
from utils import plot_learning_curve, make_env
import agents as Agents

#import sys
#sys.argv=[''] => useful if defining on colab
#del sys


#REFACTORING

#THIS SHOULD BE ITS OWN main.py file in github!

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN hyperparameters configuration") 
        
    parser.add_argument('-lr',  type=float, default=1e-3, help='Learning Rate') 
    parser.add_argument('-n_games',   type=int, default=1, help='Number of episodes to play') 
    parser.add_argument('-eps_min',  type=float, default=0.01, help='Minimum exploration rate (epsilon)') 
    parser.add_argument('-gamma',  type=float, default= 0.99, help='Discount factor (gamma)') 
    parser.add_argument('-eps_dec',  type=float, default=1e-5, help='Epsilon decay factor per episode') 
    parser.add_argument('-epsilon',   type=float, default = 1.0, help='Initial exploration rate (epsilon)') 
    parser.add_argument('-mem_size',   type=int, default=5000, help='Memory Buffer size') 
    parser.add_argument('-skip',   type=int, default=4, help='Number of frames to stack for environment') 
    parser.add_argument('-batch_size',   type=int, default = 32, help='Batch size retrieved in replay memory')
    parser.add_argument('-replace',   type=int, default=1000, help='Number of steps to update target network') 

    #could make that fancier by dictionary or func to append "-v4" automatically to make things easier for end user
    parser.add_argument('-env_name',   type=str, default="PongNoFrameskip-v4", help="""Atari environment. \nPongNoFrameskip-v4\n  
                              BreakoutNoFrameskip-v4\n 
                              SpaceInvadersNoFrameskip-v4\n 
                              EnduroNoFrameskip-v4\n 
                              AtlantisNoFrameskip-v4""") 
    
    parser.add_argument('-gpu',   type=str, default='0', help='GPU: 0 or 1') 

    parser.add_argument('-algo',  type=str, default="DQNAgent", 
                        help="Options for saving: DQNAgent, DDQNAgent, DuelingQNAgent, DuelingDDQNAgent") 
    
    parser.add_argument('-load_checkpoint',  type=bool, default=False,
                        help="load model checkpoint")
    
    parser.add_argument('-path',  type=str, default="/content", 
                        help="path for model saving/loading")
    
    #Additional arguments taken from solution github to parameterize env
    parser.add_argument('-clip_rewards', type=bool, default=False,
                        help='Clip rewards to range -1 to 1')
    parser.add_argument('-no_ops', type=int, default=0,
                        help='Max number of no ops for testing')
    parser.add_argument('-fire_first', type=bool, default=False,
                        help='Set first action of episode to fire')
    parser.add_argument('-repeat', type=int, default=4,
                            help='Number of frames to repeat & stack')
    
    args = parser.parse_args() 


    #bonus:if you have multiple GPUs: Run separate models on separate GPUs
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #env = make_env(args.env_name)
    env = make_env(env_name=args.env_name, repeat=args.repeat,
                  clip_rewards=args.clip_rewards, no_ops=args.no_ops,
                  fire_first=args.fire_first)

    best_score = -np.inf 

    #even more modular, as suggested by course
    agent_ = getattr(Agents, args.algo)

    agent = agent_(gamma = args.gamma,
                     epsilon = args.epsilon, 
                     alpha=args.lr, 
                     input_dims = env.observation_space.shape,
                     n_actions = env.action_space.n,
                     mem_size = args.mem_size,
                     eps_min = args.eps_min,
                     batch_size = args.batch_size,
                     replace = args.replace,
                     eps_dec = args.eps_dec,
                     chkpt_dir = args.path,
                     algo=args.algo,
                     env_name = args.env
                     )
    if args.load_checkpoint:
      agent.load_models()

    fname = args.algo +"_" + args.env_name + "_lr" + str(args.lr) + "_" + \
    "_" + str(args.n_games) + 'games'

    figure_file = "/content" + fname + '.png'

    #debugging statement
    n_steps = 0 
    scores, eps_history, steps_array = [], [], []

    for i in range(args.n_games):
      done, score, state = False, 0, env.reset()

      while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        score += reward

        #only store transitions if we are training, not during inference
        if not args.load_checkpoint:
          #want done as int for use in our mask 
          agent.store_transition(state, action, reward, next_state, int(done))

          agent.learn()

        state = next_state
        n_steps += 1 
      scores.append(score)
      steps_array.append(n_steps)

      avg_score = np.mean(scores[-100:])
      print("episode: ", i, 
            "score:", "score", score, 
            " average score %.1f" % avg_score,
            "best score %.2f" % best_score, 
            'epsilon %.2f' % agent.epsilon,
            'steps', n_steps)

      if avg_score > best_score:
        if not args.load_checkpoint:
          agent.save_models()
        best_score = avg_score

      eps_history.append(agent.epsilon)

    plot_learning_curve(steps_array, scores, eps_history, figure_file)