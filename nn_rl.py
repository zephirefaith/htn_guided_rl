# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Structure borrowed from:
# Malmo's Python Tutorial sample #6: Discrete movement, rewards, and learning

# Implements Policy-based Reinforcement Learning using Neural Networks and Visual Inputs

# Malmo specific imports
import MalmoPython
import json
import logging
import os
import random
import sys
import time
import Tkinter as tk

# Tensorflow specific imports
import numpy as np
import cPickle as pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import tensorflow.contrib.slim as slim

# modified class from TabQAgent for a neural net based RL agent
class QAgent:
    """Q-learning agent for discrete state/action spaces using LSTM/ConvNN and RGB-D data"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if False: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        # actions 7 and 8 are abstracted actions made up of two minecraft hotbar actions
        self.actions = ["move 1", "turn 1", "turn -1", "look 1", "look -1", "attack 1", "use 1", "slot 0", "slot 1"]
        self.decompose_action = {"slot 0":["hotbar.0 1", "hotbar.0 0"], "slot 1":["hotbar.1 1", "hotbar.1 0"]}
        # initialize neural net
        # TODO
	self.scalarInput =  tf.placeholder(shape=[None,25920],dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,4,108,60])
        self.conv1 = tf.contrib.layers.convolution2d( \
            inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
        self.conv2 = tf.contrib.layers.convolution2d( \
            inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
        self.conv3 = tf.contrib.layers.convolution2d( \
            inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
        self.conv4 = tf.contrib.layers.convolution2d( \
            inputs=self.conv3,num_outputs=512,kernel_size=[7,7],stride=[1,1],padding='VALID', biases_initializer=None)
        # hyperparameters
        self.exploration="boltzmann"
        self.gamma = 0.75
        self.learning_rate = 0.85
        # puzzle_room world specific
        self.min_x = -70
        self.min_y = 13
        self.min_z = -54
        # video data specific
        self.video_height = 60
        self.video_width = 108

    def updateQTable( self, reward, current_state ):
        """Update network to reflect what we have learnt."""

    def vid2tensor( self, current_frame):
        """Helper function to change current state's video frame to input tensor for nn"""

    def choose_action( self):
        """Helper function for choosing next action depending on different strategies"""
        """greedy, random, e-greedy, boltzmann, bayesian"""
	if self.exploration == "greedy":
            #Choose an action with the maximum expected value.
            a,allQ = sess.run([q_net.predict,q_net.Q_out],feed_dict={q_net.inputs:[s],q_net.keep_per:1.0})
            a = a[0]
            return a
        if self.exploration == "random":
            #Choose an action randomly.
            a = env.action_space.sample()
        if self.exploration == "e-greedy":
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = env.action_space.sample()
            else:
                a,allQ = sess.run([q_net.predict,q_net.Q_out],feed_dict={q_net.inputs:[s],q_net.keep_per:1.0})
                a = a[0]
            return a
        if self.exploration == "boltzmann":
            #Choose an action probabilistically, with weights relative to the Q-values.
            Q_d,allQ = sess.run([q_net.Q_dist,q_net.Q_out],feed_dict={q_net.inputs:[s],q_net.Temp:e,q_net.keep_per:1.0})
            a = np.random.choice(Q_d[0],p=Q_d[0])
            a = np.argmax(Q_d[0] == a)
            return a
        if self.exploration == "bayesian":
            #Choose an action using a sample from a dropout approximation of a bayesian q-network.
            a,allQ = sess.run([q_net.predict,q_net.Q_out],feed_dict={q_net.inputs:[s],q_net.keep_per:(1-e)+0.1})
            a = a[0]
        return a

    def act(self, world_state, agent_host, current_r ):
        """take 1 action in response to the current world state"""
        # acquiring latest observation and video frame
        current_frame = world_state.video_frames[-1].pixel
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text) # most recent observation
        # storing state information for debugging
        self.logger.debug(obs)
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        # change current state to video_frame
        current_s = self.vid2tensor(current_frame)
        current_loc = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_loc, float(obs[u'XPos']), float(obs[u'ZPos'])))
        # update Q values
        # TODO
        # select the next action
        a = self.choose_action()
        self.logger.info("Next action: %s" % self.actions[a])
        # try to send the selected action, only update prev_s if this succeeds
        try:
            # use decomposed actions in succession for "slot 0" and "slot 1" command
            if a < 7:
                agent_host.sendCommand(self.actions[a])
            else:
                agent_host.sendCommand(self.decompose_action[self.actions[a]][0])
                agent_host.sendCommand(self.decompose_action[self.actions[a]][1])
            self.prev_s = current_s
            self.prev_a = a
        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)
        return current_r

    def run(self, agent_host):
        """run the agent on the world"""

        # start conditions
        total_reward = 0
        self.prev_s = None
        self.prev_a = None
        is_first_action = True
        # main loop:
        #grab world state and continue if mission is running
        world_state = agent_host.getWorldState()
        while world_state.is_mission_running:
            current_r = 0
            if is_first_action:
                # wait until have received a valid observation
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False
            else:
                # wait for non-zero reward
                while world_state.is_mission_running and current_r == 0:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                # allow time to stabilise after action
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break

        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTableFromTerminatingState( current_r )

        self.drawQ()

        return total_reward

#main()
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
# create an RL agent
agent = QAgent()
# create a Minecraft agent
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print 'ERROR:',e
    print agent_host.getUsage()
    exit(1)
if agent_host.receivedArgument("help"):
    print agent_host.getUsage()
    exit(0)
# -- set up the mission -- #
mission_file = './wall_room.xml'
with open(mission_file, 'r') as f:
    print "Loading mission from %s" % mission_file
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
# number of retries for starting the mission
max_retries = 3
# option for testing
if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = 150
#rewards array for plotting
cumulative_rewards = []
# main mission loop
for i in range(num_repeats):
    print
    print 'Repeat %d of %d' % ( i+1, num_repeats )
    my_mission_record = MalmoPython.MissionRecordSpec()
    # minecraft agent creates world and starts mission
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print "Error starting mission:",e
                exit(1)
            else:
                time.sleep(2.5)
    print "Waiting for the mission to start",
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print "Error:",error.text
    print
    # -- run the agent in the world -- #
    # RL agent runs in the world/mission created by agent_host - the minecraft agent
    cumulative_reward = agent.run(agent_host)
    print 'Cumulative reward: %d' % cumulative_reward
    cumulative_rewards += [ cumulative_reward ]

    # -- clean up -- #
    time.sleep(0.5) # (let the Mod reset)

print "Done."

print
print "Cumulative rewards for all %d runs:" % num_repeats
print cumulative_rewards
