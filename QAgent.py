from __future__ import division
import MalmoPython
import json
import logging
import os
import random
import sys
import time
import Tkinter as tk
import numpy as np

class TabQAgent:
    """Tabular one-step Q-learning agent for discrete state/action spaces."""

    def __init__(self):
        # initializing logging services
        self.logger = logging.getLogger(__name__)
        if False: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        # action library
        self.pitch_id = [ 3, 4 ]
        self.hotbar_id = [ 7, 8 ]
        self.actions = [
                "move 1",
                "turn 1",
                "turn -1",
                "look 1", #down
                "look -1", #up
                "attack 1",
        #        "use 1",
        #        "slot 0",
        #        "slot 1"
                ]
        #self.decompose_action = {
        #        "slot 0":[
        #            "hotbar.1 1",
        #            "hotbar.1 0"
        #            ],
        #        "slot 1":[
        #            "hotbar.2 1",
        #            "hotbar.2 0"
        #            ]
        #        }
        # q-learning specific
        self.q_table = {}
        self.loc_table = {}
        self.gamma = 0.90
        self.learning_rate = 0.75
        self.exploration="e-greedy"
        self.epsilon = 0.1
        self.scale = 10
        # gold_room specific
        self.min_x = -68
        self.min_y = 13
        self.min_z = -52
        # HTN specific
        self.pitch_count = 0
        self.object_in_hand = 0
        self.relevant_items = [u'gold']
        room = ['wall', 'stairs']
        scenario = 0 # change for different scenarios
        if room[scenario] == 'wall':
            self.relevant_items.append(u'glass')
        else:
            self.relevant_items.append(u'brick')
        # for evaluation
        self.avg_q = 0
        self.num_moves = 0
        # for visualization
        self.current_loc = ()
        self.prev_loc = ()
        self.canvas = None
        self.root = None

    def reScale(self, row):
        """Scales rows of RL-MDP to restrict s.t. sum of all Q-values is 50"""
        max_lim = max(row)
        min_lim = min(row)
        # scale to 0 to 10
        if max_lim != min_lim:
            new_row = [((float(i)-min_lim)*10)/(max_lim-min_lim) for i in row]
        else:
            new_row = row
        # scale s.t. sum is 50
        sum_row = sum(new_row)
        new_row = [float(i)*50/sum_row for i in new_row]
        return new_row

    def updateQTable( self, reward, current_state ):
        """Change q_table to reflect what we have learnt."""
        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]
        # using Q-learning to update qtable
        max_current_q = max(self.q_table[current_state])
        new_q = old_q + self.learning_rate * ( reward + self.gamma * max_current_q - old_q)
        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q
        # normalize the values
        #self.q_table[self.prev_s] = self.reScale(self.q_table[self.prev_s])
        self.logger.debug("Prev: {0}, After scaling: {1}".format(new_q, self.q_table[self.prev_s][self.prev_a]))
        self.loc_table[self.prev_loc] = self.q_table[self.prev_s][self.prev_a]
        self.logger.debug("Max q-value at last location: "+str(self.loc_table[self.prev_loc]))

    def updateQTableFromTerminatingState( self, reward ):
        """Change q_table to reflect what we have learnt, after reaching a terminal state."""
        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]
        # what should the new action value be?
        new_q = reward
        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = self.learning_rate * new_q
        #self.q_table[self.prev_s] = self.reScale(self.q_table[self.prev_s])
        self.loc_table[self.prev_loc] = self.q_table[self.prev_s][self.prev_a]

    def choose_action( self, current_s):
        """Helper function for choosing next action depending on different strategies"""
        """greedy, random, e-greedy, boltzmann"""
        # TODO modify greedy, random, boltzmann for MDP
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
            if np.random.rand(1) < self.epsilon :
                rand_id = np.random.randint(len(self.actions))
                self.avg_q += self.q_table[current_s][rand_id]
                a = rand_id
            else:
                self.avg_q += max(self.q_table[current_s][:])
                a = np.argmax(self.q_table[current_s][:])
            return a
        if self.exploration == "boltzmann":
            #Choose an action probabilistically, with weights relative to the Q-values.
            Q_d,allQ = sess.run([q_net.Q_dist,q_net.Q_out],feed_dict={q_net.inputs:[s],q_net.Temp:e,q_net.keep_per:1.0})
            a = np.random.choice(Q_d[0],p=Q_d[0])
            a = np.argmax(Q_d[0] == a)
        return a

    def process_observation( self, observation):
        """processes current observation to form a state for MDP-RL"""
        # returns 6 blocks right in front of the agent, what it is staring at, item_count and pitch status
        # get yaw, and depending upon the yaw  make current state out of the 9 blocks right in front
        self.logger.debug(observation)
        direction = {'left':90.0,'right':270,'forward':180.0,'backward':0.0}
        yaw = observation.get(u'Yaw')
        if yaw is None:
            print "Incomplete Observation:"
            print(observation)
            exit(1)
        if observation.has_key(u'LineOfSight'):
            los = observation.get(u'LineOfSight')
            block_type = los[u'type']
            in_range = los[u'inRange']
        else:
            block_type = 'undefined'
            in_range = False
        # extract grid from observation
        grid = observation.get(u'around9x9', 0)
        if grid is None:
            print "Incomplete Observation: " + observation
            exit(1)
        self.logger.debug(grid)
        flag = 0
        item_count = 0
        # format "front" depending upon Yaw of the agent
        if yaw == direction['left']:
            self.logger.debug("%%Facing left%%")
            front_idx = range(5*3+1,5*1+0,-5) + range(5*3+26, 5*1+25, -5) + range(5*3+51, 5*1+50, -5)
        elif yaw == direction['right']:
            self.logger.debug("%%Facing right%%")
            front_idx = range(5*1+3,5*3+4,5) + range(5*1+28, 5*3+29, 5) + range(5*1+53,5*3+54,5)
        elif yaw == direction['forward']:
            self.logger.debug("%%Facing forward%%")
            front_idx = range(6,9) + range(6+25, 9+25) + range(6+50,9+50)
        else:
            self.logger.debug("%%Facing backward%%")
            front_idx = range(18,15,-1) + range(18+25,15+25,-1) + range(18+50,15+50,-1)
        front = [grid[block_idx] for block_idx in range(len(grid)) if block_idx in front_idx]
        self.logger.debug(front_idx)
        self.logger.debug(front)
        # check if relevant items are on the myopic horizon
        for item in self.relevant_items:
            if item in grid:
                flag = 1
                # get count for item
                item_count = grid.count(item)
                break
        current_s = (front[0],
                front[1],
                front[2],
                front[3],
                front[4],
                front[5],
                front[6],
                front[7],
                front[8],
                block_type,
                in_range,
                item_count,
                self.pitch_count,
                self.prev_a,
                )
        return current_s

    def act(self, world_state, agent_host, current_r ):
        """take 1 action in response to the current world state"""
        # make new knowledge based state for MDP
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text) # most recent observation
        # log information and store as current_s
        self.logger.debug(obs)
        current_s = self.process_observation(obs)
        # setting up additional rewards based on HTN information
        if current_s[-5] in self.relevant_items:
            current_r += 0.5
        if current_s[-3] > 0:
            current_r += 0.15 * current_s[-3]
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        current_loc = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        if not self.q_table.has_key(current_s):
            self.q_table[current_s] = self.reScale(([1] * len(self.actions)))
            self.loc_table[current_loc] = 1/len(self.actions)
        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTable( current_r, current_s )
        self.drawQ( curr_x = int(obs[u'XPos']), curr_y = int(obs[u'ZPos']) )
        # select the next action
        a = self.choose_action( current_s )
        if self.actions[a] == 'attack 1':
            current_r += -5
        self.logger.info(str(current_s[:]) + ", action: " + str( self.actions[a]))
        # try to send the selected action, only update prev_s  and prev_loc if this succeeds
        try:
            # use decomposed actions in succession for "slot 0" and "slot 1" command
            if a < 7:
                agent_host.sendCommand(self.actions[a])
                if a == 3:
                    self.pitch_count = max(self.pitch_count - 1, -2)
                if a == 4:
                    self.pitch_count = min(self.pitch_count + 1, 2)
            else:
                agent_host.sendCommand(self.decompose_action[self.actions[a]][0])
                agent_host.sendCommand(self.decompose_action[self.actions[a]][1])
                if a == 7:
                    self.object_in_hand = 1
                if a == 8:
                    self.object_in_hand = 2
        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)
            exit(1)
        self.prev_s = current_s
        self.prev_loc = current_loc
        self.prev_a = a
        self.num_moves += 1
        return current_r

    def run(self, agent_host):
        """run the agent on the world"""
        total_reward = 0
        self.prev_s = None
        self.prev_a = None
        is_first_action = True
        self.avg_q = 0
        self.num_moves = 0
        self.pitch_count = 0
        # main loop:
        world_state = agent_host.getWorldState()
        while world_state.is_mission_running:
            current_r = 0
            if is_first_action:
                # start with zero initial q_value and num_moves per iteration
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
                        if current_r > 0:
                            self.logger.info("Reward this step:"+str(current_r))
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
        # process average q values this cycle
        self.avg_q = self.avg_q / self.num_moves
        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTableFromTerminatingState( current_r )
        self.drawQ()
        return total_reward, self.avg_q, self.num_moves

    def drawQ( self, curr_x=None, curr_y=None ):
        """draws a representation of the room and updates max_a Q(s,a) for each s"""
        # TODO adjust the x-y limits and change so that whole box changes color
        scale = 40
        world_x = 7
        world_y = 16
        if self.canvas is None or self.root is None:
            self.root = tk.Tk()
            self.root.wm_title("Q-table")
            self.canvas = tk.Canvas(self.root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
            self.canvas.grid()
            self.root.update()
        self.canvas.delete("all")
        # (NSWE to match action order)
        min_value = -20
        max_value = 50
        for x in range(world_x):
            for y in range(world_y):
                s = "%d:%d" % (x + self.min_x, y + self.min_z)
                if not s in self.loc_table:
                    self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#000", fill="#fff")
                    continue
                value = int(self.loc_table[s])
                color = 255 * ( value - min_value ) / ( max_value - min_value ) # map value to 0-255
                color = max( min( color, 255 ), 0 ) # ensure within [0,255]
                color_string = '#%02x%02x%02x' % (255-color, color, 0)
                self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill=color_string)
        if curr_x is not None and curr_y is not None:
            curr_x = curr_x - self.min_x
            curr_y = curr_y - self.min_z
            self.canvas.create_rectangle( curr_x*scale, curr_y*scale, (curr_x+1)*scale, (curr_y+1)*scale, outline="#fff", fill="#555")
        self.root.update()
