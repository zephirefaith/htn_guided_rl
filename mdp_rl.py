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

# Implements TD Q-learning
# Pages: 843, 844
# Artificial Intelligence: A Modern Approach 3rd Edition
# Stuart J. Russell and Peter Norvig

import MalmoPython
import json
import logging
import os
import random
import sys
import time
import Tkinter as tk

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
        self.actions = ["move 1", "turn 1", "turn -1", "look 1", "look -1", "attack 1", "use 1", "slot 0", "slot 1"]
        self.decompose_action = {"slot 0":["hotbar.0 1", "hotbar.0 0"], "slot 1":["hotbar.1 1", "hotbar.1 0"]}
        # q-learning specific
        self.q_table = {}
        self.gamma = 0.70
        self.learning_rate = 0.85
        self.exploration="e-greedy"
        self.epsilon = 0.05
        # gold_room specific
        self.min_x = -70
        self.min_y = 13
        self.min_z = -54
        # HTN specific
        self.relevant_items = ['gold']
        room = ['wall', 'stairs']
        scenario = 0 # change for different scenarios
        if room[scenario] = 'wall':
            self.relevant_items.append('glass')
        else
            self.relevant_items.append('brick')
        # for evaluation
        self.avg_q = 0
        self.num_moves = 0

    def updateQTable( self, reward, current_state ):
        """Change q_table to reflect what we have learnt."""
        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]
        # what should the new action value be?
        # using Q-learning to update qtable
        max_current_q = max(self.q_table[current_state][:])
        new_q = (1 - self.learning_rate) * old_q + self.learning_rate * ( reward + self.gamma * max_current_q - old_q)
        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q

    def updateQTableFromTerminatingState( self, reward ):
        """Change q_table to reflect what we have learnt, after reaching a terminal state."""
        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]
        # what should the new action value be?
        new_q = reward
        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q

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
                self.avg_q + = self.q_table[current_s][rand_id]
                a = rand_id
            else:
                self.avg_q + = max(self.q_table[current_s][:])
                a = np.argmax(self.q_table[current_s][:])
            return a
        if self.exploration == "boltzmann":
            #Choose an action probabilistically, with weights relative to the Q-values.
            Q_d,allQ = sess.run([q_net.Q_dist,q_net.Q_out],feed_dict={q_net.inputs:[s],q_net.Temp:e,q_net.keep_per:1.0})
            a = np.random.choice(Q_d[0],p=Q_d[0])
            a = np.argmax(Q_d[0] == a)
        return a

    def act(self, world_state, agent_host, current_r ):
        """take 1 action in response to the current world state"""
        # make new knowledge based state for MDP
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text) # most recent observation
        # debug log information and store as current_s
        self.logger.debug(obs)
        try:
            los = ob[u'LineOfSight']
        except RuntimeError as e:
            print
            print 'ERROR: ',e
            exit(1)
        block_type = los[u'type']
        in_range = los[u'inRange']
        current_s = [block_type, in_range, self.object_in_hand, self.pitch_count, self.prev_a]
        if block_type in self.relevant_items:
            current_r += 50
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        current_loc = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        self.logger.debug("State: %s %d %d %d %d" % (current_s[0], current_s[1], current_s[2], current_s[3], current_s[4]))
        if not self.q_table.has_key(current_s):
            self.q_table[current_s] = ([0] * len(self.actions))
        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTable( current_r, current_s )
        self.drawQ( curr_x = int(obs[u'XPos']), curr_y = int(obs[u'ZPos']) )
        # select the next action
        a = self.choose_action( current_s )
        self.logger.info("Taking action: %s" % self.actions[a])
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
            self.num_moves += 1
        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)
        return current_r

    def run(self, agent_host):
        """run the agent on the world"""
        total_reward = 0
        self.prev_s = None
        self.prev_a = None
        is_first_action = True
        # main loop:
        world_state = agent_host.getWorldState()
        while world_state.is_mission_running:
            current_r = 0
            if is_first_action:
                # start with zero initial q_value and num_moves per iteration
                self.avg_q = 0
                self.num_moves = 0
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
        world_x = 8
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
        max_value = 20
        for x in range(world_x):
            for y in range(world_y):
                s = "%d:%d" % (x + self.min_x, y + self.min_z)
                value = max( self.q_table[s][:] )
                color = 255 * ( value - min_value ) / ( max_value - min_value ) # map value to 0-255
                color = max( min( color, 255 ), 0 ) # ensure within [0,255]
                color_string = '#%02x%02x%02x' % (255-color, color, 0)
                if not s in self.q_table:
                    self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill="#000")
                    continue
                else:
                    self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline=color_string, fill=color_string)
        if curr_x is not None and curr_y is not None:
            curr_x = curr_x - self.min_x
            curr_y = curr_y - self.min_z
            self.canvas.create_rectangle( curr_x*scale, curr_y*scale, (curr_x+1)*scale, (curr_y+1)*scale, outline="#fff", fill="#111")
        self.root.update()

# store reward_list, num_moves_per_episode, avg_q_value_per_episode
reward_list = []
move_list = []
avg_q_list = []
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
agent = TabQAgent()
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
max_retries = 3
if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = 200
cumulative_rewards = []
for i in range(num_repeats):
    print
    print 'Repeat %d of %d' % ( i+1, num_repeats )
    my_mission_record = MalmoPython.MissionRecordSpec()
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
    cumulative_reward, num_moves, avg_q = agent.run(agent_host)
    reward_list.append(cumulative_reward)
    move_list.append(num_moves)
    avg_q_list.append(avg_q)
    print 'Cumulative reward: %d' % cumulative_reward
    cumulative_rewards += [ cumulative_reward ]
    # -- clean up -- #
    time.sleep(0.5) # (let the Mod reset)
print "Done."
print
print "Cumulative rewards for all %d runs:" % num_repeats
print cumulative_rewards
