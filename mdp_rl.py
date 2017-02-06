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
import os
import sys
import time
import matplotlib.pyplot as plt
from QAgent import TabQAgent

# store reward_list, num_moves_per_episode, avg_q_value_per_episode
reward_list = []
move_list = []
avg_q_list = []
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
agent = TabQAgent()
agent_host = MalmoPython.AgentHost()
test = False
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
mission_file = 'mdp_version/wall_room.xml'
with open(mission_file, 'r') as f:
    print "Loading mission from %s" % mission_file
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
max_retries = 3
if test:
    num_repeats = 5
else:
    num_repeats = 250
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
        time.sleep(2.0)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print "Error:",error.text
    print
    # -- run the agent in the world -- #
    if not test: #use to toggle between test and RL execution
        cumulative_reward, avg_q, num_moves = agent.run(agent_host)
        reward_list.append(cumulative_reward)
        move_list.append(num_moves)
        avg_q_list.append(avg_q)
        print 'Cumulative reward: {0}, Number of Moves: {1}, Average Q-value: {2}'.format(cumulative_reward, num_moves, avg_q)
        cumulative_rewards += [ cumulative_reward ]
        # -- clean up -- #
        time.sleep(1.0) # (let the Mod reset)
    else:
        time.sleep(30) #let the human do the thang
print "Done."
print
print "Cumulative rewards for all %d runs:" % num_repeats
print cumulative_rewards
plt.plot(reward_list)
plt.plot(move_list)
plt.plot(avg_q_list)
