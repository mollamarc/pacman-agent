# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    def __init__(self, index):
        super().__init__(index)
        self.capsule_eaten = False  # Track if a capsule has been eaten
        self.capsule_timer = 0     # Track steps since capsule was eaten
        self.last_paths = []       # Track the last 2 paths taken to avoid repetition
        self.border_distance = 0   # Track distance to the border
        self.food_collected = 0    # Track the number of food pellets collected and returned to base
        self.target_food = None    # Track the target food inside the ally base
        self.base_timer = 0        # Track the time until the agent returns to be pacman 

    def choose_action(self, game_state):
        """
        Picks the best action based on the current game state.
        Incorporates staying home when a ghost is nearby, capsule behavior, and path diversity.
        """
                
        center_x = (self.get_food(game_state).width // 2) - 1 if self.red else (self.get_food(game_state).width // 2)
        center_y = self.get_food(game_state).height // 2 
        top_y = self.get_food(game_state).height
        
        safe_center_x = (self.get_food(game_state).width // 2) - 3 if self.red else (self.get_food(game_state).width // 2) + 2
        center = (center_x, center_y)
        safe_center = (safe_center_x, center_y)
        bottom_center = (center_x,1)
        top_center = (center_x,top_y-2)
        
        # Get legal actions for the current state
        actions = game_state.get_legal_actions(self.index)

        # Get current position and state
        my_pos = game_state.get_agent_state(self.index).get_position()
        is_pacman = game_state.get_agent_state(self.index).is_pacman

        # Get food and capsules on the opponent's side
        food_list = self.get_food(game_state).as_list()
        capsule_list = self.get_capsules(game_state)

        # Get enemy ghosts
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None]
        
        # Check if any ghost is too close (within 5 steps)
        ghost_nearby = False
        ghost_nearby_base = False

        if ghosts:
            closest_ghost_dist = min([self.get_maze_distance(my_pos, g.get_position()) for g in ghosts])
            ghost_nearby = closest_ghost_dist <= 3
            ghost_nearby_base = closest_ghost_dist <= 6
        
        # Update capsule timer
        if self.capsule_eaten:
            self.capsule_timer += 1
            if self.capsule_timer > 25:
                self.capsule_eaten = False
                #self.capsule_timer = 0
        
        # Update food collected count
        if not is_pacman:
            # Reset food collected count when returning to base
            self.food_collected = 0 
        
        # If just returned home and ghost is chasing initiate base timer        
        if not is_pacman and ghost_nearby_base:
            self.base_timer += 1
        
        print(self.base_timer)
        
        # If timer initiated,find the nearest food inside the ally base and return to pacman after 5s
        if self.base_timer <= 15 and self.base_timer > 0:
            self.base_timer += 1
            # Look for closest food
            if self.base_timer > 10 and not ghost_nearby:
                closest_food = min(food_list, key=lambda x: self.get_maze_distance(my_pos, x))
                best_action = self.move_toward_target(game_state, closest_food, actions)
                if best_action:
                    # Increment food collected count when eating food
                    if is_pacman:
                        # Check if the agent's current position matches the food position
                        check_successor = self.get_successor(game_state, best_action)
                        new_pos = check_successor.get_agent_position(self.index)
                        if new_pos == closest_food:
                            self.food_collected += 1
                    return best_action
                
            # Move toward the target position
            dist_to_top = self.get_maze_distance(my_pos, top_center)
            dist_to_center = self.get_maze_distance(my_pos, safe_center)
            dist_to_bottom = self.get_maze_distance(my_pos, bottom_center)

            if self.red:
                if dist_to_top < dist_to_center:
                    closest_ref_point = top_center
                elif game_state.get_agent_state(self.index).get_position() == bottom_center:
                    closest_ref_point = bottom_center
                else:
                    closest_ref_point = safe_center

                best_action = self.move_toward_target(game_state, closest_ref_point, actions)
                
            else: 
                if dist_to_bottom < dist_to_center:
                    closest_ref_point = bottom_center
                elif game_state.get_agent_state(self.index).get_position() == top_center:
                    closest_ref_point = top_center
                else:
                    closest_ref_point = safe_center                
                best_action = self.move_toward_target(game_state, closest_ref_point, actions)

            if best_action:
                    return best_action
            # If top/bottom position not foun, choose a random action
            return random.choice(actions)
        
        if self.base_timer >= 15:
            self.base_timer = 0


        # If a capsule is eaten, prioritize eating food within 1 step, then farthest food
        if self.capsule_eaten and self.capsule_timer < 25:
            #print('not reverted')
            if food_list:
                # Check if there is food within 1 step
                food_within_1_step = [food for food in food_list if self.get_maze_distance(my_pos, food) == 1]
                if food_within_1_step:
                    # Prioritize eating food within 1 step
                    closest_food = min(food_within_1_step, key=lambda x: self.get_maze_distance(my_pos, x))
                    best_action = self.move_toward_target(game_state, closest_food, actions)
                    if best_action:
                        return best_action
                else:
                    # Otherwise, prioritize eating food farthest from home
                    farthest_food = max(food_list, key=lambda x: self.get_maze_distance(self.start, x))
                    best_action = self.move_toward_target(game_state, farthest_food, actions)
                    if best_action:
                        return best_action

        # If carrying food and a ghost is nearby, return home
        if (is_pacman and ghost_nearby and self.capsule_eaten == False) or (len(food_list) < 3) or (self.food_collected > 2 and self.capsule_eaten == False):
            best_dist = float('inf')
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        # If there are capsules and no ghost is nearby, and we have scored 3 points, go for the capsule
        if capsule_list and not ghost_nearby:
            closest_capsule = min(capsule_list, key=lambda x: self.get_maze_distance(my_pos, x))
            best_action = self.move_toward_target(game_state, closest_capsule, actions)
            if best_action:
                # Update capsule state if a capsule is eaten
                if closest_capsule not in self.get_capsules(game_state.generate_successor(self.index, best_action)):
                    self.capsule_eaten = True
                    self.capsule_timer = 0
                return best_action

        # If there is food, go for the closest food
        if food_list:
            closest_food = min(food_list, key=lambda x: self.get_maze_distance(my_pos, x))
            best_action = self.move_toward_target(game_state, closest_food, actions)
            if best_action:
                # Increment food collected count when eating food
                if is_pacman:
                    # Check if the agent's current position matches the food position
                    check_successor = self.get_successor(game_state, best_action)
                    new_pos = check_successor.get_agent_position(self.index)
                    if new_pos == closest_food:
                        self.food_collected += 1
                        #print(f"Food collected: {self.food_collected}")
                return best_action

       
        # Default: choose a random action
        return random.choice(actions)

    def move_toward_target(self, game_state, target, actions):
        """
        Helper function to move toward a specific target (food, capsule, or home).
        """
        best_dist = float('inf')
        best_action = None
        for action in actions:
            successor = self.get_successor(game_state, action)
            pos2 = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(target, pos2)
            if dist < best_dist:
                best_action = action
                best_dist = dist
        return best_action


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def __init__(self, index):
        super().__init__(index)
        self.capsule_eaten = False  # Track if a capsule has been eaten
        self.capsule_timer = 0     # Track steps since capsule was eaten
        self.current_target_index = 0  # Track which of the 2 nearest food items to target

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # Get current position and state
        my_pos = game_state.get_agent_state(self.index).get_position()
        is_pacman = game_state.get_agent_state(self.index).is_pacman

        # Get food and capsules on our side
        food_list = self.get_food(game_state).as_list()
        capsule_list = self.get_capsules(game_state)

        # Get enemy agents
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]

        # Update capsule timer
        if self.capsule_eaten:
            self.capsule_timer += 1
            if self.capsule_timer > 25:
                self.capsule_eaten = False
                #self.capsule_timer = 0
                
        # Check if a capsule has been eaten (no capsules left on the enemy side)
        if len(capsule_list) == 0 and self.capsule_timer < 25:
            self.capsule_eaten = True
            #self.capsule_timer = 0

        # If within the 30-step capsule timer, go offensive and get the nearest food
        if self.capsule_eaten:
            if food_list:
                closest_food = min(food_list, key=lambda x: self.get_maze_distance(my_pos, x))
                best_action = self.move_toward_target(game_state, closest_food, actions)
                if best_action:
                    return best_action

        # Otherwise, stay defensive
        if not invaders:
            # Get the list of food positions
            food_list = self.get_food(game_state).as_list()

            # Calculate the center of the map
            center_x = (game_state.data.layout.width // 2)
            center_y = game_state.data.layout.height // 2

            # Find the 2 nearest food to the center
            def distance_to_center(food_pos):
                return abs(food_pos[0] - center_x) + abs(food_pos[1] - center_y)

            ally_food_list = self.get_food_you_are_defending(game_state).as_list()
            # Sort food by distance to the center
            sorted_food = sorted(ally_food_list, key=distance_to_center)

            # Select the 2 nearest food to the center
            nearest_food = sorted_food[:2]

            # If there are at least 2 food items, alternate between them
            if len(nearest_food) >= 2:
                # Get Pacman's current position
                pacman_pos = game_state.get_agent_position(self.index)

                # Determine which of the 2 nearest food to move to next
                target_food = nearest_food[self.current_target_index]

                # If Pacman reaches the target food, switch to the other one
                if pacman_pos == target_food:
                    self.current_target_index = 1 - self.current_target_index  # Toggle between 0 and 1

                # Move to the current target food
                best_action = self.move_toward_target(game_state, target_food, actions)
                if best_action:
                    return best_action

            # If there's only 1 food item, go to it
            elif len(nearest_food) == 1:
                best_action = self.move_toward_target(game_state, nearest_food[0], actions)
                if best_action:
                    return best_action

            # If no food is left, stay at the center
            else:
                center_position = (center_x, center_y)
                best_action = self.move_toward_target(game_state, center_position, actions)
                if best_action:
                    return best_action
        else:
            # If there are invaders, cut off their nearest path to their base
            closest_invader = min(invaders, key=lambda x: self.get_maze_distance(my_pos, x.get_position()))
            closest_invader_pos = closest_invader.get_position()

            # If the invader is within 2 steps, follow them to eat them
            if self.get_maze_distance(my_pos, closest_invader_pos) <= 2:
                best_action = self.move_toward_target(game_state, closest_invader_pos, actions)
                if best_action:
                    return best_action
            else:
                # Otherwise, cut off their path to their base
                # For simplicity, assume their base is on the opposite side of the board
                #if self.red:
                #    base_x = game_state.data.layout.width // 2 - 1
                #else:
                    #base_x = game_state.data.layout.width // 2
                #base_y = int(closest_invader_pos[1])
                #base_position = (base_x, base_y)

                # Move toward a point along the invader's path to their base
                best_action = self.move_toward_target(game_state, closest_invader_pos, actions)
                if best_action:
                    return best_action

        # Default: choose a random action
        return random.choice(actions)

    def move_toward_target(self, game_state, target, actions):
        """
        Helper function to move toward a specific target (food, capsule, or home).
        """
        best_dist = float('inf')
        best_action = None
        for action in actions:
            successor = self.get_successor(game_state, action)
            pos2 = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(target, pos2)
            if dist < best_dist:
                best_action = action
                best_dist = dist
        return best_action








