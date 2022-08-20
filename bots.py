#!/usr/bin/python

import numpy as np
from collections import deque as Queue
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math

# Throughout this file, ASP means adversarial search problem.


class StudentBot:
    """ Write your student bot here"""

    def __distance_helper(self, state, player_loc):

        # (x, y) coordinate pairs
        queue = Queue()
        queue.append(player_loc)

        # {Coordinate: Distance} pairs
        distance_arr = np.full(np.shape(state.board), 10000)
        distance_arr[player_loc[0], player_loc[1]] = 0

        # (x, y) coordinate pairs
        visited = set()
        visited.add(player_loc)
        while queue:
            current_loc = queue.popleft()

            safe_actions = TronProblem.get_safe_actions(state.board, current_loc)
            for action in safe_actions:
                next_coord = None
                if action == 'U':
                    next_coord = (current_loc[0] - 1, current_loc[1])
                elif action == 'D':
                    next_coord = (current_loc[0] + 1, current_loc[1])
                elif action == 'L':
                    next_coord = (current_loc[0], current_loc[1] - 1)
                elif action == 'R':
                    next_coord = (current_loc[0], current_loc[1] + 1)

                if next_coord not in visited:
                    queue.append(next_coord)
                    visited.add(next_coord)
                    distance_arr[next_coord[0], next_coord[1]] = distance_arr[current_loc[0], current_loc[1]] + 1

        return distance_arr

    def heuristic_func(self, state, asp):
        if asp.is_terminal_state(state):
            if asp.evaluate_state(state)[0] == 1:
                return 1
            else:
                return 0
        player_symbol = str(state.player_to_move() + 1)
        opp_symbol = '0'
        if player_symbol == '1':
            opp_symbol = '2'
        else:
            opp_symbol = '1'
        player_score = 0 # number of squares for the player
        opp_score = 0
        board = state.board
        board_arr = np.array(board)
        index_player = np.where(board_arr == player_symbol)
        index_opp = np.where(board_arr == opp_symbol)

        dist_for_player = self.__distance_helper(state, (index_player[0][0], index_player[1][0]))
        dist_for_opp = self.__distance_helper(state, (index_opp[0][0], index_opp[1][0]))

        rows = np.shape(board_arr)[0]
        cols = np.shape(board_arr)[1]
        exclusive_player_score = 0
        exclusive_opp_score = 0
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                if dist_for_player[row, col] != 10000 and dist_for_opp[row, col] == 10000:
                    exclusive_player_score += 1
                elif dist_for_player[row, col] == 10000 and dist_for_opp[row, col] != 10000:
                    exclusive_opp_score += 1
                if dist_for_player[row, col] < dist_for_opp[row, col]:
                    player_score += 1
                elif dist_for_opp[row, col] < dist_for_player[row, col]:
                    opp_score += 1

        exclusive_player_score = self.sigmoid((exclusive_player_score) / (exclusive_player_score+exclusive_opp_score))
        player_score = self.sigmoid((player_score) / (player_score+opp_score))
	#print(player_score)
	#print(dist_for_player)
        final_score = player_score
        return final_score
    
    def sigmoid(self, x):
        return (1/(1+math.exp(-x)))

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()
        """

        best_move = self.alpha_beta_cutoff(asp, 7, self.heuristic_func)
        if (asp.is_terminal_state(asp.transition(asp.get_start_state(), best_move))):
            board_arr = np.array(asp.get_start_state().board)
            player_symbol = str(asp.get_start_state().player_to_move() + 1)
            index_player = np.where(board_arr == player_symbol)
            possibilities = TronProblem.get_safe_actions(asp.get_start_state().board, (index_player[0][0], index_player[1][0]))
            if (possibilities):
                best_move = possibilities.pop()
        return best_move

    def cleanup(self):
        """
        Input: None
        Output: None

        This function will be called in between
        games during grading. You can use it
        to reset any variables your bot uses during the game
        (for example, you could use this function to reset a
        turns_elapsed counter to zero). If you don't need it,
        feel free to leave it as "pass"
        """

        pass

    def alpha_beta_cutoff(self, asp, cutoff_ply, heuristic_func):
        '''
        heuristic_func - a function that takes in a GameState and outputs a
            real number indicating how good that state is for the player who is
            using alpha_beta_cutoff to choose their action. You do not need to
            implement this function, as it should be provided by whomever is
            calling alpha_beta_cutoff, however you are welcome to write
            evaluation functions to test your implemention. The heuristic_func
            we provide does not handle terminal states, so evaluate terminal
            states the same way you evaluated them in the previous algorithms.
        '''

        start = asp.get_start_state()
        score, move = self.min_ab_cutoff(asp, start, -math.inf, math.inf, cutoff_ply, heuristic_func)
        
        return move

    def max_ab_cutoff(self, asp, state, alpha, beta, ply, heuristic_func):
        if asp.is_terminal_state(state):
            maximizing_player = asp.get_start_state().player_to_move()
            terminal_tuple = asp.evaluate_state(state)
            if(maximizing_player == 0):
                return terminal_tuple[1]*100, None
            else:
                return terminal_tuple[0]*100, None
        if ply <= 0:
            return heuristic_func(state, asp), None
        value = -math.inf
        move = None
        for action in asp.get_available_actions(state):
            max_value, _ = self.min_ab_cutoff(asp, asp.transition(state, action), alpha, beta, ply-1, heuristic_func)
            if max_value > value:
                value, move = max_value, action
                alpha = max(alpha, value)
            if value >= beta:
                return value, move

        return value, move

    def min_ab_cutoff(self, asp, state, alpha, beta, ply, heuristic_func):
        if asp.is_terminal_state(state):
            maximizing_player = asp.get_start_state().player_to_move()
            terminal_tuple = asp.evaluate_state(state)
            if(maximizing_player == 0):
                return terminal_tuple[1]*100, None
            else:
                return terminal_tuple[0]*100, None
        if ply <= 0:
            return heuristic_func(state, asp), None
        value = math.inf
        move = None
        for action in asp.get_available_actions(state):
            min_value, _ = self.max_ab_cutoff(asp, asp.transition(state, action), alpha, beta, ply-1, heuristic_func)
            if min_value < value:
                value, move = min_value, action
                beta = min(beta, value)
            if value <= alpha:
                return value, move
        return value, move

class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if possibilities:
            return random.choice(possibilities)
        return "U"

    def cleanup(self):
        pass


class WallBot:
    """Hugs the wall"""

    def __init__(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def cleanup(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if not possibilities:
            return "U"
        decision = possibilities[0]
        for move in self.order:
            if move not in possibilities:
                continue
            next_loc = TronProblem.move(loc, move)
            if len(TronProblem.get_safe_actions(board, next_loc)) < 3:
                decision = move
                break
        return decision
