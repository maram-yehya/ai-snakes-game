from environment import Player, GameState, GameAction, get_next_state
from utils import get_fitness
import numpy as np
from enum import Enum
from scipy.spatial.distance import cityblock
import time as tm

def heuristic(state: GameState, player_index: int) -> float:
    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """
    # Insert your code here...

    if not state.snakes[player_index].alive:
        return state.snakes[player_index].length
    pass
    optimistic = 0
    fruits = state.fruits_locations
    num_fruits=0
    #we calculate the dist between each fruit and the snakes's head 
    for fruit in fruits:
        dist = cityblock(state.snakes[player_index].head, fruit)
        optimistic += 1 / dist
        num_fruits += 1
    closest = np.inf
    for fruit in fruits:
        dist = cityblock(state.snakes[player_index].head, fruit)
        if dist < closest:
            closest = dist

    if cityblock(state.snakes[player_index].head, state.board_size) == 0:
        return state.snakes[player_index].length

    position = state.snakes[player_index].position
    # we want to get away from eating ourselves
    far = 0
    for tuple in position:
        dist = cityblock(state.snakes[player_index].head, tuple)
        far += dist
        
    opponents = sum(cityblock(state.snakes[player_index].head, s.head) for s in state.snakes if s.alive)
    if fruits.__len__()==0 :
        return state.snakes[player_index].length+far-opponents
    if opponents!=0:
        opponents=1/(opponents)
        
    
    
    #d|dis not 1| 48\47\47\46\45\
    return state.snakes[player_index].length + optimistic-opponents + far*(1/(7*num_fruits*num_fruits))
    #opp 1:40/47/49/28/52/50/4/35/41/31/37/16
    #return state.snakes[player_index].length + optimistic*opponents*1/closest 
    
class MinimaxAgent(Player):
    time=0
    """
    This class implements the Minimax algorithm.
    Since this algorithm needs the game to have defined turns, we will model these turns ourselves.
    Use 'TurnBasedGameState' to wrap the given state at the 'get_action' method.
    hint: use the 'agent_action' property to determine if it's the agents turn or the opponents' turn. You can pass
    'None' value (without quotes) to indicate that your agent haven't picked an action yet.
    """

    class Turn(Enum):
        AGENT_TURN = 'AGENT_TURN'
        OPPONENTS_TURN = 'OPPONENTS_TURN'

    class TurnBasedGameState:
        """
        This class is a wrapper class for a GameState. It holds the action of our agent as well, so we can model turns
        in the game (set agent_action=None to indicate that our agent has yet to pick an action).
        """

        def __init__(self, game_state: GameState, agent_action: GameAction):
            self.game_state = game_state
            self.agent_action = agent_action

        @property
        def turn(self):
            return MinimaxAgent.Turn.AGENT_TURN if self.agent_action is None else MinimaxAgent.Turn.OPPONENTS_TURN

    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        pass
     
        turnBasedGS = self.TurnBasedGameState(state, None)
        
        score = self.minimax(state, 2, turnBasedGS)
     
        return score[1]

    def minimax(self, state: GameState, depth: int, tb: TurnBasedGameState) -> [float, GameAction]:

     
        t = property(tb.turn).fget

        if depth == 0 or state.is_terminal_state:
            score = heuristic(state, self.player_index)
            return score, tb.agent_action

        if t == self.Turn.AGENT_TURN:
            best_score = -np.inf
            best_action = None
            

            for action in state.get_possible_actions(self.player_index):

                # the opponenet turn
                tb.__init__(state, action)
                
                score = self.minimax(state, depth, tb)
               

                if score[0] > best_score:
                    best_score = score[0]
                    best_action = action

          

            return best_score, best_action
        else:
            
            best_score = np.inf
            best_action = None
            agent_action = tb.agent_action
            for opponents_action in state.get_possible_actions_dicts_given_action(tb.agent_action, self.player_index):
                opponents_action[self.player_index] = agent_action

               
                next_state = get_next_state(state, opponents_action)

                # will be our agent turn
                tb.__init__(state, None)
                score = self.minimax(next_state, depth - 1, tb)

                if score[0] < best_score:
                    best_score = score[0]
                    best_action = opponents_action[self.player_index]
                   
           
            return best_score, best_action


class AlphaBetaAgent(MinimaxAgent):
    time=0
    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        pass
       
        turnBasedGS = self.TurnBasedGameState(state, None)
       
        score = self.alphaBeta(state, turnBasedGS, 2, -np.inf, np.inf)
        
        return score[1]

    def alphaBeta(self, state: GameState, tb: MinimaxAgent.TurnBasedGameState, depth: int, alpha: float, beta: float):

        t = property(tb.turn).fget

        if depth == 0 or state.is_terminal_state:
            score = heuristic(state, self.player_index)
            return score, tb.agent_action

        if t == self.Turn.AGENT_TURN:
            best_score = -np.inf
            best_action = None
            
              

            for action in state.get_possible_actions(self.player_index):

                # the opponenet turn
                tb.__init__(state, action)
                
                score = self.alphaBeta(state, tb, depth, alpha, beta)
               

                if score[0] > best_score:
                    best_score = score[0]
                    best_action = action
                alpha = max(best_score, alpha)
                if best_score >= beta:
                    return np.inf, None

            return best_score, best_action
        else:
           
            best_score = np.inf
            best_action = None
            agent_action = tb.agent_action
            for opponents_action in state.get_possible_actions_dicts_given_action(tb.agent_action, self.player_index):
                opponents_action[self.player_index] = agent_action

               
                next_state = get_next_state(state, opponents_action)

                # will be our agent turn
                tb.__init__(state, None)
                score = self.alphaBeta(next_state, tb, depth - 1, alpha, beta)

                if score[0] < best_score:
                    best_score = score[0]
                    best_action = opponents_action[self.player_index]
                    
                beta = min(beta, best_score)
                if best_score <= alpha:
                    return -np.inf, None
        
            return best_score, best_action


def SAHC_sideways():
    """
    Implement Steepest Ascent Hill Climbing with Sideways Steps Here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    pass

    actions = (GameAction.LEFT, GameAction.STRAIGHT, GameAction.RIGHT)
    initial_state = []
    i = 0
    # choosing random initial state
    while i < 50:
        initial_state.append(np.random.choice(actions))
        i += 1

    current = initial_state
    max = get_fitness(current)
    best_solutions = []
    index = 0
    size = 0
    limit = 50
    while index < 50:
        best_res = -np.inf
        best_solutions.clear()
        size = 0
        side_ways = 0
        for action in actions:
            # op(current):

            new = []
            for ac in current:
                new.append(ac)
            new[index] = action
            # new_val:
            res = get_fitness(new)
            if res > best_res:
                best_res = res
                best_solutions.clear()
                size = 1
                best_solutions.append(new)
            elif res == best_res:
                best_solutions.append(new)
                size += 1

        if best_res > max:
            # if there is many states with same solution we choose a random one:
            max = best_res
            arr = np.arange(size)
            i = np.random.choice(arr)
            current = best_solutions[i]
            side_ways = 0

        elif best_res == max and side_ways < limit:

            arr = np.arange(size)
            i = np.random.choice(arr)
            current = best_solutions[i]
            side_ways += 1
        index += 1
        res = get_fitness(current)

  

    print(current)


def local_search():
    """
    Implement your own local search algorithm here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state/states
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    pass
    k = 5
    new_beam = []
    ans=[]
    ans_value=-np.inf

    actions = (GameAction.LEFT, GameAction.STRAIGHT, GameAction.RIGHT)
    for index in np.arange(k):
        i = 0
        initial_state = []
        # choosing random initial state
        while i < 50:
            initial_state.append(np.random.choice(actions))
            i += 1
        
        new_beam.append(initial_state)
        value=get_fitness(initial_state)
        if ans_value < value :
            ans=initial_state
            ans_value=value
    improving_moves = []


    while index < 50:
        # beam is array of states
        beam = []

        for b in new_beam:
            new = []
           
            for ac in b:
                new.append(ac)

            beam.append(new)
        new_beam.clear()
        improving_moves.clear()
        segma = 0
        for s in beam:
            for action in actions:
                new = []
                for ac in s:
                    new.append(ac)
                new[index] = action
                old=get_fitness(s)
                n=get_fitness(new)
               
                delta_v = n - old
                if delta_v > 0:
                   
                    improving_moves.append([new, np.absolute(delta_v)])
                    segma += delta_v
                    if n>ans_value:
                       ans=new
                       ans_value=n
                        
               
        
                    


        if improving_moves.__len__() == 0:
            
            
            print(ans)
            return

        # create array-p
     
                
        p = []
        for move in improving_moves:
            p.append(move[1] / segma)

        for i in np.arange(k):
            arr = np.arange(improving_moves.__len__())
            i = np.random.choice(arr, 1, p)
            j= i[0]
            move=improving_moves.pop(j)
            m=move.pop(0)
            new_beam.append(m)
           
         
            move.insert(0,m)
            improving_moves.insert(j,move)

        index += 1
    print(ans)
    return 

class TournamentAgent(MinimaxAgent):

    def get_action(self, state: GameState) -> GameAction:
        pass
        turnBasedGS = self.TurnBasedGameState(state, None)
       
        score = self.alphaBetaT(state, turnBasedGS, 4, -np.inf, np.inf)
        
        return score[1]

    def alphaBetaT(self, state: GameState, tb: MinimaxAgent.TurnBasedGameState, depth: int, alpha: float, beta: float):

        t = property(tb.turn).fget

        if depth == 0 or state.is_terminal_state:
            score = heuristic(state, self.player_index)
            return score, tb.agent_action

        if t == self.Turn.AGENT_TURN:
            best_score = -np.inf
            best_action = None
            
              

            for action in state.get_possible_actions(self.player_index):

                # the opponenet turn
                tb.__init__(state, action)
                
                score = self.alphaBetaT(state, tb, depth, alpha, beta)
               

                if score[0] > best_score:
                    best_score = score[0]
                    best_action = action
                alpha = max(best_score, alpha)
                if best_score >= beta:
                    return np.inf, None

            return best_score, best_action
        else:
           
            best_score = np.inf
            best_action = None
            agent_action = tb.agent_action
            for opponents_action in state.get_possible_actions_dicts_given_action(tb.agent_action, self.player_index):
                opponents_action[self.player_index] = agent_action

               
                next_state = get_next_state(state, opponents_action)

                # will be our agent turn
                tb.__init__(state, None)
                score = self.alphaBetaT(next_state, tb, depth - 1, alpha, beta)

                if score[0] < best_score:
                    best_score = score[0]
                    best_action = opponents_action[self.player_index]
                    
                beta = min(beta, best_score)
                if best_score <= alpha:
                    return -np.inf, None
        
            return best_score, best_action
        
        
        


if __name__ == '__main__':
   SAHC_sideways()
   local_search()

