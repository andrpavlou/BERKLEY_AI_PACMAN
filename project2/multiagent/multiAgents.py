# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """



    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        
        "*** YOUR CODE HERE ***"
        pacman_pos = successorGameState.getPacmanPosition()
        ghost_pos = successorGameState.getGhostPositions()
        ghost_pos = ghost_pos[0]
        
        min_distance = float('inf')
        food = currentGameState.getFood()

        for foodcoords in food.asList():
            distance =  (manhattanDistance(foodcoords, pacman_pos))
            if distance < min_distance:
                min_distance = distance

            death_distance = manhattanDistance(ghost_pos, pacman_pos)
            if death_distance == 0:
                return float('-inf')

        return -min_distance



def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        score = self.minimax(0, 0, gameState)  
        return score[0]  

    def minimax(self, depth, agent, gameState):
        
        if agent == gameState.getNumAgents():
            agent = 0
            depth += 1

        if depth == self.depth:
            return 0, self.evaluationFunction(gameState)

        best_strat = [0, float('-inf')]        #Stores best score and best action
        
        for action in gameState.getLegalActions(agent):  
            # Max (pacman is agent 0)
            if agent == 0:  
                next_game_state = gameState.generateSuccessor(agent, action)
                state = self.minimax(depth, agent + 1, next_game_state)

            if agent == 0 and state[1] > best_strat[1]:                
                best_strat[0] = action
                best_strat[1] = state[1]

            # Min (ghost)
            if agent != 0:  
                next_game_state = gameState.generateSuccessor(agent, action)
                state = self.minimax(depth, agent + 1, next_game_state)

            if agent != 0 and (best_strat[1] == float('-inf') or state[1] < best_strat[1]):
                best_strat[0] = action
                best_strat[1] = state[1]

        # Leaf node
        if gameState.isWin() or gameState.isLose():
            return 0, self.evaluationFunction(gameState)
        
        return best_strat  
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        score = self.alphabeta(0, 0, gameState, float('-inf'), float('+inf'))  
        return score[0]  

    def alphabeta(self, depth, agent, gameState, alpha, beta):
        
        if agent == gameState.getNumAgents():
            agent = 0
            depth += 1

        if depth == self.depth:
            return 0, self.evaluationFunction(gameState)

        best_strat = [0, float('-inf')]        #Stores best score and best action
        
        for action in gameState.getLegalActions(agent):  
            # Max (pacman is agent 0)
            if agent == 0:  
                next_game_state = gameState.generateSuccessor(agent, action)
                state = self.alphabeta(depth, agent + 1, next_game_state, alpha, beta)

            if agent == 0 and state[1] > best_strat[1]:                
                best_strat[0] = action
                best_strat[1] = state[1]
            
            if agent == 0:
                alpha = max(alpha, state[1])   

            # Min (ghost)
            if agent != 0:  
                next_game_state = gameState.generateSuccessor(agent, action)
                state = self.alphabeta(depth, agent + 1, next_game_state, alpha, beta)

            if agent != 0 and (best_strat[1] == float('-inf') or state[1] < best_strat[1]):
                best_strat[0] = action
                best_strat[1] = state[1]

            if agent != 0:
                beta = min(beta, state[1])

            if alpha > beta:
                break

        # Leaf node
        if gameState.isWin() or gameState.isLose():
            return 0, self.evaluationFunction(gameState)
        
        return best_strat  


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        score = self.expectimax(0, 0, gameState)  
        return score[0]  

    def expectimax(self, depth, agent, gameState):
        
        if agent == gameState.getNumAgents():
            agent = 0
            depth += 1

        if depth == self.depth:
            return 0, self.evaluationFunction(gameState)

        best_strat = [0, float('-inf')]        #Stores best score and best action
        
        actions  = gameState.getLegalActions(agent)
        for action in actions:  
            # Max (pacman is agent 0)
            if agent == 0:  
                next_game_state = gameState.generateSuccessor(agent, action)
                state = self.expectimax(depth, agent + 1, next_game_state)

            if agent == 0 and state[1] > best_strat[1]:                
                best_strat[0] = action
                best_strat[1] = state[1]

            # Min (ghost)
            if len(actions) != 0:
                chance = 1 / len(actions)

            if agent != 0:  
                next_game_state = gameState.generateSuccessor(agent, action)
                state = self.expectimax(depth, agent + 1, next_game_state)

            if agent != 0 and (best_strat[1] == float('-inf')):
                best_strat[0] = action
                best_strat[1] = state[1] * chance
            elif agent != 0:
                best_strat[0] == action
                best_strat[1] += state[1] * chance

        # Leaf node
        if gameState.isWin() or gameState.isLose():
            return 0, self.evaluationFunction(gameState)
        
        return best_strat  
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
        
    "*** YOUR CODE HERE ***"
    pacman_pos = currentGameState.getPacmanPosition()
    ghost_pos = currentGameState.getGhostPositions()
    scared_time = currentGameState.getGhostStates()[0].scaredTimer
    ghost_pos = ghost_pos[0]
    
    min_distance = float('inf')
    food = currentGameState.getFood()

    for foodcoords in food.asList():
        distance =  (manhattanDistance(foodcoords, pacman_pos))
        if distance < min_distance:
            min_distance = distance

    death_distance = manhattanDistance(ghost_pos, pacman_pos)

    if death_distance == 0:
        return float('-inf')
    
    move = 0
    scared_pen = 0

    if death_distance > min_distance  or scared_time > 0:
        move = 100
    else:
        move = -100

    if scared_time > 0:
        scared_pen = -3 / death_distance

    if currentGameState.isWin():
        return float('inf')

    return currentGameState.getScore() - min_distance + move - scared_pen

# Abbreviation
better = betterEvaluationFunction
