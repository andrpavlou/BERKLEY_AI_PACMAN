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
        
        min_distance = float('inf') #Stores the minimum distance between ghost/pacman (manhattan distance).
        food = currentGameState.getFood() #Current food grid. (True when there is food in a certain position)

        #Find minimum distance between pacman and food dots.
        for foodcoords in food.asList():
            distance =  (manhattanDistance(foodcoords, pacman_pos))
            if distance < min_distance:
                min_distance = distance
        
        #Distance between ghost/pacman (manhattan distance).
        death_distance = manhattanDistance(ghost_pos, pacman_pos)
        #If manhattan distance is zero it means it is a lose state, so return -inf.
        if death_distance == 0:
            return float('-inf')

        return successorGameState.getScore() - min_distance



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
        self.index = 0 #Pacman is always agent index 0.
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        "*** YOUR CODE HERE ***"
        score = self.minimax(0, 0, gameState)  
        return score[0]  

    #Minimax function taking into consideration the depth/agent/gamestate (agent zero is pacman, else ghosts).
    def minimax(self, depth, agent, gameState):

        #Means iterated all agents so increase depth, and go back to agent 0.
        if agent == gameState.getNumAgents():
            agent = 0
            depth += 1

        #Leaf node, so return its value. 
        if depth == self.depth:
            return 0, scoreEvaluationFunction(gameState)

        best_strat = [0, float('-inf')] #Stores best score and best move.
        
        #Iterate through all the legal actions of the current agent.
        for action in gameState.getLegalActions(agent):  
            #Pacman (max).
            if agent == 0:  
                #Find next state, call recursively minimax, for next agent until leaf node. 
                next_game_state = gameState.generateSuccessor(agent, action)
                state = self.minimax(depth, agent + 1, next_game_state)

            #Find the highest scores, and save the action that caused it.
            if agent == 0 and state[1] > best_strat[1]:                
                best_strat[0] = action
                best_strat[1] = state[1]

            #Ghosts (min).
            if agent != 0:  
                next_game_state = gameState.generateSuccessor(agent, action)
                state = self.minimax(depth, agent + 1, next_game_state)

            #Find the lowest scores, and save the action that caused it or if the best score is unchaned for initilization just save it.
            if agent != 0 and (best_strat[1] == float('-inf') or state[1] < best_strat[1]):
                best_strat[0] = action
                best_strat[1] = state[1]

        #Return the score if pacman loses or wins.
        if gameState.isWin() or gameState.isLose():
            return 0, scoreEvaluationFunction(gameState)
        
        return best_strat  
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        "*** YOUR CODE HERE ***"
        score = self.alphabeta(0, 0, gameState, float('-inf'), float('+inf'))  
        return score[0]  

    #Same as minimax with addition of alpha, beta variables.
    def alphabeta(self, depth, agent, gameState, alpha, beta):
        
        #Means iterated all agents so increase depth, and go back to agent 0.
        if agent == gameState.getNumAgents():
            agent = 0
            depth += 1

        #Leaf node, so return its value. 
        if depth == self.depth:
            return 0, scoreEvaluationFunction(gameState)

        best_strat = [0, float('-inf')] #Stores best score and best action.
        
        for action in gameState.getLegalActions(agent):  
            #Alphabeta algorithm's prune idea. If alpha is greater than beta just prune the tree by stopping.
            if alpha > beta:
                break

            #Pacman.
            if agent == 0:  
                #Find next state, call recursively minimax, for next agent until leaf node. 
                next_game_state = gameState.generateSuccessor(agent, action)
                state = self.alphabeta(depth, agent + 1, next_game_state, alpha, beta)
                alpha = max(alpha, state[1]) #Save the max value between alpha and current score (state[1]).


            #Find the highest scores, and save the action that caused it.
            if agent == 0 and state[1] > best_strat[1]:                
                best_strat[0] = action
                best_strat[1] = state[1]

            # Ghost.
            if agent != 0:  
                next_game_state = gameState.generateSuccessor(agent, action)
                state = self.alphabeta(depth, agent + 1, next_game_state, alpha, beta)
                beta = min(beta, state[1]) #Save the min value between beta and current score (state[1]).


            if agent != 0 and (best_strat[1] == float('-inf') or state[1] < best_strat[1]):
                best_strat[0] = action
                best_strat[1] = state[1]


        #Return the score if pacman loses or wins.
        if gameState.isWin() or gameState.isLose():
            return 0, scoreEvaluationFunction(gameState)
        
        return best_strat  


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        "*** YOUR CODE HERE ***"
        score = self.expectimax(0, 0, gameState)  
        return score[0]  
    
    #Same as minimax with taking into consideration the probabilities needed for min.
    def expectimax(self, depth, agent, gameState):
        #Means iterated all agents so increase depth, and go back to agent 0.
        if agent == gameState.getNumAgents():
            agent = 0
            depth += 1

        #Leaf node, so return its value. 
        if depth == self.depth:
            return 0, self.evaluationFunction(gameState)

        best_strat = [0, 0] #Stores best score and best action.
        
        actions  = gameState.getLegalActions(agent)
        for action in actions:  
            #Pacman (max).
            if agent == 0:  
                next_game_state = gameState.generateSuccessor(agent, action)
                state = self.expectimax(depth, agent + 1, next_game_state)

            #same as minimax, for max.
            if agent == 0 and state[1] > best_strat[1]:                
                best_strat[0] = action
                best_strat[1] = state[1]

            #Calculate the probilities, considering how many actions are there for the agent.
            if len(actions) != 0:
                chance = 1 / len(actions)

            #For ghosts the value that needs to be stored is the addition of old best value with 
            #the current value, multiplied with its probability.
            if agent != 0:  
                next_game_state = gameState.generateSuccessor(agent, action)
                state = self.expectimax(depth, agent + 1, next_game_state)
                best_strat[0] = action
                best_strat[1] += state[1] * chance


        #Return the score if pacman loses or wins.
        if gameState.isWin() or gameState.isLose():
            return 0, self.evaluationFunction(gameState)
        
        return best_strat  
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    """
        
    "*** YOUR CODE HERE ***"
    #Similar idea with Q1, but with current state and some additions.
    pacman_pos = currentGameState.getPacmanPosition()
    scared_time = currentGameState.getGhostStates()[0].scaredTimer
    ghost_pos = currentGameState.getGhostPositions()
    ghost_pos = ghost_pos[0]
    
    min_distance = float('inf')
    food = currentGameState.getFood()

    #Find minimum manhattan distance between all food dots and pacman.
    for foodcoords in food.asList():
        distance =  (manhattanDistance(foodcoords, pacman_pos))
        if distance < min_distance:
            min_distance = distance

    #Manhattan distance between current pacman position and ghost.
    death_distance = manhattanDistance(ghost_pos, pacman_pos)
    
    #Returns -inf if, the death distance is zero which means it is a lose state.
    if death_distance == 0:
        return float('-inf')
    #Return inf, if it is a win state.
    if currentGameState.isWin():
        return float('inf')
    
    move = 0
    scared_boost = 0

    #If minimum MD between pacman and food is less than MD between pacman 
    #and ghost, or the pacman consumed capsules and ghosts are scared,
    #pacman can keep moving.
    if death_distance > min_distance  or scared_time > 0:
        move = 100
    #Otherwise receive a penalty because ghost is close.
    else:
        move = -100
    
    #Ghost are scared, so add an aditional boost to move around. No need to check for dividing with zero
    #because of former edge case.
    if scared_time > 0:
        scared_boost = 3 / death_distance #(3 can be changed to other numbers as well, but 3 works better).   

    #Taking into consideration the score of current game state with the addition of the extra calculated variables.
    return currentGameState.getScore() - min_distance + move + scared_boost

# Abbreviation
better = betterEvaluationFunction
