# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    from util import Stack

    stack = Stack()
    visited = []
    path = []

    #Empty list which will soon store the path.
    stack.push((problem.getStartState(), []))

    
    while(not stack.isEmpty()):
        coords, path = stack.pop()

        #Return the final path if the last coordinates are the goal.
        if problem.isGoalState(coords):
            return path

        if coords not in visited:
            visited.append(coords)
            for i in problem.getSuccessors(coords):
                #Adds to the end of the list, path the last action.
                stack.push((i[0], path + [i[1]]))
    return path

#Same proccess as as DFS but instead it uses a queue.
def breadthFirstSearch(problem: SearchProblem):
    from util import Queue

    queue = Queue()
    visited = []
    path = []

    #Empty list which will soon store the path.
    queue.push((problem.getStartState(), []))

    while(not queue.isEmpty()):
        coords, path = queue.pop()

        if problem.isGoalState(coords):
            return path

        if coords not in visited:
            visited.append(coords)
            for i in problem.getSuccessors(coords):
                #Adds to the end of the list, path the last action.
                queue.push((i[0], path + [i[1]]))
    return path

#Same proccess as as DFS but instead it uses a priority queue, based on the cost of actions.
def uniformCostSearch(problem: SearchProblem):
    from util import PriorityQueue

    pqueue = PriorityQueue()
    visited = []
    path = []

    pqueue.push((problem.getStartState(), []), 0)
    

    while(not pqueue.isEmpty()):
        #Pops the element with the highest priority.
        coords, path = pqueue.pop()

        if problem.isGoalState(coords):
            return path

        if coords not in visited:
            visited.append(coords)
            for i in problem.getSuccessors(coords):
                #Adds to the end of the list, path the last action, and its priority based on the cost of actions
                #of the old path + the cost of the successor's cost.
                pqueue.push((i[0], path + [i[1]]), problem.getCostOfActions(path) + i[2])
    return path

#Trivial heuristic.
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

#Same proccess as as uniformCostSearch with the priority being cost of actions of the  old path + cost of current successor's +
# its heuristic value.
def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    from util import PriorityQueue

    pqueue = PriorityQueue()
    visited = []
    path = []

    pqueue.push((problem.getStartState(), []), 0)
    

    while(not pqueue.isEmpty()):
        coords, path = pqueue.pop()

        if problem.isGoalState(coords):
            return path

        if coords not in visited:
            visited.append(coords)
            for i in problem.getSuccessors(coords):
                #New priority: cost of the old path + cost of current successor + its heuristic 
                pqueue.push((i[0], path + [i[1]]), problem.getCostOfActions(path) + i[2] + heuristic(i[0], problem))
    return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
