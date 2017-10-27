# this is a backup file for B551 Assignment 1

# row_num = x  col_num = y
# for any position[x,y] in 4X4 matrix, is equivalent to 4(x-1)+y-1 th element in a 16-list
# for example, [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,0,15,14]] here position of 0 is in 4th row and 2nd col
# if stored in a list: [1,2,3,4,5,6,7,8,9,10,11,12,13,0,15,14], position of 0 is 4*(4-1)+2-1
# so by dividing position of a number in the list with 4, remainder - 1 will be actual col, dividend int will be
# actual row

########  Basic Functions ########

def findMoves(board):  # This is actually the successor function of 15-puzzle
    zero_i = [i for i,j in enumerate(board) if j == 0][0]
    row = zero_i//4
    col = zero_i%4
    moves = []
    
    if row > 0:
        moves.append('U')
    if row < 3:
        moves.append('D')
    if col > 0:
        moves.append('L')
    if col < 3:
        moves.append('R')
    if row == 0:
        moves.append('DD')
    elif row == 3:
        moves.append('UU')
    if col == 0:
        moves.append('RR')
    elif col == 3:
        moves.append('LL')
    
    return moves

def moveBoard(board,direction):
    # if movement of empty tile is REVERSE of other numbered tiles,i.e. actual movements
    newBoard = board[:]
    zero_i = [i for i,j in enumerate(board) if j == 0][0]
    row = zero_i//4
    col = zero_i%4
    
    if not onEdge(board,zero_i):
        
        if direction == 'U':
            row += 1
        elif direction == 'D':
            row -= 1
        elif direction == 'L':
            col += 1
        elif direction == 'R':
            col -= 1
    
    else:  # define moves of opposite side slidding into opening
        
        if (direction == 'U' and row == 3) or (direction == 'D' and row == 0):
            row = 3 - row
        elif (direction == 'L' and col == 3) or (direction == 'R' and col == 0):
            col = 3 - col
        else:
            if direction == 'U':
                row += 1
            elif direction == 'D':
                row -= 1
            elif direction == 'L':
                col += 1
            elif direction == 'R':
                col -= 1
    
    new_i = row * 4 + col
    newBoard[zero_i],newBoard[new_i] = newBoard[new_i],newBoard[zero_i]
    return newBoard

def successorBoard(board):
    newBoards = []
    for i in ['L','R','U','D']:
        newBoards.append((moveBoard(board,i),i))
            
    return newBoards

def isGoal(board):
    goal = range(1,16)
    goal.append(0)
    if board == goal:
        return True
    else:
        return False
    
def isSolvable(board):
    count = 0
    zero_i = [i for i,j in enumerate(board) if j == 0][0]
    
    for i in range(len(board)):
        if i != zero_i:
            count_i = 0
            for j in range(i+1,len(board)):
                if j != zero_i:
                    if board[j] < board[i]:
                        count_i += 1
            count += count_i
    # goal state is 3
    if (count + (zero_i//4)) % 2 == 1:
        return True
    else:
        return False

####### stateKeeper Class and its Heuristic Functions #######
from heapq import *

class stateKeeper():
    def __init__(self,array):
        self.board = array
        self.currentCost = 0
        self.moves = [] # move direction from its prior state
        
    def hMisplaced(self):
        count = 0
        board = self.board
    
        for i in range(len(board)):

            if i != len(board)-1:
                target = i + 1
            else:
                target = 0

            if board[i] != target:
                count += 1

        return count
    
    def hHummingDistance(self):
        distance = 0
        board = self.board

        for i in range(len(board)):
            row = i // 4
            col = i % 4

            if board[i] != 0:
                row_g = (board[i]-1) // 4
                col_g = (board[i]-1) % 4
            else:
                row_g = 15 // 4
                col_g = 15 % 4

            if row in [0,3] and col in [0,3]:
                distance_i = min(abs(row-row_g),abs(4-row-row_g)) + min(abs(col-col_g),abs(4-col-col_g))
            elif row in [0,3]:
                distance_i = min(abs(row-row_g),abs(4-row-row_g)) + abs(col-col_g)
            elif col in [0,3]:
                distance_i = abs(row-row_g) + min(abs(col-col_g),abs(4-col-col_g))
            else:
                distance_i = abs(row-row_g) + abs(col-col_g)
            #distance_i = abs(row-row_g) + abs(col-col_g)
            #print i,distance_i
            distance += distance_i

        return distance
    
    def hReverse(self):
        board = self.board
        count = 0

        for i in range(len(board)):
            for j in range(i+1,len(board)):
                if board[i] > board[j] and board[i] != 0 and board[j] != 0:
                    count += 1


        return count

######## Main Function #######

def main(pq,h = 'hMisplaced',tried = 0):

    while len(pq) > 0:
        tried += 1
        currentState = heappop(pq)[1]
        
        if isGoal(currentState.board):
            print '# of states tried: ', tried
            return currentState.moves
        else:
            for (board,move) in successorBoard(currentState.board):
                state = stateKeeper(board)
                state.moves = currentState.moves[:]
                state.moves.append(move)
                
                # some heuristic functions here
                # then store the successor and its movement having lowest cost = h(x) + g(x)
                state.currentCost = currentState.currentCost + 1

                if str(state.board) not in visited:
                    visited[str(state.board)] = True

                    if h == 'hMisplaced':
                        heappush(pq,(state.currentCost + state.hMisplaced(),state))
                    elif h == 'hHummingDistance':
                        heappush(pq,(state.currentCost + state.hHummingDistance(),state))

    print "No Solution Found"
    return

def main2(pq):
    tried = 0
    while len(pq) > 0:
        tried += 1
        currentState = heappop(pq)[2]
        
        if isGoal(currentState.board):
            print '# of states tried: ', tried
            return currentState.moves
        else:
            for (board,move) in successorBoard(currentState.board):
                state = stateKeeper(board)
                state.moves = currentState.moves[:]
                state.moves.append(move)
                
                # some heuristic functions here
                # then store the successor and its movement having lowest cost = h(x) + g(x)
                state.currentCost = currentState.currentCost + 1

                if str(state.board) not in visited:
                    visited[str(state.board)] = True

                    heappush(pq,(state.currentCost + state.hManhattanDistance(),state.currentCost + state.hReverse(),state))

    print "No Solution Found"
    return

def main3(pq):
    tried = 0
    while len(pq) > 0:
        tried += 1
        #print pq[0]
        currentState = heappop(pq)[1]
        
        if isGoal(currentState.board):
            print '# of states tried: ', tried
            return currentState.moves
        else:
            for (board,move) in successorBoard(currentState.board):
                state = stateKeeper(board)
                state.moves = currentState.moves[:]
                state.moves.append(move)
                
                # some heuristic functions here
                # then store the successor and its movement having lowest cost = h(x) + g(x)
                state.currentCost = currentState.currentCost + 1

                if str(state.board) not in visited:
                    visited[str(state.board)] = True

                    heappush(pq,(state.currentCost + (state.hManhattanDistance() + state.hReverse())/2.0,state))

    print "No Solution Found"
    return
        
######## Helper Functions ##########
def onEdge(board,zero_i):
    # judge if empty tile is on edge of the board
    
    if zero_i//4 == 0 or zero_i//4 == 3: # if 0 is on first row or last row
        return True
    elif zero_i%4 == 0 or zero_i%4 == 3: # if 0 is on first col or last col
        return True
    else:
        return False

def drawBoard(board): 
    # just help visualize board in a matrix style
    if type(board) == type([]):
        return '\n'.join([str(board[(0+4*i):(4+4*i)]) for i in range(4)])
    else:
        return board

####### Run the Program #########
import sys
arguments = sys.argv
filename = arguments[1]

f = open(filename,'r')
initial = []
for i in f:
    for j in (str.split(i)):
        initial.append(int(j))
    
initial = stateKeeper(initial)

from datetime import datetime

if not isSolvable(initial.board):
    print 'Board Not Solvable :('
else:
    
    print 'Initial Board:'
    print drawBoard(initial.board)
    print
    
    # print '-------- ' + 'hReverse' + ' --------'

    # startTime = datetime.now()

    # pq = [] # a priority queue storing all states
    # visited = {} # a hashmap storing states that are visited

    # heappush(pq,(initial.currentCost,initial)) #items in pq:[g(x),board,move]
    # visited[str(initial.board)] = True

    # result2 = main(pq,'hReverse')
    # board = initial.board[:]
    
    # print 'Total time used: '
    # print datetime.now() - startTime
    # print result2

    # print '-------- ' + '(hManhattanDistance+hReverse)/2' + ' --------'

    # startTime = datetime.now()

    # pq = [] # a priority queue storing all states
    # visited = {} # a hashmap storing states that are visited

    # heappush(pq,(initial.currentCost,initial)) #items in pq:[g(x),board,move]
    # visited[str(initial.board)] = True

    # result2 = main3(pq,'hReverse')
    # board = initial.board[:]
    
    # print 'Total time used: '
    # print datetime.now() - startTime
    # print result2

    method = 'hMisplaced'
    print '-------- ' + 'hManhattanDistance,hReverse' + ' --------'

    startTime = datetime.now()

    pq = [] # a priority queue storing all states
    visited = {} # a hashmap storing states that are visited

    heappush(pq,(initial.currentCost,initial.currentCost,initial)) #items in pq:[g(x),board,move]
    visited[str(initial.board)] = True

    result2 = main2(pq)
    board = initial.board[:]
    
    print 'Total time used: '
    print datetime.now() - startTime
    print result2
