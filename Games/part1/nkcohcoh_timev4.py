import sys
import time
from heapq import *
# Global Variable
N, K = 3, 3
TABLE_STATE = "......w.."
TIME = 5
board = []
Depth_Limit = 10
max_depth = 0
player = 0

MAX_INFINITY = +999
MIN_INFINITY = -999

def board_to_table_state(board):
    st = []
    for r in range(N):
        for c in range(N):
            if board[r][c] == 1:
                st.append('w')
            elif board[r][c] == -1:
                st.append('b')
            else:
                st.append('.')
    print("New board:")
    print ("".join(st))

def table_state_to_board(table_state):
    global board
    board = [[0 for c in range(N)] for r in range(N)]
    indx = 0
    for r in range(N):
        for c in range(N):
            if TABLE_STATE[indx] == 'w':
                board[r][c] = 1
            elif TABLE_STATE[indx] == 'b':
                board[r][c] = -1
            indx += 1

def print_board(board):
    print ('########## Board #########')
    for r in range(N):
        for c in range(N):
            if board[r][c] == 1:
                print 'w ',
            elif board[r][c] == -1:
                print 'b ',
            else:
                print '. ',
        print ('')

def number_of_piece(board):
    count = 0
    for r in range(N):
        for c in range(N):
            if board[r][c] != 0:
                count += 1
    return count

def print_move(parent_board, best_board):
    for r in range(N):
        for c in range(N):
            if parent_board[r][c] != best_board[r][c]:
                print("Hmm, I'd recommend putting your marble at row {}, column {}".format(r, c))
#                board_to_table_state(best_board)
                print_board(best_board)
                return

def terminal_test(board):
    global player
    #print 'terminal_test(board):',board

    # player = -1 if total_piece % 2 == 0 else +1
    # check each row
    for r in range(N):
        count_self, count_opp = 0, 0
        for c in range(N):
            if board[r][c] == player:
                count_self += 1
                count_opp = 0
            elif board[r][c] == -player:
                count_opp += 1
                count_self = 0
            else:
                count_self, count_opp = 0, 0
                
            if count_self == K:
                return -1
            elif count_opp == K:
                return +1
    # check each column
    for c in range(N):
        count_self, count_opp = 0, 0
        for r in range(N):
            if board[r][c] == player:
                count_self += 1
                count_opp = 0
            elif board[r][c] == -player:
                count_opp += 1
                count_self = 0
            else:
                count_self, count_opp = 0, 0
                
            if count_self == K:
                return -1
            elif count_opp == K:
                return +1
    
    # check each diagonal
    col = 0
    for row in range(N):
        count_self_pd1, count_opp_pd1 = 0, 0
        count_self_pd2, count_opp_pd2 = 0, 0
        count_self_sd1, count_opp_sd1 = 0, 0
        count_self_sd2, count_opp_sd2 = 0, 0
        for r in range(0, N - row):
            if  board[row + r][col + r] == player :
                count_self_pd1 += 1
                count_opp_pd1 = 0
            elif board[row + r][col + r] == -player:
                count_opp_pd1 += 1
                count_self_pd1 = 0
            else:
                count_self_pd1, count_opp_pd1 = 0, 0
                 
            if board[col + r][row + r] == player:
                count_self_pd2 += 1
                count_opp_pd2 = 0
            elif board[col + r][row + r] == -player:
                count_opp_pd2 += 1
                count_self_pd2 = 0
            else:
                count_self_pd2, count_opp_pd2 = 0, 0
                
            if board[ (N - 1) - row - r ][r] == player:
                count_self_sd1 += 1
                count_opp_sd1 = 0
            elif board[ (N - 1) - row - r ][r] == -player:
                count_opp_sd1 += 1
                count_self_sd1 = 0
            else:
                count_self_sd1, count_opp_sd1 = 0, 0
                
            if board[ (N - 1) - r ][row + r] == player:
                count_self_sd2 += 1
                count_opp_sd2 = 0
            elif board[ (N - 1) - r ][row + r] == -player:
                count_opp_sd2 += 1
                count_self_sd2 = 0
            else:
                count_self_sd2, count_opp_sd2 = 0, 0
                
            if count_self_pd1 == K or count_self_pd2 == K or count_self_sd1 == K or count_self_sd2 == K:
                return -1
            elif count_opp_pd1 == K or count_opp_pd2 == K or count_opp_sd1 == K or count_opp_sd2 == K:
                return +1

    total_piece = number_of_piece(board)
    if total_piece == N * N:
        return 0
    
    return 404

def add_piece(board, row, col, player):
    return board[0:row] + [board[row][0:col] + [player, ] + board[row][col + 1:]] + board[row + 1:]

''' new modification here '''
def write_book(board,(row,col),book): # record changes in book
    pos = row*N+col
    book['h'][row].append(pos)
    book['v'][col].append(pos)
    book['up'][row+col].append(pos)
    book['down'][N-1-(row-col)].append(pos)

    return book

def init_player_and_book(board):
    book = {}
    book['h'] = [[] for i in range(N)] # marble lists for every row(for both colors[w:[],b:[]])
    book['v'] = [[] for i in range(N)] # marble lists for every col
    book['up'] = [[] for i in range(2*N-1)] # marble lists for slopes = 1
    book['down'] = [[] for i in range(2*N-1)] # marble lists for slopes = -1
    local_player = 0
    
    for r in range(N):
        local_player += sum(board[r])
    local_player = 1 if local_player == 0 else -1
    
    for r in range(N):
        for c in range(N):
            if board[r][c] != -local_player:
                book = write_book(board,(r,c),book)
    
    return local_player,book

def heuristic(board,(r,c),book):
    score = 0
    for key in book.keys():
        for i in range(len(book[key])): # i = which row/col/slope lines
            if len(book[key][i]) >= K:
                l = book[key][i]
                if r*N+c in l:
                    for pos in range(len(l)-1):

                        if key == 'h':
                            if l[pos] == l[pos+1] - 1:
                                score += 1  
                            else:
                                break
                        elif key == 'v':
                            if l[pos] == l[pos+1] - N:
                                score += 1 
                            else:
                                break
                        elif key == 'up':
                            if l[pos]/N == l[pos+1]/N - 1 and l[pos]%N == l[pos+1]%N + 1:
                                score += 1 
                            else:
                                break
                        else:
                            if l[pos]/N == l[pos+1]/N - 1 and l[pos]%N == l[pos+1]%N - 1:
                                score += 1 
                            else:
                                break
    return score

''' new modification ends '''
def successors(board):
    pq = []
    local_player,book = init_player_and_book(board)
    for (row,col) in [ (r, c) for r in range(0, N) for c in range(0, N) if board[r][c] == 0 ]:
        succ_board = add_piece(board,row,col,local_player)
        heappush(pq, (heuristic(board,(row,col),book),succ_board))

    return pq

''' 
-Book: 
-book['h'][(0,0)_indexed_as_first_row][player=1_or_2]
-book['v'][(0,0)_indexed_as_first_col][player=1_or_2]
-book['up'][(0,0)_indexed_as_first_upward_slope][player=1_or_2]
-book['down'][(n,0)_indexed_as_first_downward_slope][player=1_or_2]
-(x,y) belongs to book['h'][x], book['v'][y], book['up'][x+y], book['down'][n-1-(x-y)]
-'''
def minmax_decision(board):
    global player
    global max_depth
    global Depth_Limit
    fringe = [board]

    best_value = 0
    best_board = board
    
    
    player = 1 if number_of_piece(best_board) % 2 == 0 else -1
    start_time = time.time()
    while terminal_test(best_board) == 404 and len(fringe) > 0:
        num = number_of_piece(best_board)
        player = 1 if num % 2 == 0 else -1
        parent_board = fringe.pop()
        max_depth = 0
        best_value = MIN_INFINITY
        best_depth = -1


        successor = successors(parent_board)
        #print 'successors:',successor
        while len(successor) > 0:
            s = heappop(successor)[1]
            # if num < N*N/3:
            #     best_board = s
            # else:
                # print ("successors:")
                # print_board(s)
            temp = min_value(s, MIN_INFINITY, MAX_INFINITY, 0)
            # print (max_depth)
            if temp > best_value:
                best_depth = max_depth
                best_value = temp
                best_board = s
            elif temp == best_value and max_depth > best_depth :
                best_depth = max_depth
                best_value = temp
                best_board = s
        print(time.time() - start_time)
        print("Best Depth:{},Depth Limit:{}".format(best_depth, Depth_Limit))
        #print_move(parent_board, best_board)
        if (time.time() - start_time) < TIME and best_depth == Depth_Limit:
            Depth_Limit += 2
            fringe[:] = []
            best_board = parent_board
        else:
            fringe[:] = []
            print_move(parent_board, best_board)
            start_time = time.time()
    
        fringe.append(best_board)
        # print_move(parent_board,best_board)
        # print ('\n +++++++ Now Move ++++++')
        # print_board(best_board)
        # fringe.append(best_board)

def min_value(board, alpha, beta, depth):
    global max_depth
    global Depth_Limit
    val = terminal_test(board)
    # print("Min:{}->{}".format(val,board))
    # print_board(board)
    # print_board(board)
    # print ("depth:{}; val:{}".format(depth,val))
    if val != 404: 
        if depth > max_depth: max_depth = depth
        return val
    
    best_value = MAX_INFINITY
    depth += 1
    
    successor = successors(board)
    while len(successor) > 0:
        s = heappop(successor)[1]
        if depth == Depth_Limit:
            max_depth = depth
            return 0
        temp = max_value(s, alpha, beta, depth)
        if temp < best_value:
            best_value = temp
        if best_value <= alpha: return best_value
        beta = min(beta, best_value)
    
    # print ("Best depth:{}; val:{}".format(depth,best_value))
    return best_value

def max_value(board, alpha, beta, depth):
    global max_depth
    global Depth_Limit
    val = terminal_test(board)
    # print("Max:{}->{}".format(val,board))
    # print_board(board)
    # print_board(board)
    # print ("depth:{}; val:{}".format(depth,val))
    if val != 404: 
        if depth > max_depth: max_depth = depth
        return val
    
    best_value = MIN_INFINITY
    depth += 1
    
    successor = successors(board)
    while len(successor) > 0:
        s = heappop(successor)[1]
    
        if depth == Depth_Limit:
            max_depth = depth
            return 0
        temp = min_value(s, alpha, beta, depth)
        if temp > best_value:
            best_value = temp
        if best_value >= beta: return best_value
        alpha = max(alpha, best_value)
    
    # print ("Best depth:{}; val:{}".format(depth,best_value))
    return best_value


# Read Command Line
if len(sys.argv) == 5 :
    N = int(sys.argv[1])
    K = int(sys.argv[2])
    TABLE_STATE = sys.argv[3]
    TIME = float(sys.argv[4])

table_state_to_board(TABLE_STATE)
print_board(board)
board_to_table_state(board)
minmax_decision(board)
# print successors(board)
# t = heappop(successors(board))
# print t

