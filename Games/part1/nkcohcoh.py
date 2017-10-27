import sys
import time

# Global Variable
N, K = 3, 3
TABLE_STATE = "........."
TIME = 5
board = []
Depth_Limit = 1
max_depth = 0
player = 0
start_time = 0
RETURN_TIME = 0.1

MAX_INFINITY = +999
MIN_INFINITY = -999

# # The following 2 functions are David Crandell's given code from Assignment0. I just reuse it customedly.
# # Code start here
def add_piece(board, row, col, player):
    return board[0:row] + [board[row][0:col] + [player, ] + board[row][col + 1:]] + board[row + 1:]
def successors(board):
    current_player = 1 if number_of_piece(board) % 2 == 0 else -1
    return [ add_piece(board, r, c, current_player) for r in range(0, N) for c in range(0, N) if board[r][c] == 0 ]
# # Code end here

# Convert State(string) to Board(2D List)
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
    return ("".join(st))

# Convert Board(2D List) to State(string)
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

# Print the 2D Board list
def print_board(board):
    print ('Board')
    for r in range(N):
        for c in range(N):
            if board[r][c] == 1:
                print ('w '),
            elif board[r][c] == -1:
                print ('b '),
            else:
                print ('. '),
        print ('')

# Sum of the number of pieces on the board
def number_of_piece(board):
    count = 0
    for r in range(N):
        for c in range(N):
            if board[r][c] != 0:
                count += 1
    return count

# Print the chosen move
def print_move(parent_board, best_board):
    for r in range(N):
        for c in range(N):
            if parent_board[r][c] != best_board[r][c]:
                print("Hmm, I'd recommend putting your marble at row {}, column {}.".format(r + 1, c + 1))
                st = board_to_table_state(best_board)
                print("New board:\n" + st)
                return st
    print("Hmm, I'd recommend no move. The game is over.")
    st = board_to_table_state(best_board)
    print(st)
    return st

# Check whether the board is in Goal State or not 
def terminal_test(board):
    global player

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

# MinMax Algorithm
def minmax_decision(board):
    global start_time
    global player
    global max_depth
    global Depth_Limit
    
    fringe = [board]
    best_value = 0
    parent_board = board
    best_board = board
    player = 1 if number_of_piece(best_board) % 2 == 0 else -1
    print("Thinking! Please wait...\n")
    start_time = time.time()
    
    while not is_time_finish() and terminal_test(best_board) == 404:
        player = 1 if number_of_piece(best_board) % 2 == 0 else -1
        parent_board = fringe.pop()
        max_depth = 0
        best_value = MIN_INFINITY
        best_depth = -1

        for s in successors(parent_board):
            temp = min_value(s, MIN_INFINITY, MAX_INFINITY, 0)
            if temp > best_value:
                best_depth = max_depth
                best_value = temp
                best_board = s
            elif temp == best_value and max_depth > best_depth :
                best_depth = max_depth
                best_value = temp
                best_board = s
                
        if not is_time_finish() and best_depth == Depth_Limit:
            Depth_Limit = Depth_Limit * 2
            fringe[:] = []
            best_board = parent_board
        else:
            fringe[:] = []
            break
            
        fringe.append(best_board)
    
    return print_move(parent_board, best_board)

# Min function
def min_value(board, alpha, beta, depth):
    global max_depth
    global Depth_Limit
    
    val = terminal_test(board)
    if val != 404: 
        if depth > max_depth: max_depth = depth
        return val
    
    best_value = MAX_INFINITY
    depth += 1
    
    for s in successors(board):
        if depth == Depth_Limit:
            max_depth = depth
            return 0
        temp = max_value(s, alpha, beta, depth)
        if temp < best_value:
            best_value = temp
        if best_value <= alpha: return best_value
        beta = min(beta, best_value)
        
        if is_time_finish():
            return best_value if best_value != MAX_INFINITY else 0
    
    return best_value

# Max function
def max_value(board, alpha, beta, depth):
    
    global max_depth
    global Depth_Limit
    
    val = terminal_test(board)
    if val != 404: 
        if depth > max_depth: max_depth = depth
        return val
    
    best_value = MIN_INFINITY
    depth += 1
    
    for s in successors(board):
        if depth == Depth_Limit:
            max_depth = depth
            return 0
        temp = min_value(s, alpha, beta, depth)
        if temp > best_value:
            best_value = temp
        if best_value >= beta: return best_value
        alpha = max(alpha, best_value)
        
        if is_time_finish():
            return best_value if best_value != MIN_INFINITY else 0

    return best_value

# Check if Time is finished or not
def is_time_finish():
    global start_time
    return (time.time() - start_time) >= (TIME - RETURN_TIME)

# For call in a Tournament
def run_player(n, k, Table_State, Time):
    global N
    global K
    global TABLE_STATE
    global TIME
    N, K, TABLE_STATE, TIME = n, k, Table_State, Time
    table_state_to_board(TABLE_STATE)
    return minmax_decision(board)

def main():
    global N
    global K
    global TABLE_STATE
    global TIME
    # Read Command Line 
    if len(sys.argv) == 5 :
        N = int(sys.argv[1])
        K = int(sys.argv[2])
        TABLE_STATE = sys.argv[3]
        TIME = float(sys.argv[4])
    table_state_to_board(TABLE_STATE)
#     print_board(board)
    board_to_table_state(board)
    minmax_decision(board)

if __name__ == "__main__": main()
