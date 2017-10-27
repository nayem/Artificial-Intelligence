import sys
import time
import os
from nkcohcoh import run_player

# Global Variable
N, K = 4, 2
# TABLE_STATE = ".b.w....."
TABLE_STATE = "................"
TIME = 500

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

def table_state_to_board(TABLE_STATE):
    board = [[0 for c in range(N)] for r in range(N)]
    indx = 0
    for r in range(N):
        for c in range(N):
            if TABLE_STATE[indx] == 'w':
                board[r][c] = 1
            elif TABLE_STATE[indx] == 'b':
                board[r][c] = -1
            indx += 1
    return board
    
def print_board(board):
    print ('Board:')
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

def is_tournament_end(TABLE_STATE):
    global N
    global K
    board = table_state_to_board(TABLE_STATE)
    player = 1
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
                print ("Black Wins!")
                return True
            elif count_opp == K:
                print ("White Wins!")
                return True
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
                print ("Black Wins!")
                return True
            elif count_opp == K:
                print ("White Wins!")
                return True
    
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
                print ("Black Wins!")
                return True
            elif count_opp_pd1 == K or count_opp_pd2 == K or count_opp_sd1 == K or count_opp_sd2 == K:
                print ("White Wins!")
                return True

    total_piece = number_of_piece(board)
    if total_piece == N * N:
        print ("Draw!")
        return True
    
    return False

def tournament(N, K, TABLE_STATE, TIME):
    num = 0
    print "[" + str(num) + "]",
    num += 1
    print_board(table_state_to_board(TABLE_STATE))
    while not is_tournament_end(TABLE_STATE):
        start_time = time.time()
#         TABLE_STATE = os.system("nkcohcoh_single_playerv1.py" + " " + str(N) + " " + str(K) + " " + str(TIME))
        TABLE_STATE = run_player(N, K, TABLE_STATE, 5)
        
        print "[" + str(num) + "]",
        num += 1
        print_board(table_state_to_board(TABLE_STATE))
        prev_player = "White" if number_of_piece(table_state_to_board(TABLE_STATE)) % 2 != 0 else "Black"
        print("{} makes this move in {} seconds".format(prev_player, time.time() - start_time))
        
        if not is_tournament_end(TABLE_STATE):
            start_time = time.time()
#             TABLE_STATE = os.system("nkcohcoh_single_playerv1.py" + " " + str(N) + " " + str(K) + " " + str(TIME))
            TABLE_STATE = run_player(N, K, TABLE_STATE, TIME)
            
            print "[" + str(num) + "]",
            num += 1
            print_board(table_state_to_board(TABLE_STATE))
            prev_player = "White" if number_of_piece(table_state_to_board(TABLE_STATE)) % 2 != 0 else "Black"
            print("{} makes this move in {} seconds".format(prev_player, time.time() - start_time))

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
    tournament(N, K, TABLE_STATE, TIME)

if __name__ == "__main__": main()
