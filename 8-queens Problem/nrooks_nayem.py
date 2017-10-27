# nrooks.py : Solve the N-Rooks problem!
# D. Crandall, August 2016
#
# The N-rooks problem is: Given an empty NxN chessboard, place N rooks on the board so that no rooks
# can take any other, i.e. such that no two rooks share the same row or column.

# This is N, the size of the board.
# [Nayem - in 1min highet N: Nrooks=115, NQueens=11]
N=11

# [Nayem - Que5 for Nqueen] 
# Count # of pieces diagonally in the whole board
def isvalid_position(board, row, col):
    # check primary diagonal
    r, c = row-min(row,col), col-min(row,col)
    while r<N and c<N:
        if board[r][c] != 0:
            return False  
        r +=1
        c +=1
    # check secondary diagonal    
    if (row+col) < N-1:
        r,c = (row+col), 0
    else:
        r, c = N-1, (row+col)-(N-1)
        
    while r>=0 and c<N:
        if board[r][c] != 0:
            return False 
        r -=1
        c +=1
    return True

# Count # of pieces in given row
def count_on_row(board, row):
    return sum( board[row] ) 

# Count # of pieces in given column
def count_on_col(board, col):
    return sum( [ row[col] for row in board ] ) 

# Count total # of pieces on board
def count_pieces(board):
    return sum([ sum(row) for row in board ] )

# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    return "\n".join([ " ".join([ "Q" if col else "_" for col in row ]) for row in board])

# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, row, col):
    return board[0:row] + [board[row][0:col] + [1,] + board[row][col+1:]] + board[row+1:]

# Get list of successors of given board state
def successors(board):
    return [ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) ]

# [Nayem -> Que3 for N-rooks]
def successors2(board):
    if count_pieces(board) == N:
        return []
    return [ add_piece(board, r, c) for r in range(N) for c in range(N) if(board[r][c]==0)]

# [Nayem -> Que4 for N-rooks]
def successors3(board):   
    return [ add_piece(board, r, c) for r in range(N) if (count_on_row(board, r)<1) for c in range(N) if(count_on_col(board, c)<1)]

# Nayem -> Que5(N-queens)
def successorsQueen(board):
    return [ add_piece(board, r, c) for r in range(N) if(count_on_row(board, r)<1) for c in range(N) if(count_on_col(board, c)<1  and isvalid_position(board,r,c) )]

# [Nayem -> Que4 for N-rooks]
# To check if board is a goal state
def is_goal3(board):
    return count_pieces(board) == N

# Nayem -> Que4 for N-rooks
# Solve n-rooks!
def solve3(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successors3( fringe.pop() ):
#             print ("Succesor:\n" + printable_board(s) + "\n")
            if is_goal3(s):
                return(s)
            fringe.append(s)    # stack(DFS)
    return False

# Nayem -> Que5 for N-queens
# Solve n-queens!
def solveQueen(initial_board):
    solutionNrooks = solve3(initial_board) # To run N-rooks
    print ("N-rooks:\n"+printable_board(solutionNrooks) if solutionNrooks else "Sorry, no solution for N-rooks found. :(")
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successorsQueen( fringe.pop() ):
#             print ("Succesor:\n" + printable_board(s) + "\n")
            if is_goal3(s):
                return(s)
            fringe.append(s)
    return False

# check if board is a goal state
def is_goal(board):
    return count_pieces(board) == N and \
        all( [ count_on_row(board, r) <= 1 for r in range(0, N) ] ) and \
        all( [ count_on_col(board, c) <= 1 for c in range(0, N) ] )

# Solve n-rooks!
def solve(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successors2( fringe.pop() ):
            if is_goal(s):
                return(s)
            fringe.append(s)    # stack(DFS)
#             fringe.insert(0,s)  # queue(BFS)
    return False

# The board is stored as a list-of-lists. Each inner list is a row of the board.
# A zero in a given square indicates no piece, and a 1 indicates a piece.
initial_board = [[0]*N]*N
print ("Starting from initial board:\n" + printable_board(initial_board) + "\n\nLooking for solution...\n")
solution = solveQueen(initial_board)
print ("N-queens:\n"+printable_board(solution) if solution else "Sorry, no solution for N-Queens found. :(")
