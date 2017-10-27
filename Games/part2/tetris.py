# Simple tetris program! v0.1
# D. Crandall, Sept 2016

# Formulation of the search problem:
# We considered the search problem as a search over all possible locations and rotations of a piece. We did a slight simplification by disallowing the game to maneuver a piece into a complicated
# position through successive left and right commands. We could have added this, but we decided it provided little utility for the time taken to implement (plus we cannot use that for the simple version).

# The search algorithm uses a successor function to generate all possible states given the piece in play. Then the algorithm scores these pieces using a set of heuristics and places the piece in the
# location/rotation that maximizes the score. To generate the score of a piece we found that it is necessary to consider multiple different objects. For instance, we want to maximize the number of
# completed lines after we drop a piece, we want to maximize the width of our board, we want to minimize the sum of heights and the largest height, we want to minimize the number of holes and we want to minimize
# the variability of our column heights. However we do not know which of these objects would be the most important to use to score optimially. To determine the weight we wish to give to each of these,
# we used a genetic algorithm to optimize the weights.

# Our major problem that we ran into was time to train the genetic algorithm. We further simplified the optimization by limiting individual games to only 1 million points. This is because we had one
# agent that as of right now is still running and has been running for 42 hours that has scored about 6.2 billion points. Unfortunately we don't have time to see this individual in the population
# to be further optimized (the variance in its score is absolutely massive). Instead we optimized for a slightly lower average but also a much smaller deviation.




from AnimatedTetris import *
from SimpleTetris import *
from kbinput import *
# from joblib import Parallel, delayed
# import numpy
import time, sys, random, math

import copy

# generation = 0


class HumanPlayer:
    def get_moves(self, piece, board):
        print "Type a sequence of moves using: \n  b for move left \n  m for move right \n  n for rotation\nThen press enter. E.g.: bbbnn\n"
        moves = raw_input()
        return moves

    def control_game(self, tetris):
        while 1:
            c = get_char_keyboard()
            commands =  { "b": tetris.left, "n": tetris.rotate, "m": tetris.right, " ": tetris.down }
            commands[c]()

#### Move Decision Functions ####

def mean(data):
    n = len(data)
    return sum(data)/n # in Python 2 use sum(data)/float(n)

def standev(data):
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss

# Takes the state of the tetris game and returns a 6-tuple containing the values of the heuristics
def getHeuristics(tetris):
    board = tetris.get_board()
    holes = 0
    agg = 0
    bumps = 0
    lines = 0
    column_heights = [0] * len(board[0])
    maxheight = 0
    maxwidth = 0
    row_widths = [0] * len(board)

    # Iterate over the board
    for c in range(len(board[0])):
        for r in range(len(board)):
            if board[r][c] == "x":
                row_widths[r] += 1
            # Increment column height if there is an 'x' or if there is a hole
            if board[r][c] == "x" or column_heights[c] >=  1:
                column_heights[c] += 1
                agg += 1
            # Count the number of holes
            if board[r][c] != "x" and column_heights[c] >=  1:
                holes += 1
        # Calculate the deviation of the heights of adjacent columns
        if c > 0:
            bumps += abs(column_heights[c - 1] - column_heights[c])
        # Get maximum height of columns
        if column_heights[c] > maxheight:
            maxheight = column_heights[c]
    for width in row_widths:
        if width > maxwidth:
            maxwidth = width
        if width == len(board[0]):
            lines += 1
    std = standev(column_heights)
    # sys.exit()
    return (agg, lines, holes, bumps, maxheight, maxwidth, std)

# Rotate a peice n times
def rotate(tetris, n):
    for i in range(n):
        tetris.rotate()

# Move a piece to the center of the board
def centerPiece(tetris):
    piece = tetris.get_piece()
    offset = 5 - piece[2]
    tetris.move(offset, tetris.piece)

# Drop a piece ot the bottom of the board but *do not remove completed lines*
def down(tetris):
    while not TetrisGame.check_collision(tetris.state, tetris.piece, tetris.row+1, tetris.col):
        tetris.row += 1

# Gives a hash function for different pieces
def getPieceNumber(tetris):
    piece = tetris.piece
    if piece == ["xxxx"] or piece == ["x", "x", "x", "x"]:
        return 0
    elif piece == ["xx", "xx"]:
        return 1
    elif piece == ["xx ", " xx"] or piece == [" x", "xx", "x "]:
        return 2
    else:
        return 4

# Gives the number of unique rotations a piece has
def numRotations(piece):
    if piece == 0:
        return 2
    elif piece == 1:
        return 1
    elif piece == 2:
        return 2
    else:
        return 4

# Gives the current width of a piece
def pieceWidth(tetris):
    piece = tetris.piece
    max = 0
    for c in range(len(piece)):
        if len(piece[c]) > max:
            max = len(piece[c])
    return max

# Calculate the score using only the current piece and the heuristics. Iterates over all possible rotations and locations.
def getScores(tetris, weights):
    scores = []
    pieceNum = getPieceNumber(tetris)
    rots = numRotations(pieceNum)
    for r in range(rots):
        for c in range(-5, 5, 1):
            t = copy.deepcopy(tetris)
            centerPiece(t)
            rotate(t, r)
            if pieceWidth(t) + c > 5:
                break;
            t.move(c, t.piece)
            down(t)
            t.state = TetrisGame.place_piece(t.state, t.piece, t.row, t.col)
            t.new_piece()

            agg, cl, holes, bumps, maxheight, maxwidth, std = getHeuristics(t)

            score = weights[0] * agg + weights[1] * cl + weights[2] * holes + weights[3] * bumps + weights[4] * maxheight + weights[5] * maxwidth + weights[6] * std
            scores.append([score, r, c])

            # Below is the code for calculating the heuristics based on the look-ahead piece. I found that this adds very little to the ability of the AI, but significantly increases
            # computation time. For this reason, I am leaving it commented (and not necessarily complete). At some later point in time I may decide to use this.

            # pieceNum2 = getPieceNumber(t)
            # rots2 = numRotations(pieceNum)

            # for r2 in range(rots2):
            #     for c2 in range(-5, 5, 1):
            #         t2 = copy.deepcopy(t)
            #         centerPiece(t2)
            #         rotate(t2, r2)
            #         if pieceWidth(t2) + c > 5:
            #             break;
            #         t2.move(c2, t2.piece)
            #         down(t2)
            #         t2.state = TetrisGame.place_piece(t2.state, t2.piece, t2.row, t2.col)


    return scores

# Finds the best score and returns the actions required to get to that score
def getBestLocation(tetris, weights):
    scores = getScores(tetris, weights)
    # print tetris.piece
    # print scores
    # sys.exit()
    max_score = -1
    max_val = -100000
    for score in scores:
        if score[0] > max_val:
            max_score = score
            max_val = score[0]
    return max_score[1:3]

#####
# This is the part you'll want to modify!
# Replace our super simple algorithm with something better
#
class ComputerPlayer:
    # Given a new piece (encoded as a list of strings) and a board (also list of strings),
    # this function should generate a series of commands to move the piece into the "optimal"
    # position. The commands are a string of letters, where b and m represent left and right, respectively,
    # and n rotates.
    #
    weights = []
    def loadSimulation(self, weights): # This allows me to load many sets of weights to train in parallel
        self.weights = weights
    def loadTetrisObject(self, tetris): # This allows the computer to simulate games
        self.tetris = tetris
    def get_moves(self, piece, board):
        # super simple current algorithm: just randomly move left, right, and rotate a few times
        # print self.weights
        loc = getBestLocation(self.tetris, self.weights)
        piece = self.tetris.get_piece()
        offset = 5 - piece[2]
        rot = "n" * loc[0]
        mov = ""
        cen = ""
        if offset < 0:
            cen = "b" * abs(offset)
        else:
            cen = "m" * abs(offset)
        if loc[1] < 0:
            mov = "b" * abs(loc[1])
        else:
            mov = "m" * abs(loc[1])

        # print rot + mov
        # print loc
        # self.tetris.print_board(False)
        # score = self.tetris.get_score()
        # if score % 1000000 == 0 and score != 0:
        #     print score
        #
        # if score > 1000000:
        #     return ""

        # if generation < 3 and score > 500:
        #     return ""

        return cen + rot + mov

    # This is the version that's used by the animted version. This is really similar to get_moves,
    # except that it runs as a separate thread and you should access various methods and data in
    # the "tetris" object to control the movement. In particular:
    #   - tetris.col, tetris.row have the current column and row of the upper-left corner of the
    #     falling piece
    #   - tetris.get_piece() is the current piece, tetris.get_next_piece() is the next piece after that
    #   - tetris.left(), tetris.right(), tetris.down(), and tetris.rotate() can be called to actually
    #     issue game commands
    #   - tetris.get_board() returns the current state of the board, as a list of strings.
    #
    def control_game(self, tetris):
        # another super simple algorithm: just move piece to the least-full column
        while 1:
            time.sleep(0.1)

            loc = getBestLocation(tetris, self.weights)
            # print loc

            centerPiece(tetris)

            rotate(tetris, loc[0])
            tetris.move(loc[1], tetris.piece)
            tetris.down()


###################
#### main program

(player_opt, interface_opt) = sys.argv[1:3]

population_size = 100
games = 20
cores = 4
mutation_chance = .03
mutation_amount = .2
replace_amount = .3

# fixed variables
num_weights = 7

# Normalize the weights in the unit n-sphere (as of right now is a 6-sphere, may add more heuristics and forget to update this comment)
def normalize(weights):
    sum = 0
    for i in weights:
        sum += pow(i, 2)
    if sum != 0:
        for i in range(len(weights)):
            weights[i] = weights[i] / math.sqrt(sum)
    return weights

# Simulate an instance of this game
def runGame(individual, num):
    tetris = SimpleTetris()
    player = ComputerPlayer()
    player.loadSimulation(individual[1])
    player.loadTetrisObject(tetris)
    try:
        tetris.start_game(player)
    except EndOfGame as s:
        return (tetris.get_score(), num)

# Take two individuals from the population and reproduce
def makeBaby(parents):
    baby = [0.0, [0]*num_weights]
    if parents[0][0] == 0:
        parents[0][0] = .001
    if parents[1][0] == 0:
        parents[1][0] = .001
    for i in range(num_weights):
        baby[1][i] = parents[0][1][i] * parents[0][0] + parents[1][1][i] * parents[1][0]

    if random.uniform(0, 1) < mutation_chance:
        for i in range(num_weights):
            baby[1][i] = baby[1][i] + random.uniform(-mutation_amount, mutation_amount)
    normalize(baby[1])
    return baby


def train():
    # Create the population
    generation = 0
    population = []
    for i in range(population_size):
        weights = normalize([random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)])
        population.append([0.0, weights])

    # Initialize the game with some of the best settings from previous optimizations (This is just to speed things up)
    population[0][1] = [-0.35269532056341407, 0.2078447559499224, -0.7646661444653853, -0.18741035785882304, -0.26933756105448103, 0.37420167257467196, 0]
    # individual score: 10885.7075
    population[1][1] = [-0.31518148368343696, 0.4103277558023909, -0.7735055983886067, -0.17704571725347007, -0.31197494542030524, 0.07285123097770088, 0]
    # individual score: 10786.23
    # Run until the average is over 100k
    while(population[population_size - 1][0] < 100000):
        # This allows me to run simulations in parallel on a multi-core machine. Dramatically speeds up optimization
        # Commenting out the fitness calculation because I used a nonstandard library and don't want to make the ai's download the module to run this code.
        fitnesses = Parallel(n_jobs=cores)(delayed(runGame)(population[i], i) for i in range(population_size) for j in range(games))
        for fitness in fitnesses: # Assign averages over runs as the fitness
            f, i = fitness
            population[i][0] += f

        for p in population:
            p[0] /= float(games)

        offspring = []
        # Replace some number of the worst individuals with new babies
        for j in range(int(replace_amount * population_size)):
            random.shuffle(population)
            parents = [[-1], [-2]]
            # Grab two best individuals from a random selection of 10% of the population
            for i in range(int(.1 * population_size)):
                if population[i][0] > parents[0][0]:
                    parents[1] = parents[0]
                    parents[0] = population[i]
                elif population[i][0] > parents[1][0]:
                    parents[1] = population[i]
            offspring.append(makeBaby(parents))

        def getFitness(ind):
            return ind[0]

        population = sorted(population, key = getFitness)
        # print population
        # sys.exit()
        for i in range(int(replace_amount * population_size)):
            population[i] = offspring[i]

        print "best: " + str(population[population_size - 1][0])
        avg = 0.0
        for ind in population:
            avg += ind[0]
        print "avg: " + str(avg / float(population_size))
        print "weights: " + `normalize(population[population_size - 1][1])`
        print "generation: " + `generation`
        generation += 1



# train()

try:
    if player_opt == "human":
        player = HumanPlayer()
    elif player_opt == "computer":
        player = ComputerPlayer()
    else:
        print "unknown player!"

    player.loadSimulation([-0.35269532056341407, 0.2078447559499224, -0.7646661444653853, -0.18741035785882304, -0.26933756105448103, 0.37420167257467196, 0])

    if interface_opt == "simple":
        tetris = SimpleTetris()

    elif interface_opt == "animated":
        tetris = AnimatedTetris()
    else:
        print "unknown interface!"

    player.loadTetrisObject(tetris)
    tetris.start_game(player)

except EndOfGame as s:
    print "\n\n\n", s
