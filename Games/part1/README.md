##Assignment 2: Games (Group: andnpatt-knayem-bxiong-a2)
###Problem 1: n-k-coh-coh

####Brief Description: 
A board state of n-k-coh-coh is given where (w)->white marbel, (b)->black marbel and (.)->empty spave in the boad.
The player who plays first is the white marbel player. Also we have a given Time in which we have to suggest a move.
So for any board configeration, we create an Interative Deepening MinMax tree. After traversing a fixed depth, we chech whether we have enough time left to traverse any more depth. If enough time remains, we go for another depth; otherwise returns the current best move. Moreover, if time exceeds before iterating all nodes of current depth, then we return the best value we see so far.

####State Space: 
All the combinations of White and Black marbels of board size NxN.
####Successor Function, s(x): 
All the new board configeration possible after placing a marbel(color of the marbel determines by player's) in any empty space.
####Edge Weight: 
Uniform, all states are equal probable.
####Heuristic Function: 
No heuristic function is used since we tested the performance for case when N=10,K=5 and N=6,K=3, with time limits ranging from 1 to 10 seconds, the program without heuristic function outperform the one with heuristic function by 1 to 2 rounds, and usually is almost twice as fast as the heuristic one when no time limit is attached.

####Assumption: 
First player always starts the game by placing White marbel. And both player should play optimally. If wining is not possible from any board states, then player will try to prolong the game duration.
####Problem: 
Too many successor states. Tried the mirror/symeetric states, but they didn't work out fine.

####Design Decision:
To play a psudo-Tournament, we write function run_player(). This returns the New board in string.  
