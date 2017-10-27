# Assignment1: Searching (Group: bxiong-andnpatt-knayem-a1)
# Problem 3:
#
# Brife Description: A file of friend-list is given. 
# I want to place the friends in such a way that in a table the seated people are not mutually friend.
# So first, I make the friendship graph(Adjacency Matrix, 1->friend, 0->not friend) from the file, then later use this same graph to place the Mutually_Not_Friend people in a table.
# By the definition of the problem, all the people of a table are mutually not friend, which means there will be a non-friend edge(0) between each pair of people in the table like- a complete graph(clique).
# Now, the optimal solution of table assignation means requireing less table where seat_per_table is given. 
# So if I can place maxium number of people in each table, then I will need least number of table. 
# So that, I start by looking for Mutually_Not_Friend complete graph of size seat_per_table. If I get any clique like that, assign a table for them. 
# Then again look for table to place other friends. 
# If not clique of size seat_per_table is avaiable, then decrease the size of seat_per_table and look for smaller clique until all friends are assigend to some table.
#
# State Space: All the combinations of friends who are not mutually friend to each other and each combinations is size SEATS_PER_TABLE to 1.
# Successor Function, s(x): All the friends who are not mutually friend of x and yet not assigend to any table.
# Edge Weight: Uniform, in the reverse friendship graph if not-friend there is an edge(0), otherwise no edge(1).
# Heuristic Function: Consider seat_per_table size clique first, then consider smaller size clique. 
# Because, placing all the people in seat_per_table will require least number of table. And this understimates than the actual number of table needed.
# Assumption: When several same size cliques are found, then the priority is given to the clique which sum of degee of all node is minimum.
# Because if the people who has less not friend has less edge(more constrainted). So placing them first in a table is better.
# Problem: Finding clique in a graph is a NP-hard problem. 
# So I start from seat_per_table size clique and decrease the size until no friend is left to assign in a table.
#
# Search Algorithm:
# Repeat until all friends are assigend and clique_size->[seat_per_table ... 1]
#	for x in all friends which are not assigend:
#		s -> s + generate_successor(x)
#	clique -> List_of_all_cliques in (s)
# 	Calculate priority for each clique
#	fringe -> min_Priority_Queue()
#	fringe.push(key, clique)
#	Repeat Until fringe is empty:
#		c->fringe.pop()
#		assign a table to all the friends of c
#		Remove the assigend friends from friend list
#		Remove all the clique with the assigend friends from fringe

import sys
import heapq
import itertools
# Global variables
FILE_NAME = "friendship.txt"
SEATS_PER_TABLE = 3
node_dict = dict()  # [name]->index map of matrix
vertex_dict = dict()    # [name]->Vertex Object map
table_map = dict()  # [index]->frind_list, Final table assignment of friend
adjacency_matrix = []

class Vertex:
    def __init__(self, key, value, degree=0, is_table_assigned=False, assigned_table_id=-1):
        self.key = key
        self.value = value
        self.degree = degree
        self.is_table_assigned = is_table_assigned
        self.assigned_table_id = assigned_table_id

def create_graph():
    global adjacency_matrix
    # Read file to get number of friend
    node_set = set()
    try:
        fh = open(FILE_NAME)
        for word in fh.read().split():
            node_set.add(word)
    except IOError as e:
        print("File No Found ({})".format(e))
    # [name]->index map of matrix
    for key in node_set:
            node_dict[key] = len(node_dict)    
            
    adjacency_matrix = [[0 for c in range(len(node_dict))] for r in range(len(node_dict))]
    # Fill up the Adjacency Matrix 
    fh = open(FILE_NAME)
    for line in fh.readlines():
        new = True
        for word in line.split(): 
            if(new):
                first = word
            else:
                adjacency_matrix[node_dict[first]][node_dict[word]] = 1
                adjacency_matrix[node_dict[word]][node_dict[first]] = 1
            new = False
    # Create all Vertex Object
    for key in node_dict.keys():
        degree = len(node_dict) - sum(adjacency_matrix[node_dict[key]]) - 1
        vertex_dict[key] = Vertex(key, node_dict[key], degree)
# Sum of all vertex degree in a clique  
def get_clique_sum(clique):
    sum = 0
    for key in clique:
        sum += vertex_dict[key].degree
    return sum
# Generate list of successor including the vertex
def generate_successor(node):
    successor = []
    for key in vertex_dict.keys():
        if vertex_dict[key].is_table_assigned == False and adjacency_matrix[ node_dict[node] ][ node_dict[key] ] == 0:
            successor.append(key)
    return successor
# Return the cliques only
def filter_complete_graph(graph):
    complete_graph = []
    for scc in graph:
        flag = True
        edge_list = list(itertools.combinations(scc, 2))
        for e in edge_list:
            if adjacency_matrix[node_dict[e[0]]][node_dict[e[1]]] == 1:
                flag = False
                break
        if flag:
            complete_graph.append(list(scc))
    return complete_graph
# Assign a table to a clique
def assign_clique(clique, fringe):
    table_map[len(table_map)] = clique
    for scc in clique:
        vertex_dict[scc].is_table_assigned = True
        vertex_dict[scc].assigned_table_id = len(table_map)
        node_dict.pop(scc)
        # Remove all the other clique of this vertex
        N, n, i = len(fringe), 0, 0
        while n < N:
            if scc in fringe[i][1]:
                fringe.pop(i)
            else:
                i += 1
            n += 1
# Print the final table assignment
def print_table():
    print len(table_map),
    for key in table_map.keys():
        print ",".join(table_map[key]),
# Start assigning table
def table_assign():
    fringe=[]
    seat_num = SEATS_PER_TABLE
    while seat_num <= SEATS_PER_TABLE and len(node_dict) > 0:
        all_combination = set()
        for key in node_dict.keys():
            children = generate_successor(key)
            for scc in list(itertools.combinations(children, seat_num)):
                all_combination.add(scc)
        complete_graph = filter_complete_graph(all_combination)
         
        for scc in complete_graph:
            heapq.heappush(fringe, (get_clique_sum(scc), scc))
         
        while len(fringe):
            assign_clique(heapq.heappop(fringe)[1], fringe)
        seat_num -= 1
    print_table() 
        
if(len(sys.argv) == 3):
    FILE_NAME = sys.argv[1]
    SEATS_PER_TABLE = int(sys.argv[2]) 
create_graph()
table_assign()
