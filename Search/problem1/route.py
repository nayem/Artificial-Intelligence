#!/usr/bin/python

# Formulation of search problem:
# We consider the statespace as a collection of nodes (cities) with connecting edges (roads).
# Each city has on average 3-4 roads connecting to it. The successor function returns a list of all
# roads connecting to the current city minus all roads that have already been visited. The weights
# were defined as follows:
# Segments: uniform cost for every road
# Distance: road length (where road length = 0 where not specified)
# Time: road length / speed limit (where speed limit = 35mph when not specified)
# Scenic: Weight of 1 for every road with speed limit > 55mph
# The heuristics are defined as follows:
# Segments: H(x) = 0 for all roads (there were no heuristics that we could come up with that were admissible)
# Distance: H(x) = estimated distance to goal.
#   The estimated distance is the geographical distance between coordinates or the previous geographical distance - distance travelled
# Time: H(x) = estimated distance (as defined above) / max_speed_limit
# Scenic: H(x) = 0 for all roads (again no admissible heuristics to the best of our knowledge)

# Description of search algorithms
# BFS
# BFS takes a start state and evaluates every successor to that city. Then it evaluates every successor to those successors.
# It continues as such until it either finds a goal state or runs out of successors
# DFS
# DFS takes a start state and finds every successor, it then evaluates one of those successors. It continues as such evaluating the longest route first.
# IDS
# IDS Evaluates a state and adds its successors to the fringe. It evaluates until the route length is equal to the iterative depth.
# The depth continues lengthening after every node meets this condition or until it finds the goal state.
# ASTAR
# ASTAR evaluates a state and puts the successors in order based on their cost in the fringe.
# All four of the search algorithms take states from the successor function and append them to the fringe in some predefined way.

# We made a few assumptions with the dataset given to us. First, there were several cities without coordinates. We used an estimation of distance to account for this.
# We also made the assumption that an acceptable speed limit for roads without speed limits was 35mph. This is based on the assumption that you can at minimum go 35mph anywhere, so worst case we are always underestimating.
# A major problem that we faced was the size of the search space. We used an inefficient data structure to store the data.
# Because the access time was linear in the length of the data, most searches would take upwards of 1-2hrs. Once converting to a constant time access data structure, the searches took at most a few minutes.

# Which search algorithm seems to work best?
# Astar is the fastest of the algorithms for nearly every search besides a few very specific searches. If a search is the first path traversed by dfs, then it can be more efficient (though this is highly rare).

# Which algorithm is fastest according to experimentation?
# ASTAR: 5.9 seconds for 500 trials
# DFS: 8.7 seconds for 500 trials
# BFS: 1hr and 6 minutes for 500 trials
# IDS: 1hr and 6 minutes for 500 trials
# All of these were measured from Bloomington, Indiana to Chicago

# Which algorithm uses the least memory
# DFS uses the least ram, but only negligibly less than astar.

# Which heuristic did you use?
# The best we found was euclidean distance between the current location and the goal location. We could have improved it by taking into account direction when estimating distance. This would have still been an esitmation, but could have been better.

# Which is the farthest city from Bloomington?
# Skagway,_Alaska

import sys
import csv
from heapq import *
import math

# Get distance between coordinates in miles
def distance(origin, destination):
    lat1, lon1 = origin

    if lat1 == 0 and lon1 == 0:
        return

    lat2, lon2 = destination
    radius = 3959 # miles

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

# ------------Parse Command Line Args-----------

arguments = sys.argv
if len(sys.argv) != 5:
    raise Exception('Not enough arguments!')

start_city = arguments[1]
end_city = arguments[2]
routing_option = arguments[3]
routing_algorithm = arguments[4]

if routing_option not in ['segments', 'distance', 'time', 'scenic']:
    raise Exception("Don't know that routing option")

if routing_algorithm not in ['bfs', 'dfs', 'ids', 'astar']:
    raise Exception("Don't know that routing algorithm")

# ------------Classes-----------

# stores city data
class City(object):
    name = ""
    latitude = 0
    longitude = 0
    number = 0 # location in global city list

    def __init__(self, name, latitude, longitude, number):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.number = number

# stores road data
class Road(object):
    name = ""
    city1 = ""
    city2 = ""
    length = 0
    speed = 0

    def __init__(self, city1, city2, length, speed, name):
        self.city1 = city1
        self.city2 = city2
        self.length = length
        self.speed = speed
        self.name = name

class State(object):
    visited_roads = [] # list of every road in this route
    cost = 0.0 # accumulation of costs
    last_dist = 0 # last estimated distance traveled (used in case a road has an invalid distance or a city has no coordinates)
    city = 0 # The city that this state is currently in

    # Helper function to guarantee the visited roads passes a new list, not a pointer to the old one
    def copyVisited(self):
        return list(self.visited_roads)

# Returns the last city visited by a state (abstracted due to rapidly changing api during development)
def getLastCity(state):
    return(state.city)

# Returns the last road traveled to get to current state
def getLastRoad(state):
    return(state.visited_roads[-1])

# Returns the destination city given a road and a starting city
def getConnectingCityName(city, road):
    if city.name == road.city1:
        return road.city2
    elif city.name == road.city2:
        return road.city1
    else:
        raise Exception("uh oh. This road doesn't connect to that city at all!")

# Adds a city to the global city list (used because many junctions are not listed in the city data, but are treated like cities)
def addCity(city_name):
    city_num = len(cities)
    city_hash[city_name] = city_num
    city = City(city_name, 0, 0, city_num)
    cities.append(city)
    return(city)

# ------------Global Variables-----------

cities = [] # list of all cities in space
roads = [] # list of all roads in space
city_edges = {} # dictionary containing every road that connects to a city
city_hash = {} # Dictionary mapping a city name to its index in the global city list
goal_coords = (0,0) # Target coordinates used for some of the heuristics

# ------------Read Data from Files-----------

with open('city-gps.txt', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter = ' ')
    i = 0
    m = -1
    it = 0
    for row in reader:
        cities.append(City(row[0], float(row[1]), float(row[2]), i))
        city_hash[row[0]] = i
        i = i + 1

with open('road-segments.txt', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter = ' ')
    for row in reader:
        speed = 35
        if row[3] != '':
            speed = int(row[3])
        if speed == 0:
            speed = 35

        dist = int(row[2])

        road = Road(row[0], row[1], dist, speed, row[4])
        roads.append(road)
        exists = city_edges.get(row[0], -1)
        if exists != -1:
            city_edges[row[0]].append(road)
        else:
            city_edges[row[0]] = [road]
        exists = city_edges.get(row[1], -1)
        if exists != -1:
            city_edges[row[1]].append(road)
        else:
            city_edges[row[1]] = [road]

# ------------Search-----------

# Returns true if we have visited a road before on this route
def isPreviousState(state, road_name):
    return next((road for road in state.visited_roads if road.name == road_name), None) != None

# Returns list of states from a current state
def successors(state):
    next_states = []
    city = getLastCity(state)
    edges = city_edges[city.name]
    for edge in edges:
        city_name = getConnectingCityName(city, edge)
        city_num = city_hash.get(city_name, -1)
        new_city = 0
        if city_num == -1:
            new_city = addCity(city_name)
        else:
            new_city = cities[city_num]
        if not isPreviousState(state, edge.name):
            new_state = State()
            new_state.city = new_city
            new_state.visited_roads = state.copyVisited()
            new_state.visited_roads.append(edge)
            next_states.append(new_state)
            if routing_option == "segments":
                new_state.cost = state.cost + 1
            elif routing_option == "distance":
                new_state.cost = state.cost + edge.length
            elif routing_option == "time":
                new_state.cost = float(state.cost) + (float(edge.length) / float(edge.speed))
            elif routing_option == "scenic":
                if edge.speed >= 55:
                    new_state.cost = state.cost + 1
    return(next_states)

# Returns true if we are at the goal city
def isGoal(state):
    return(getLastCity(state).name == end_city.name)

def bfs(start):
    fringe = [start]
    while (len(fringe) > 0):
        for s in successors(fringe.pop()):
            if isGoal(s):
                return(s.visited_roads)
            fringe.insert(0, s)
    return []

def dfs(start):
    fringe = [start]
    while (len(fringe) > 0):
        for s in successors(fringe.pop()):
            if isGoal(s):
                return(s.visited_roads)
            fringe.append(s)
    return []

def ids(start):
    def inner_dfs(route, depth):
        if depth == 0:
            return
        if isGoal(route[-1]):
            return route
        for s in successors(route[-1]):
            if s not in route:
                next_route = inner_dfs(route + [s], depth - 1)
                if next_route:
                    return next_route

    for depth in range(1, 10000):
        route = inner_dfs([start], depth)
        if route:
            return route[-1].visited_roads

    return []

# Estimates the distance between current state and goal. (Uses coordinates if it can, reverts to an estimation otherwise)
def estimate_distance(state):
    if state.city.latitude == 0 and state.city.longitude == 0:
        return state.last_dist - getLastRoad(state).length
    return distance(goal_coords, (state.city.latitude, state.city.longitude))

def astar(start):
    pq = [(0,start)]
    while len(pq) > 0:
        (p,state) = heappop(pq)

        if isGoal(state):
            return state.visited_roads
        else:
            for s in successors(state):
                heuristic = 0
                if routing_option == "segments":
                    heuristic = 0
                elif routing_option == "distance":
                    dist = estimate_distance(state)
                    state.last_dist = dist
                    heuristic = dist
                elif routing_option == "time":
                    dist = estimate_distance(state)
                    state.last_dist = dist
                    heuristic = float(dist) / 75.0 # need max speed here
                elif routing_option == "scenic":
                    heuristic = 0

                heappush(pq,(s.cost + heuristic, s))

    print "No Solution Found"
    return []

end_city = cities[city_hash[end_city]]
start_city = cities[city_hash[start_city]]
start_state = State()
start_state.city = start_city
goal_coords = (end_city.latitude, end_city.longitude)
start_coords = (start_city.latitude, start_city.longitude)
start_state.last_dist = distance(goal_coords, start_coords)

route = []

if routing_algorithm == "bfs":
    route = bfs(start_state)
elif routing_algorithm == "dfs":
    route = dfs(start_state)
elif routing_algorithm == "ids":
    route = ids(start_state)
elif routing_algorithm == "astar":
    route = astar(start_state)

# ------------- Print results -----------------

print "start in " + start_city.name
last_city_name = start_city.name
time = 0.0
dist = 0.0
strings = ""
for road in route:
    last_city_name = getConnectingCityName(cities[city_hash[last_city_name]], road)
    strings += last_city_name + " "
    dist += float(road.length)
    time += float(road.length) / float(road.speed)
    print "take " + road.name + " for " + str(road.length) + " miles to " + last_city_name

print "Total time: " + str(time) + " hours"
print "Total distance: " + str(dist) + " miles"

print str(dist) + " " + str(time) + " " + strings
