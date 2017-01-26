# -----------------------------------------------------------------------------
# ---- Setup ------------------------------------------------------------------
# -----------------------------------------------------------------------------

# load numpy
# load time
# load itertools

import numpy as np
import time
import itertools

# load the data in as a multidimensional array
    # coord is the coordinates of each location, 48 USA Locations

coord = np.loadtxt('C:\\Users\\Nick\\Desktop\\TSP.txt', dtype = 'int32')

# lets extract the second and third columns in coord
    # these values represent the coordinate position of each location

# make a copy of the data

copy = coord

# redefine coord as an empty array

coord = np.empty((0, 2), dtype = 'int32')

# iterate through each row in copy and add the second and third columns to coord

for i in range(len(copy)):
    coord = np.append(coord, np.array([copy[i][1:3]]), axis = 0)

# -----------------------------------------------------------------------------
# ---- Population -------------------------------------------------------------
# -----------------------------------------------------------------------------

# randomly sample N solutions that all meet the maxdistance requirement

# setup input values:
    # it is the while loop iteration counter
    # done is the while loop control variable 
    # N is the population size
        # N must be even
    # maxdistance is the performance requirement a solution must meet to be in the population
    # pop stores the current set of solution canidates
    # popkeep stores all accepted solutions

it = 0
done = False
N = 100000
maxdistance = 500000
popkeep = np.empty((0, len(coord)), dtype = 'int32')

while done == False:
    
    # randomly sample solutions and append them to pop
    
    pop = np.empty((0, len(coord)), dtype = 'int32')
    
    for i in range(100):
        pop = np.append(pop, np.array([np.random.choice(a = len(coord), size = len(coord), replace = False)]), axis = 0)
    
    cusum = 0
    distance = np.empty((0, 1), dtype = 'int32')
            
    # compute the cumulative distance across all pairs for every solution in pop
    
    for j in range(100):
        for i in range(len(coord) - 1):
            if i == 0:
                cusum = cusum + np.sqrt((coord[pop[j][i]][0] - coord[pop[j][i + 1]][0])**2 + (coord[pop[j][i]][1] - coord[pop[j][i + 1]][1])**2) + np.sqrt((coord[pop[j][i]][0] - coord[pop[j][len(pop[j]) - 1]][0])**2 + (coord[pop[j][i]][1] - coord[pop[j][len(pop[j]) - 1]][1])**2)
            else:
                cusum = cusum + np.sqrt((coord[pop[j][i]][0] - coord[pop[j][i + 1]][0])**2 + (coord[pop[j][i]][1] - coord[pop[j][i + 1]][1])**2)
        
        distance = np.append(distance, round(cusum, 0))
    
    # append all acceptable solutions to popkeep
    
    keep = np.where(distance <= maxdistance)
    
    if(len(keep[0]) > 0):
        for i in range(len(keep[0])):
            popkeep = np.append(popkeep, [pop[keep[0][i]]], axis = 0)
    
    # track iterations
    
    it = it + 1
    
    # check if N solutions that all meet the maxdistance requirement have been found
    
    if(len(popkeep) >= N):
        pop = popkeep[range(N)]
        popcopy = pop
        done = True

# -----------------------------------------------------------------------------
# ---- GA ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

# tsp_ga is a function that returns the optimality gap of a tsp ga run
    # this is for experimentation purposes to find the correct input values

def tsp_ga(timelimit, threshold, N, F, PC, C, PM, M):
            
    # timehist is a history of the runtime at the end of each iteration
    # xhist is a multidimensional array containing the solution vectors of the best solution from eahc each iteration
    # dhist is the distance value of each solution corrresponding to xhist    
    # coshist is the mean cos product of all possible solution pairs
    # vectors is an array of all possible pairs of solutions
    # cos is an array of the cos of the dot product of each pair of solutions 
    
    pop = popcopy    
    timehist =  np.empty((0, 1), dtype = 'float32')
    xhist =  np.empty((0, len(coord)), dtype = 'int32')
    dhist =  np.empty((0, 1), dtype = 'int32')
    # coshist = np.empty((0, 1), 'float32')
    # vectors = np.array(list(itertools.product(range(N), range(N))))    
    # vectors = np.delete(vectors, np.arange(0, len(vectors), (len(vectors) / N) + 1, dtype = 'int32'), axis = 0)
    # cos = np.empty((0, 1), 'float32')
       
    done = False
    
    while done == False:
    
        # timestart is the start time of the GA
        
        timestart = time.clock()    
        
        # ---------------------------------------------------------------------
        # ---- Competition ----------------------------------------------------
        # ---------------------------------------------------------------------
        
        # compute the distance of all solutions
        
        # cusum will be a solution's total distance
        # distance will be an array of lenth N to represent the distance of each solution in pop
        
        cusum = 0
        distance = np.empty((0, 1), dtype = 'int32')
        
        # iterate through each solution in pop
        # iterate through each sequential pair of locations in each solution
        # for each pair of locations, compute the distance between them using the pythagorean theorem on the coordiantes from coord
        # compute the cumulative distance across all pairs for every solution in pop
        
        # notice that the first iteration sums the first:second distance and first:last distance
            # this is so the solution is one closed loop tour
        
        for j in range(N):
            for i in range(len(coord) - 1):
                if i == 0:
                    cusum = cusum + np.sqrt((coord[pop[j][i]][0] - coord[pop[j][i + 1]][0])**2 + (coord[pop[j][i]][1] - coord[pop[j][i + 1]][1])**2) + np.sqrt((coord[pop[j][i]][0] - coord[pop[j][len(pop[j]) - 1]][0])**2 + (coord[pop[j][i]][1] - coord[pop[j][len(pop[j]) - 1]][1])**2)
                else:
                    cusum = cusum + np.sqrt((coord[pop[j][i]][0] - coord[pop[j][i + 1]][0])**2 + (coord[pop[j][i]][1] - coord[pop[j][i + 1]][1])**2)
            
            distance = np.append(distance, round(cusum, 0))
        
        # index is just an empty vector to store the index value of the best tour of the current iteration
        # xbest is the best tour of the current iteration
        # dbest is the distance of the best tour
        
        index = np.empty((0, 1), dtype = 'int32')
        xbest = pop[np.append(index, np.where(distance == min(distance)))[0]]
        dbest = min(distance)
        
        # sample = np.random.choice(a = len(vectors), size = 10000, replace = False)
        
        # for i in sample:
        #     cos = np.append(cos, np.dot(pop[vectors[i][0]], pop[vectors[i][1]]) / np.linalg.norm(pop[vectors[i][0]]) / np.linalg.norm(pop[vectors[i][1]]))
            
        # randomly sample N competitions
        
        comp = np.empty((0, F), dtype = 'int32')
        
        # iterate through each solution and group it with (F - 1) randomly chosen solutions
        # comp is a multidimensional array where each row is a competition and the columns represent the F-many competitors
        
        for i in range(N):
            comp = np.append(comp, np.array([np.append(np.array([i]), np.random.choice(a = len(coord), size = F - 1, replace = False))]), axis = 0)
        
        # won is an empty vector that will hold the solutions that won each competition
        # results is the distance of each competitor in each competion of comp
        
        won = np.empty((0, 1), dtype = 'int32')
        results = distance[comp]
        
        # iterate through each row of results and determine which competitor had the minimum distance in the competition
        # store the solution id of the winning competitor for each competition in won
        
        for i in range(N):
            winner = np.empty((0, 1), dtype = 'int32')
            winner = np.append(winner, np.where(results[i] == min(results[i])))[0]
            winner = comp[i][winner]
            won = np.append(won, winner)
        
        # extract all of the winners as the new population
            
        pop = pop[won]
        
        # ---------------------------------------------------------------------
        # ---- Cross Sectioning -----------------------------------------------
        # ---------------------------------------------------------------------
        
        # sample is the sample of the solutions from pop that will go through cross sectioning
        # sample must take an even integer value to create pairs
        
        sample = np.random.choice(a = N, size = np.floor(PC * N).astype('int32'), replace = False)
        
        if np.floor(len(sample) / 2) < (len(sample) / 2):
            sample = sample[range(len(sample) - 1)]
        
        # cross is the pair of solutions that will be cross sectioned
        
        pair = np.floor(len(sample) / 2).astype('int32')
        cross = np.reshape(sample, (pair, 2))
        
        # lower is the minimum value that the upper bound of a cut can take
        # upper is the maximum value that the upper bound of a cut can take
        # uppercut is the upper bound
        # cuts is every cut that will be made in this cross section phase
        
        lower = np.floor(0.1 * len(coord)).astype('int32')
        upper = len(coord) - lower
        cuts = np.empty((0, 2), dtype = 'int32')
        
        # iterate through each cut, C, to determine all upper bounds for every cut
        # iterate thorugh each upper bound, uppercut, to determine all lower bounds for every cut
        
        for j in range(C):
            uppercut = np.random.choice(a = range(lower, upper), size = len(cross))
            
            for i in range(len(uppercut)):
                cuts = np.append(cuts, [np.array([np.random.choice(a = uppercut[i] - lower + 1, size = 1)[0], uppercut[i]])], axis = 0)  
        
        # sections is the sequence of cross sections to make on the solution pairs in cross
        
        sections = np.reshape(cuts, (C, len(cross), 2))
        
        # segment1 and segment2 are the two sections that will be swapped between each pair of solutions in cross
        # iterate through each cut, C, that each pair must go through
        # iterate through each pair, cross, to perform a particular cut
        
        for j in range(C):
            for i in range(len(cross)):
                segment1 = pop[cross[i][0]][range(sections[j, i, 0], sections[j, i, 1])]
                segment2 = pop[cross[i][1]][range(sections[j, i, 0], sections[j, i, 1])]
                
                pop[cross[i][1]][range(sections[j, i, 0], sections[j, i, 1])] = segment1
                pop[cross[i][0]][range(sections[j, i, 0], sections[j, i, 1])] = segment2
        
        # ---------------------------------------------------------------------
        # ---- Mutation -------------------------------------------------------
        # ---------------------------------------------------------------------
        
        # sample is the sample of the solutions from pop that will go through mutation
        # sample must take an even integer value to perform swaps
        
        sample = np.random.choice(a = N, size = np.floor(PM * N).astype('int32'), replace = False)
        
        if np.floor(len(sample) / 2) < (len(sample) / 2):
            sample = sample[range(len(sample) - 1)]
        
        # iterate through each cut, C, to determine all upper bounds for every cut
        # iterate thorugh each upper bound, uppercut, to determine all lower bounds for every cut
        
        for j in range(M):
            mutate = np.random.choice(a = len(coord), size = 2 * len(sample))
            
            for i in range(len(sample)):
                swap1 = mutate[i]
                swap2 = mutate[(2 * len(sample)) - i - 1]
                
                pop[sample[i]][swap1] = swap2
                pop[sample[i]][swap2] = swap1
                
        # ---------------------------------------------------------------------
        # ---- Store Iteration Data -------------------------------------------
        # ---------------------------------------------------------------------
        
        # update runtime, timehist, xhist, dhist, and coshist with the information from the current iteration
        
        xhist = np.append(xhist, np.array([xbest]), axis = 0)
        dhist = np.append(dhist, dbest)
        # coshist = np.append(coshist, np.mean(cos))
        runtime = time.clock() - timestart
        timehist = np.append(timehist, runtime)
        
        # ---------------------------------------------------------------------
        # ---- End Conditions -------------------------------------------------
        # ---------------------------------------------------------------------
    
        # time limit condition
        
        if timelimit <= sum(timehist):
            done = True
            
        # convergence condition
        
        # if np.mean(cos) >= threshold:
        #     done = True
    
    # return gap metrics
    
    mingap = 100 * ((min(dhist) - 10628) / 10628)
    maxgap = 100 * ((max(dhist) - 10628) / 10628)
    meangap = 100 * ((np.mean(dhist) - 10628) / 10628)
    gap = np.array([mingap, meangap, maxgap])
    
    return gap
    
# -----------------------------------------------------------------------------
# ---- Experimentation --------------------------------------------------------
# -----------------------------------------------------------------------------

# set up the input values to vary
    # timelimit is the maximum time in seconds that the ga is allowed to run before it terminates
    # threshold is the required population average value for the cos of the dot product of a population of paired solutions
        # a value approaching 1 means the population is converging onto one similar solution
    # N is the population size
    # F is the number of solutions that must compete with eachother per competition
    # PC is the proportion of the population that will go through cross sectioning
    # C is the number of cross sections that will be made on each solution
    # PM is the proportion of the population that will go through mutation
    # M is the number of mutations that will be made on each solution
    
timelimit = 60 * 2
threshold = 0.9
N = 1000
F = 2
PC = [1/4, 2/3]
C = [1, 3]
PM = [0.05, 0.10]
M = 2

# set up all possible combinations

doe = np.array(list(itertools.product(range(len(PC)), range(len(C)), range(len(PM)))))

len(doe)

# run all combinations in doe and store the average gap results for each run in gaphist

gaphist = np.empty((0, 3))

for i in range(len(doe)): 
    gaphist = np.append(gaphist, np.array([tsp_ga(timelimit = timelimit, threshold = threshold, N = N, F = F, PC = PC[doe[i, 0]], C = C[doe[i, 1]], PM = PM[doe[i, 2]], M = M)]), axis = 0)

# determine the best combination

doe[0]

# -----------------------------------------------------------------------------
# ---- Best Run ---------------------------------------------------------------
# -----------------------------------------------------------------------------

# set up the input values

timelimit = 3600 * 2
threshold = 0.9
N = 100000
F = 2
PC = 1/4
C = 1
PM = 0.05
M = 2

# manually run the body of the tsp_ga function

# how man iterations were completed

len(dhist)

# best solution

    # tour

index = np.empty((0, 1), dtype = 'int32')
xbest = xhist[np.append(index, np.where(dhist == min(dhist)))[0]]
xbest

    # distance

min(dhist)

    # gap %
    # optimal distance:
        # 10628

gap = 100 * ((min(dhist) - 10628) / 10628)
gap

# export GA data







