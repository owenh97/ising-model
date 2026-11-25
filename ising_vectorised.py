# Computing neighbour sum in ising.py was naive loop-based. We fix it with Vectorisation
# Vectorisation shifts the entire 2D array in 4 directions at once to compute the neighbor sum for every spin simultaneously
# Doing one operation on the whole array at once instead of looping cell by cell

import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(0)
grid = np.random.choice([-1,1], size=(50,50))

# np.roll adds every cell's neighbor sum in one ago
neighbour_sum = (
    np.roll(grid,1,axis=0)+
    np.roll(grid,-1,axis=0)+
    np.roll(grid,1,axis=1) +
    np.roll(grid,-1,axis=1)
)

#Formula ΔE, the energy change, for every cell at once.
delta_E = 2 * grid * neighbour_sum

#Build a checkerboard, because we want to instruct the algorithm where it is allowed to calculate
# Therefore we can flip lots of cells at once without neighbors clashing. 
# black cells never touch each other (every black's neighbor are white)
# So flipping every black cells at once is safe.
# Without it, two touching cells flipping together violate the physics.
i, j = np.indices(grid.shape)
black = (i+j) % 2 == 0 

#Check if it works with alternating bnw grid
#print(black[:6, :6])

#Recompute first (the grid changed when black flipped)
#Then flip the white cells

def sweep(grid, T):
    i,j = np.indices(grid.shape)                           #two grids: row numbers, and column numbers
    black = (i+j) % 2 == 0                                 # mark checkboard cells (row+ col even = black)
    for color in (black, ~black):                          #run the inside twice, black cells, then white (~black)
        nsum = (np.roll(grid,1,0) + np.roll(grid,-1,0) +   
                np.roll(grid,1,1) + np.roll(grid,-1,1))    #each cell's 4 neighbours, summed u
        dE = 2 * grid * nsum                               #energy change if each cell flipped
        accept = np.random.random(grid.shape) < np.exp(-dE/T) #which flips are allowed
        grid[color & accept] *= -1                              #flip this colour's accepted cells

grid = np.random.choice([-1,1], size=(50,50))
start = time.time()

for step in range(100):
    sweep(grid,1.0)

#Assess the time taken after vectorisation
print("vectorised:", time.time() - start,"s")
#We got 0.007 for 100 sweeps, which is much faster after optimisation


# Since T=2.27 is the fence between cold and hot, results near this point swings wildly
# So we measure magnetisation and its jitter 

def magnetisation(grid):
    return abs(grid.mean())

temps = np.arange(1.0,4.0,0.1) #From 1.0-4.0 with 0.1 interval

for size in [10,20,40,60]:
    jitters= [] 

    for T in temps:
        grid = np.ones((size,size), dtype=int)
        for step in range(200):  #Sweep the grid 200 times first as a warmup, let it reach equilibrium at this temp
            sweep(grid, T)
        mags = [] 
        for step in range(200):  #Now it's settled, run 200 more times and record the magnetisation each time.
            sweep(grid, T)
            mags.append(magnetisation(grid))
        chi = size * size * np.var(mags) / T #Formula that measures jitters (magnetic susceptivility)
        jitters.append(chi)
    plt.plot(temps, jitters, marker='o', label=f'{size}x{size}')

plt.axvline(2.27, linestyle='--', color='red')
plt.xlabel('Temperature')
plt.ylabel('Magnetisation jitter (≈ susceptibility)') #susceptibility = how strongly the system responds to a push/ how easily the grid mag swings
plt.legend()
plt.show()

#We got a plot that reveals a sharp mountain peak around T=2.27, proving that the machine simulates real physics
#Real life eg: Any piece of hot iron cooling down to room temp must pass thru the Curie critical point, where it experiences a surge of magnetic fluctuations.