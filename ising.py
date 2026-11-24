import numpy as np 
import matplotlib.pyplot as plt

#random.choice takes in parameter -1 or 1, and a size of 5 rows and 5 cols
np.random.seed(2)
grid = np.random.choice(a=[-1,1], size =(50,50)) 

print (grid)

"""
[[-1  1  1  1  1]
 [-1 -1 -1 -1 -1]
 [-1  1  1  1 -1]
 [-1 -1  1 -1  1]
 [ 1 -1  1 -1 -1]]

If (i,j) = (2,2) 
above (1,2) value = -1 
below (3,2) value = 1
left  (2,1) value = 1 
right (2,3) value = 1
"""

shape = grid.shape
N = shape[0]
print(N)

def curr(i,j):
    return (grid[i,j])

def above(i,j):
    return (grid[(i-1)%N,j])

def below(i,j):
    return (grid[(i+1)%N, j])

def left(i, j):
    return (grid[i,(j-1)%N])

def right(i, j):
    return (grid[i, (j+1)%N])

now = curr(4,4)
up = above(4,4)
down = below(4,4)
l = left(4,4)
r = right(4,4)

print (f"This is the current spin : {now}")
print (f"This is the spin above : {up}")
print (f"This is the spin below : {down}")
print (f"This is the spin leftwards : {l}")
print (f"This is the spin rightwards : {r}")

# A Spin wants to match its neighbour
# Two opposite is tense, high energy , nature wants to resolve it
# Physics convention sticks a minus sign in front so that agreement = low energy
# The energy of that pair is -(spin x neighbor) 
# Aligned -> -1 (low, good)
# Opposed -> +1 (high, tense)

def energy(i,j):
    current = curr(i,j)

    neighbour_sum = above(i,j) + below(i,j) + left(i,j) + right(i,j)
    return -(current)*(neighbour_sum)

print(f"Energy of (4,4) with its neighbors : {energy(4,4)}")

# Set up done 

"""
Now, we're going to walk up to a single spin and ask "should you flip?"
#The answer comes from one question : does flipping make the system happier (lower energy) or unhappier (higher energy)?
The quantity that answers it is delta E "ΔE", the change in energy if this spin flips. Not the energy itself, the change.

Case 1 : A spin that's aligned with its neighbours is already low-energy 
Flip it and it starts clashing -> energy goes up -> ΔE is positive -> flipping is a bad move

Case 2 : A spin that's clashing with its neighbours is high energy (tense). Flip it and it falls into alignemnt
-> energy goes down -> ΔE is negative -> flipping is a good move.

Summary : Ideally we want ΔE < 0. 
"""

def delta_E(i,j):
    """
    This func answers ; if i flip this one spin, does the grid get happier, or unhappier, and by how much?
    The negative ΔE output we get means "flipping makes things better, flip it".
    The positive ΔE output means flipping it makes things worse

    ΔE isn't the thing we care about at the end, it's a diagnostic test that answers
    "would flipping this spin move things toward harmony or away from it?"
    """
    return -2 * energy (i,j)

print (f"This is the delta_E of (4,4) {delta_E(4,4)}")

# Done setting up the diagnostic test

"""
Set up Metropolis algorithm
p = e^(−ΔE / T), this p means how likely a spin is to flip when external heat tries to infiltrate it
How likely the external heat energy will knock a spin out of alignment and force it to flip against its neighbors

Two cases, hot and cold ΔE = 2

Case 1 : Cold, T = 0.5
chance e^(-2/0.5) ≈ 0.02. Happens 2% of the time.

Case 2 : Hot, T = 5
chance e^(-2/5) ≈ 0.67. Happens 67% of the time. Constant rebellion. the grid turns into chaos

If it's a -ΔE or ΔE = 0 we flip it immediately, no questions asked. because it's a good flip
But if it's a ΔE positive, we need to compute p, draw a random r between - and 1, flip only if r < p.
"""

def metropolis_alg(i,j,T):
    d = delta_E(i,j)
    #If the delta is negative or 0 it means that this is for sure a good flip
    if d <= 0 : 
        grid[i,j] = - 1 * grid[i,j]
    #if its a positive delta, we find p, and draw a random r between 0 and , flip only if r < p 
    else:
        p = np.exp(-d/T)
        rand = np.random.random() #Draw a random
        if rand < p: 
            grid[i,j] = -1 * grid[i,j]

print(f"Before flip : {grid[4,4]}")
metropolis_alg(4,4,5)
print(f"After flip : {grid[4,4]}")


"""
A sweep = give every spin a chance to flip

"""


def sweep(T):
    for i in range(N):
        for j in range(N):
            metropolis_alg(i,j,T)

#sweep(5)
#print(grid) #First sweep ont his 5x5 grid

# Try bigger sweep

#1 Sweep gives every single spin on grid a chance to flip
for step in range(100):
    sweep(5) #T below 2.27 is cold, above 2.27 is hot
#plt.imshow(grid)
#plt.show() # Shows our grid froze to ice at T =1 


"""
Generate labelled datasets, build a classifier
"""


X = []
y = []
n_samples = 500  #Test in small amounts first, scale up once it works
 
for i in range(n_samples):
    #cold grids

    grid = np.random.choice(a=[-1,1], size=(50,50))
    t = np.random.random() * 1.0 + 1.0 #random cold temp in between, between [1.0, 2.0); range [a,b] : random() * (b-a) + a
    for step in range(500):
        sweep(t)
    X.append(grid)
    y.append(0) #for each run we save one 0 = ordered 

for i in range(n_samples):

    #hot grids

    grid = np.random.choice(a=[-1,1], size=(50,50))
    t = np.random.random() * 1.4 + 2.6 #random hot temperature between [2.6-4.0)
    for step in range(500):
        sweep(t)
    X.append(grid)
    y.append(1) #unordered, hot grids, labelled 1


#print(len(X))
#print(len(y))

#Save the first 200 datasets
#np.save('X.npy', np.array(X))
#np.save('y.npy', np.array(y))


#Save 1000 grids 
np.save('X_1000.npy', np.array(X))
np.save('y_1000.npy', np.array(y))

#Now we want to test what does the model do with the questions it was never shown?
#So we generate datasets, consists of grids that are swept on different temperatures
#There are 10 grids per temperature, swept from temp 1.0-4.0, and we feed the Neural network we built previously
"""
X_sweep = []
temps = []
grids_per_temp = 10

for t in np.arange(1.0,4.0,0.15):
    for i in range(grids_per_temp):
        grid = np.random.choice(a=[-1,1], size=(50,50))
        for step in range(100):
            sweep(t)
        X_sweep.append(grid.copy())
        temps.append(t)

print(len(X_sweep))
np.save("X_sweep.npy", np.array(X_sweep))
np.save("temps.npy", np.array(temps))
"""