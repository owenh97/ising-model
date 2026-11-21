import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

X = np.load('X.npy')
y = np.load('y.npy')

X_flat = X.reshape(200,-1) # reshape lets logistic reg read 2d grids by making grids lines of numbers

X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=0
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#Verify train/test split

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)

#We found our score = 0.45, basically a coin flip, 18/40 correctly guessed.
#So let's test our data magnetisation
#This tests out of hot and cold grids, how ordered they are individually.

for label in [0,1]:
    grids = X[y==label]
    mags = np.abs(grids.mean(axis=(1,2)))
    print("label", label, "-> avg |magnetisation|:", mags.mean()) 

# magnetisation is the avg of all the spins in one grid. In one grid, add up all 2500 values (-1 and +1) and divide by 2500
# output : label 0 -> avg |magnetisation|: 0.7141440000000001, this means the cold grids are genuinely strongly ordered
# label 1 -> avg |magnetisation|: 0.046655999999999996, this means the hot grids are genuinely very disordered

"""
Conclusion :
1. Logistic Reg scores ~0.45 (a coin flip chance), while the magnetisation check confirms that the dataset is valid (ordered |M| ≈ 0.71 vs disordered |M| ≈ 0.047)
2. The problem stems from spin-flip symmetry: both +M and -M represent the same ordered phase, so only the magnitude matters while the sign cancels out
3. Because Logistic Reg is linear, a single straight threshold can't split both extremes (+0.71 and -0.71) from the centre (0).
4. This is the classic XOR/symmetry failure in linear classification
5. Fix : a non-linear model (neural net/CNN) that can learn from symmetry-invariant features directly from the raw grid
"""

#Implement Neural network, 200 datasets
#What's different? |x| = ReLU(x) + ReLU(-x); RELU(-x) keeps the negative side flipped up

nn = MLPClassifier(hidden_layer_sizes=(16,), alpha=1, max_iter=1000, random_state=0) #one hidden layer with 16 neurons
nn.fit(X_train, y_train)
print("neural net:", nn.score(X_test, y_test))

print("nn TRAIN:", nn.score(X_train, y_train)) #Output (without alpha): train is 1.0
print("nn TEST:", nn.score(X_test, y_test)) #Output (without alpha): test is 0.7

#Train High, Test Low. This is overfitting.
#Overfitting -> use regularization, L2 penalty, by adding alpha above

#For 200 datasets:
#When alpha=0.0001, nn TEST = 0.7
#When alpha=1, nn TEST = 0.8
#When alpha=10, nn TEST = 0.975

#L2 Penalty worked

#Now let's test larger datasets
#For 1000 datasets
#When alpha=0.0001, nn TEST = 0.825
#When alpha=1, nn TEST = 0.875
#When alpha=10, nn TEST = 1.0

#With a big enough dataset, the model clearly learned to tell cold grids from clearly hot ones.
#As it could tell the magnetisation gap was enormous (0.71 vs 0.047)

#Now let's test what does the model do with the questions it was never shown?
#Load the 10 grids at each temperature from 1.0 to 4.0. Feed it to the Neural network model to gauge results

X_sweep = np.load("X_sweep.npy")
temps = np.load("temps.npy")

#Flatten and predict

X_sweep_flat = X_sweep.reshape(len(X_sweep),-1)
preds = nn.predict(X_sweep_flat) #0 or 1 for each grid

#Group by temperature 
#We have 200 predictions and 200 temperatures (10 grids share each temp)
#For each temperature, find what fraction of its 10 grids the model called disordered (1)

unique_temps = np.unique(temps) #Each temperature has 10 appearances, so we remove duplicates and only have a list of unique temps
fraction_disordered = []
for t in unique_temps:
    mask = (temps == t) #We produce 20masks, each masks tells us which entry is made with, that particular temperature, basically mapping. We get [True, False, True....]
    fraction_disordered.append(preds[mask].mean()) #Mask basically grabs the True entries, and average out the [1,0,1,0], giving us the mean to append to fraction_disordered

#Plot temperature vs that fraction

plt.plot(unique_temps, fraction_disordered, marker='o')
plt.axvline(2.27, linestyle='--', color="red") #mark the true cut off for comparison
plt.xlabel("Temperature")
plt.ylabel("Fraction classified disordered")
plt.show()