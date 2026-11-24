import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X = np.load('X_1000.npy')
y = np.load('y_1000.npy')

#converts np array into pytorch tensor
#unsqueeze(1) adds a size-1 dimension at position 1. (N, 50, 50) -> (N, 1, 50, 50)
#4D shape reads as : (how many grids, how many feature-layers per grid, height, width)
#We make it 4D because CNN expects every 2D image to explicitly state how many color channels it has
#Channel/Feature layer : how many stacked sheets make up one grid. A standard color photo has 3 (RGB), but this ising only has one (B&W)

X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1) 
y_t = torch.tensor(y, dtype=torch.long)

print(X_t.shape, y_t.shape)

#What happens inside the CNN pipeline

#1. Conv Layer : Think of this as a tiny magnifiying glass that scans across the 50 x 50 grid.
#If we use 16 filters, the CNN outputs 16 new feature layers per grid, where each layer highlights a different pattern it spotted.

#2. ReLU : the non-linearity we already met (keep positives, strip out negatives, turn negative numbers into 0). 
#Rmv weak noise so strong pattern matches pass forward, and lets the network learn complex boundaries

#3. MaxPool : Downsampling. Shrinks the grid down (50x50 to 25x25) by looking at every 2x2 square and keeping only the biggest number.
# It zooms out to keep the high contrasting/dramatic effects int ehg rid while throwing away tiny, unneeded details. (Such as there's a huge indicator of a hot grid/cold grid)

#4. Flatten : Unrolling into a list : Takes all the shrunk down 2d squares and stretches them out into one long, flat line of numbers
# Combines everything into 2 scores: ordered vs disordered. The bigger score wins

# Shape Flow : 
# (N,1,50,50) --conv--> (N,8,50,50) --relu--> (N,8,50,50)
#             --pool--> (N,8,25,25) --flatten--> (N, 8*25*25=5000)
#             --linear--> (N,2)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,8,kernel_size=3, padding=1) # 1 channel in, 8 ouput channels/ filters, 3x3 filter, padding=1 adds a 1-pixel border
        self.pool = nn.MaxPool2d(2)  #2x2 downsample
        self.fc = nn.Linear(8 * 25 * 25, 2)  #flattened features -> 2 class scores (ordered vs disordered)
        #nn.Linear takes in 2 args nn.Linear(in_features, out_features).
        #our in_features = channels x pooled_height x pooled_width

    def forward(self, x):
        x = self.conv(x)  # (N,1,50,50) -> (N,8,50,50)
        x = torch.relu(x) # non-linearity
        x = self.pool(x)  # -> (N,8,25,25)
        x = x.flatten(1)  # -> (N,5000) (flatten everything except the batch dim)
        x = self.fc(x)    # -> (N,2)
        return x 

#model = CNN()
#out = model(X_t[:4]) #feed the first 4 grids
#print (out.shape) # output : torch.Size([4, 2]), meaning our output is 4 rows, 2 cols. Rows = grid we fed. Cols = 2 class scores for each grid

# The training loop (3 main parts)

#1. Loss function : one number measuring how wrong the model's predictions are. CrossEntropyLoss(). High = badly. low = conf

#2. Optimiser : the thing that adjusts the weights to reduce the loss. Adam is a smart version of gradient descent. 
# We hand it to model.parameters() so it knows which weights to tune

#3. Loop : repeat many times. Reduce the loss, optimise, and nudge weights

X_train, X_test, y_train, y_test = train_test_split(
    X_t, y_t, test_size=0.2, random_state=0
)

model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001) #Learning rate, how big the nudge is

for epoch in range(50): #One pass = one epoch, one nudge over the whole training set. 50 nudges here.
    out = model(X_train) #forward : predict(logits)
    loss = loss_fn(out, y_train) #measure how wrong

    optimiser.zero_grad() #clear the old gradients (Pytorch accumuldates them by default, so we reset each pass or they pile up)
    loss.backward() #Backpropagation. Calculus that works out for each weight which direction to nudge it to lower the loss.
    optimiser.step() #Apply the nudge. Move every weight a small step in direction backward() found.

    print(epoch, loss.item()) #Pull the loss out as plain number so we can see the traction

# evaluate on held-out grids
with torch.no_grad(): 
    pred = model(X_test).argmax(1) #pick the bigger of the 2 scores -> 0 or 1 (disordered or ordered)
    acc = (pred == y_test).float().mean()
    print("test accuracy:", acc.item())

#Over 50 iterations, we achieved test accuracy = 1.0

# Perform prediction on X_sweep.npy
# We want to know "where does the model think the transition is?""
X_sweep = np.load('X_sweep.npy')
temps = np.load('temps.npy')

#prep for the CNN : add the channel dimension, exactly like the training data
X_sweep_t = torch.tensor(X_sweep, dtype=torch.float32).unsqueeze(1)

with torch.no_grad():
    preds = model(X_sweep_t).argmax(1).numpy() #0 or 1 per grid, as a numpy array

#group by temperature
unique_temps = np.unique(temps)
fraction_disordered = []
for t in unique_temps:
    mask = (temps == t)
    fraction_disordered.append(preds[mask].mean())

plt.plot(unique_temps, fraction_disordered, marker='o')
plt.axvline(2.27, linestyle='--', color='red')
plt.xlabel('Temperature')
plt.ylabel('Fraction classified disordered')
plt.title('CNN phase transition detection')
plt.show()