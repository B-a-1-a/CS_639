## Bug 1: Redundant Random Seed Resets Causing Poor Training Dynamics

**Output/Error Printout Showcasing the Issue:**
```
Epoch 1/10 - Loss: 1.0984
Epoch 2/10 - Loss: 1.0984
...
Epoch 9/10 - Loss: 1.0974
Epoch 10/10 - Loss: 1.0970
```
*(Training loss for LR=0.01 would steadily decrease to ~0.75, but then repeatedly jump back up to ~1.15 in a cyclical pattern without converging cleanly. Additionally, LR=1.0 showed an upward trend instead of oscillating.)*

**Cause:**
The `np.random.seed(0)` function was being called redundantly in multiple locations: inside the `NeuralNetwork.__init__()` constructor and again at the start of the `train()` method. This meant that the random state was being reset to `0` right before the initial dataset split, again right before initializing the model weights, and yet again at the very beginning of the training loop. This caused the mini-batch shuffling sequence during the first epoch to perfectly mirror the random sequence used to split the training and testing data, leading to degenerate and correlated mini-batches that prevented the model from learning generalizable patterns smoothly. 

**Debugging Process:**
1. I noticed that while the loss for `LR=0.01` was decreasing initially, it would exhibit strange, large jumps back up to near-initial loss values (e.g., jumping from 0.75 back to 1.15) instead of converging smoothly.
2. Similarly, for `LR=1.0`, the expected behavior was violent oscillation, but the model instead showed a steady upward trend in loss.
3. This cyclical instability pointed towards an issue with how the data was being presented to the model across epochs.
4. Reviewing the assignment specification ("set `np.random.seed(0)` prior to training and then initialize weights"), I realized the seed should only be set *once* before constructing each model. 
5. I removed the `np.random.seed(0)` calls from inside the `__init__` and `train` methods.
6. Instead, I placed a single `np.random.seed(0)` call immediately before the `model = NeuralNetwork(...)` instantiation in the main testing loops. This ensured all models started with identical initial weights, but allowed the NumPy random state to naturally flow into the training loop's `np.random.shuffle()` calls, producing proper, uncorrelated mini-batches that allowed the network to converge correctly.
