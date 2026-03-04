import os
import struct
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, task_type='classification', normalize_gradient=False):
        """
        Initialize the neural network with given sizes.
        Includes bias units implicitly via b1 and b2 parameters.
        Weights are initialized from U(-0.01, 0.01).
        Caller must set np.random.seed(0) prior to construction.
        """
        self.task_type = task_type
        self.normalize_gradient = normalize_gradient
        # We represent the bias units as separate vectors b1 and b2, 
        # which is mathematically equivalent to having a fixed +1 input feature.
        self.W1 = np.random.uniform(-0.01, 0.01, (input_size, hidden_size))
        self.b1 = np.random.uniform(-0.01, 0.01, (hidden_size,))
        
        self.W2 = np.random.uniform(-0.01, 0.01, (hidden_size, output_size))
        self.b2 = np.random.uniform(-0.01, 0.01, (output_size,))

    def forward(self, X):
        """
        Perform a forward pass through the network.
        X: (N, input_size)
        Returns Z2 (logits) and a cache of intermediate values for the backward pass.
        """
        # Linear step 1
        Z1 = np.dot(X, self.W1) + self.b1
        
        # ReLU activation
        A1 = np.maximum(0, Z1)
        
        # Linear step 2 (No activation function per requirements)
        Z2 = np.dot(A1, self.W2) + self.b2
        
        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2
        }
        
        return Z2, cache
    
    def softmax(self, Z):
        """
        Compute softmax values for each set of scores in Z.
        Numerically stable by subtracting the max logit.
        """
        # Z shape is (N, output_size)
        # Shift scores for numerical stability
        shifted_Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(shifted_Z)
        probabilities = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return probabilities
    
    def compute_loss(self, Y_pred, Y_true):
        """
        Compute categorical cross-entropy loss or MSE loss.
        """
        if self.task_type == 'classification':
            N = Y_true.shape[0]
            # Clip Y_pred to avoid log(0)
            epsilon = 1e-15
            Y_pred_clipped = np.clip(Y_pred, epsilon, 1 - epsilon)
            loss = -np.sum(Y_true * np.log(Y_pred_clipped)) / N
            return loss
        elif self.task_type == 'regression':
            return np.mean((Y_pred - Y_true) ** 2)

    def backward(self, cache, Y_true):
        """
        Compute gradients for the network parameters.
        cache: Dictionary containing intermediate values from forward pass
        Y_true: (N, output_size) one-hot encoded true labels
        """
        N = Y_true.shape[0]
        
        X = cache["X"]
        Z1 = cache["Z1"]
        A1 = cache["A1"]
        Z2 = cache["Z2"]
        
        # Output layer gradients
        if self.task_type == 'classification':
            A2 = self.softmax(Z2)
            dZ2 = A2 - Y_true # (N, output_size)
            if self.normalize_gradient:
                dZ2 = dZ2 / N
        elif self.task_type == 'regression':
            dZ2 = 2 * (Z2 - Y_true) / N # (N, output_size)
        
        dW2 = np.dot(A1.T, dZ2) # (hidden_size, output_size)
        db2 = np.sum(dZ2, axis=0) # (output_size,)
        
        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T) # (N, hidden_size)
        
        # Derivative of ReLU
        dZ1 = dA1 * (Z1 > 0).astype(float) # (N, hidden_size)
        
        dW1 = np.dot(X.T, dZ1) # (input_size, hidden_size)
        db1 = np.sum(dZ1, axis=0) # (hidden_size,)
        
        grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2
        }
        
        return grads
    
    def update_parameters(self, grads, learning_rate):
        """
        Update model parameters using simple SGD.
        Applies gradient clipping for regression to prevent overflow.
        """
        if self.task_type == 'regression':
            max_norm = 10.0
            for key in grads:
                norm = np.linalg.norm(grads[key])
                if norm > max_norm:
                    grads[key] = grads[key] * max_norm / norm
        
        self.W1 -= learning_rate * grads["W1"]
        self.b1 -= learning_rate * grads["b1"]
        self.W2 -= learning_rate * grads["W2"]
        self.b2 -= learning_rate * grads["b2"]
    
    def train(self, X_train, Y_train, epochs, learning_rate, batch_size=32):
        """
        Train the network using mini-batch SGD.
        Reshuffle training data at the start of each epoch.
        """
        N = X_train.shape[0]
        history = {'loss': []}
        
        for epoch in range(epochs):
            # Reshuffle at the start of each epoch
            indices = np.arange(N)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]
            
            # Mini-batch SGD
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                X_batch = X_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]
                
                # Forward pass
                Z2, cache = self.forward(X_batch)
                
                # Backward pass
                grads = self.backward(cache, Y_batch)
                
                # Update parameters
                self.update_parameters(grads, learning_rate)
            
            # Evaluate average train loss for this epoch
            Z2_full, _ = self.forward(X_train)
            if self.task_type == 'classification':
                A2_full = self.softmax(Z2_full)
                epoch_loss = self.compute_loss(A2_full, Y_train)
            elif self.task_type == 'regression':
                epoch_loss = self.compute_loss(Z2_full, Y_train)
            history['loss'].append(epoch_loss)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
            
        return history
    
    def predict(self, X):
        """
        Predict class labels or output values for given inputs.
        """
        Z2, _ = self.forward(X)
        if self.task_type == 'classification':
            A2 = self.softmax(Z2)
            return np.argmax(A2, axis=1)
        elif self.task_type == 'regression':
            return Z2

def question_1():
    os.makedirs("plots", exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = pd.read_csv(url, header=None)
    
    # Drop any rows with missing values just in case
    df = df.dropna()
    
    X = df.iloc[:, :4].values
    y_labels = df.iloc[:, 4].values
    
    # One-hot encode labels
    unique_labels = np.unique(y_labels)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    y_indices = np.array([label_to_idx[label] for label in y_labels])
    
    Y = np.zeros((y_indices.size, len(unique_labels)))
    Y[np.arange(y_indices.size), y_indices] = 1
    
    # 80/20 train/test split
    np.random.seed(0)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(0.8 * X.shape[0])
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    output_size = len(unique_labels)
    
    print(f"Train set: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Test set: X={X_test.shape}, Y={Y_test.shape}")
    
    print("\n--- Parts A, B, C ---")
    learning_rates = [1, 1e-2, 1e-3, 1e-8]
    hidden_size = 5
    input_size = X_train.shape[1]
    
    plt.figure(figsize=(10, 6))
    
    models_abc = {}
    
    for lr in learning_rates:
        np.random.seed(0)
        model = NeuralNetwork(input_size, hidden_size, output_size)
        history = model.train(X_train, Y_train, epochs=10, learning_rate=lr)
        models_abc[lr] = model
        
        plt.plot(range(1, 11), history['loss'], label=f'LR = {lr}')
        
    plt.title("Average Training Loss per Epoch by Learning Rate (Hidden Size = 5)")
    plt.xlabel("Epoch")
    plt.ylabel("Average Cross-Entropy Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/question_1_part_a.png")
    print("Saved plot for Part A: plots/question_1_part_a.png")
    
    print("\nPart B / C Results:")
    for lr, model in models_abc.items():
        # Test loss
        Z2_test, _ = model.forward(X_test)
        A2_test = model.softmax(Z2_test)
        test_loss = model.compute_loss(A2_test, Y_test)
        
        # Test accuracy
        predictions = np.argmax(A2_test, axis=1)
        true_labels = np.argmax(Y_test, axis=1)
        accuracy = np.mean(predictions == true_labels)
        
        print(f"LR: {lr} | Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy:.4f}")
        
    print("\n--- Parts D, E ---")
    hidden_sizes = [2, 8, 16, 32]
    lr_de = 1e-2
    
    plt.figure(figsize=(10, 6))
    
    models_de = {}
    
    for hs in hidden_sizes:
        np.random.seed(0)
        model = NeuralNetwork(input_size, hs, output_size)
        history = model.train(X_train, Y_train, epochs=10, learning_rate=lr_de)
        models_de[hs] = model
        
        plt.plot(range(1, 11), history['loss'], label=f'Hidden Size = {hs}')
        
    plt.title("Average Training Loss per Epoch by Hidden Size (LR = 1.0e-02)")
    plt.xlabel("Epoch")
    plt.ylabel("Average Cross-Entropy Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/question_1_part_d.png")
    print("Saved plot for Part D: plots/question_1_part_d.png")
    
    print("\nPart E Results:")
    for hs, model in models_de.items():
        # Test loss
        Z2_test, _ = model.forward(X_test)
        A2_test = model.softmax(Z2_test)
        test_loss = model.compute_loss(A2_test, Y_test)
        
        # Test accuracy
        predictions = np.argmax(A2_test, axis=1)
        true_labels = np.argmax(Y_test, axis=1)
        accuracy = np.mean(predictions == true_labels)
        
        print(f"Hidden Size: {hs} | Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy:.4f}")

def question_2():
    os.makedirs("plots", exist_ok=True)
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)
    
    # Drop any rows with missing values
    df = df.dropna()
    
    # Process categorical column 'ocean_proximity'
    df = pd.get_dummies(df, columns=['ocean_proximity'])
    
    # Target variable as float
    y = df['median_house_value'].values.reshape(-1, 1).astype(float)
    X = df.drop('median_house_value', axis=1).values.astype(float)
    
    # 80/20 train/test split
    np.random.seed(0)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(0.8 * X.shape[0])
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Standardize input features to zero mean and variance of 1
    mean_X = np.mean(X_train, axis=0)
    std_X = np.std(X_train, axis=0)
    std_X[std_X == 0] = 1.0 # prevent division by zero
    
    X_train = (X_train - mean_X) / std_X
    X_test = (X_test - mean_X) / std_X
    
    # Also standardize the target to prevent gradient explosion
    mean_y = np.mean(y_train)
    std_y = np.std(y_train)
    
    y_train_scaled = (y_train - mean_y) / std_y
    y_test_scaled = (y_test - mean_y) / std_y
    
    output_size = 1
    input_size = X_train.shape[1]
    
    print(f"\n--- Question 2 ---")
    print(f"Train set: X={X_train.shape}, Y={y_train.shape}")
    print(f"Test set: X={X_test.shape}, Y={y_test.shape}")
    
    print("\n--- Parts A, B, C ---")
    learning_rates = [1, 1e-2, 1e-3, 1e-8]
    hidden_size = 5
    
    plt.figure(figsize=(10, 6))
    
    models_abc = {}
    
    for lr in learning_rates:
        np.random.seed(0)
        model = NeuralNetwork(input_size, hidden_size, output_size, task_type='regression')
        history = model.train(X_train, y_train_scaled, epochs=10, learning_rate=lr)
        models_abc[lr] = model
        
        plt.plot(range(1, 11), history['loss'], label=f'LR = {lr}')
        
    plt.title("Q2: Average Training Loss per Epoch by Learning Rate (Hidden Size = 5)")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/question_2_part_a.png")
    print("Saved plot for Part A: plots/question_2_part_a.png")
    
    print("\nPart B / C Results (standardized scale):")
    for lr, model in models_abc.items():
        # Test MSE in standardized scale
        y_test_pred_scaled = model.predict(X_test)
        test_mse = model.compute_loss(y_test_pred_scaled, y_test_scaled)
        
        print(f"LR: {lr} | Test MSE: {test_mse:.6f}")
        
    print("\n--- Parts D, E ---")
    hidden_sizes = [2, 8, 16, 32]
    lr_de = 1e-2
    
    plt.figure(figsize=(10, 6))
    
    models_de = {}
    
    for hs in hidden_sizes:
        np.random.seed(0)
        model = NeuralNetwork(input_size, hs, output_size, task_type='regression')
        history = model.train(X_train, y_train_scaled, epochs=10, learning_rate=lr_de)
        models_de[hs] = model
        
        plt.plot(range(1, 11), history['loss'], label=f'Hidden Size = {hs}')
        
    plt.title("Q2: Average Training Loss per Epoch by Hidden Size (LR = 1.0e-02)")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/question_2_part_d.png")
    print("Saved plot for Part D: plots/question_2_part_d.png")
    
    print("\nPart E Results (standardized scale):")
    for hs, model in models_de.items():
        # Test MSE in standardized scale
        y_test_pred_scaled = model.predict(X_test)
        test_mse = model.compute_loss(y_test_pred_scaled, y_test_scaled)
        
        print(f"Hidden Size: {hs} | Test MSE: {test_mse:.6f}")

def question_3():
    os.makedirs("plots", exist_ok=True)
    data_dir = "q3Data"

    # --- Helper functions for reading MNIST IDX files (gzipped) ---
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            images = data.reshape(num_images, rows, cols)
        return images

    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    X_train = load_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    y_train = load_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    X_test = load_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    y_test = load_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    # Flatten 28x28 images to 784-element vectors
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Normalize to [0, 1]
    X_train = X_train.astype(float) / 255.0
    X_test = X_test.astype(float) / 255.0

    # Use a subset of 20,000 training examples for runtime efficiency
    np.random.seed(0)
    subset_indices = np.random.choice(X_train.shape[0], 20000, replace=False)
    X_train = X_train[subset_indices]
    y_train = y_train[subset_indices]

    # One-hot encode labels (10 classes)
    num_classes = 10
    Y_train = np.zeros((y_train.size, num_classes))
    Y_train[np.arange(y_train.size), y_train] = 1

    Y_test = np.zeros((y_test.size, num_classes))
    Y_test[np.arange(y_test.size), y_test] = 1

    input_size = 784
    output_size = num_classes

    print(f"\n--- Question 3 (MNIST) ---")
    print(f"Train set: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Test set: X={X_test.shape}, Y={Y_test.shape}")

    # --- Parts A, B, C: Vary learning rate with hidden_size = 5 ---
    print("\n--- Parts A, B, C ---")
    learning_rates = [1, 1e-2, 1e-3, 1e-8]
    hidden_size = 5

    plt.figure(figsize=(10, 6))

    models_abc = {}

    for lr in learning_rates:
        np.random.seed(0)
        model = NeuralNetwork(input_size, hidden_size, output_size, normalize_gradient=True)
        history = model.train(X_train, Y_train, epochs=10, learning_rate=lr)
        models_abc[lr] = model

        plt.plot(range(1, 11), history['loss'], label=f'LR = {lr}')

    plt.title("Q3 (MNIST): Average Training Loss per Epoch by Learning Rate (Hidden Size = 5)")
    plt.xlabel("Epoch")
    plt.ylabel("Average Cross-Entropy Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/question_3_part_a.png")
    print("Saved plot for Part A: plots/question_3_part_a.png")

    print("\nPart B / C Results:")
    for lr, model in models_abc.items():
        # Test loss (no parameter updates)
        Z2_test, _ = model.forward(X_test)
        A2_test = model.softmax(Z2_test)
        test_loss = model.compute_loss(A2_test, Y_test)

        # Test accuracy
        predictions = np.argmax(A2_test, axis=1)
        true_labels = np.argmax(Y_test, axis=1)
        accuracy = np.mean(predictions == true_labels)

        print(f"LR: {lr} | Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy:.4f}")

    # --- Parts D, E: Vary hidden size with LR = 1e-2 ---
    print("\n--- Parts D, E ---")
    hidden_sizes = [2, 8, 16, 32]
    lr_de = 1e-2

    plt.figure(figsize=(10, 6))

    models_de = {}

    for hs in hidden_sizes:
        np.random.seed(0)
        model = NeuralNetwork(input_size, hs, output_size, normalize_gradient=True)
        history = model.train(X_train, Y_train, epochs=10, learning_rate=lr_de)
        models_de[hs] = model

        plt.plot(range(1, 11), history['loss'], label=f'Hidden Size = {hs}')

    plt.title("Q3 (MNIST): Average Training Loss per Epoch by Hidden Size (LR = 1.0e-02)")
    plt.xlabel("Epoch")
    plt.ylabel("Average Cross-Entropy Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/question_3_part_d.png")
    print("Saved plot for Part D: plots/question_3_part_d.png")

    print("\nPart E Results:")
    for hs, model in models_de.items():
        # Test loss
        Z2_test, _ = model.forward(X_test)
        A2_test = model.softmax(Z2_test)
        test_loss = model.compute_loss(A2_test, Y_test)

        # Test accuracy
        predictions = np.argmax(A2_test, axis=1)
        true_labels = np.argmax(Y_test, axis=1)
        accuracy = np.mean(predictions == true_labels)

        print(f"Hidden Size: {hs} | Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy:.4f}")


def main():
    question_1()
    question_2()
    question_3()

if __name__ == "__main__":
    main()
