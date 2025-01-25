import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
dfX = pd.read_csv('logisticX.csv', header=None).values
dfY = pd.read_csv('logisticY.csv', header=None).values

test_ratio = 0.20
train_size = int((1 - test_ratio) * dfX.shape[0])
X_train = dfX[:train_size]
X_test = dfX[train_size:]
y_train = dfY[:train_size].flatten()
y_test = dfY[train_size:].flatten()

# Normalize the data
X_mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

def Zscore(X):
    return (X - X_mean) / std

X_train = Zscore(X_train)
X_test = Zscore(X_test)

def initialize_weights(n_features):
    return np.zeros(n_features), 0.0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logloss(y_true, y_pred):
    n = y_true.shape[0]
    epsilon = 1e-10  # To avoid log(0)
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

def pred(w, b, X):
    return sigmoid(np.dot(X, w) + b)

def train(X_train, y_train, epochs, eta0):
    n_features = X_train.shape[1]
    w, b = initialize_weights(n_features)
    train_loss = []

    for epoch in range(epochs):
        # Calculate predictions for the entire batch
        y_pred = pred(w, b, X_train)

        # Compute batch gradients
        dw = np.dot((y_train - y_pred), X_train) / X_train.shape[0]
        db = np.sum(y_train - y_pred) / X_train.shape[0]

        # Update weights and bias
        w += eta0 * dw
        b += eta0 * db

        # Record training loss
        train_loss.append(logloss(y_train, y_pred))

        # Convergence check
        if epoch > 0 and abs(train_loss[-1] - train_loss[-2]) < 1e-7:
            break

    return w, b, train_loss

# Task 1: Train with learning rate = 0.1
eta0 = 0.1
epochs = 100
w_1, b_1, train_loss_1 = train(X_train, y_train, epochs, eta0)

print(f"Task 1: Weights: {w_1}, Bias: {b_1}")
final_cost = train_loss_1[-1]
print(f"Final Cost After Convergence: {final_cost}")

# Task 2: Plot Cost Function vs Iterations
plt.plot(train_loss_1)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Function vs Iterations (Learning Rate = 0.1)')
plt.show()

# Task 3: Plot Dataset and Decision Boundary
def plot_decision_boundary(w, b):
    x_vals = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, label='Decision Boundary', color='red')

plt.scatter(X_train[:, 0][y_train == 0], X_train[:, 1][y_train == 0], label='Class 0', alpha=0.7)
plt.scatter(X_train[:, 0][y_train == 1], X_train[:, 1][y_train == 1], label='Class 1', alpha=0.7)
plot_decision_boundary(w_1, b_1)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Dataset with Decision Boundary')
plt.legend()
plt.show()

# Task 4: Train with Learning Rates 0.1 and 5
eta1, eta2 = 0.1, 5
w_1, b_1, train_loss_1 = train(X_train, y_train, 100, eta1)
w_2, b_2, train_loss_2 = train(X_train, y_train, 100, eta2)

plt.plot(train_loss_1, label='Learning Rate = 0.1')
plt.plot(train_loss_2, label='Learning Rate = 5')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Function vs Iterations for Different Learning Rates')
plt.legend()
plt.show()

# Task 5: Confusion Matrix and Metrics
def compute_confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def compute_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1_score

def evaluate_model(w, b, X, y):
    y_pred = (pred(w, b, X) >= 0.5).astype(int)
    cm = compute_confusion_matrix(y, y_pred)
    accuracy, precision, recall, f1 = compute_metrics(cm)
    return cm, accuracy, precision, recall, f1

cm, acc, prec, recall, f1 = evaluate_model(w_1, b_1, X_train, y_train)

print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
