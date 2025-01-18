import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data loading and preprocessing
dfX = pd.read_csv('linearX.csv',header=None).values
dfY = pd.read_csv('linearY.csv',header=None).values

# Required normalization as per assignment note 7
X_mean = np.mean(dfX)
X_std = np.std(dfX)
X_norm = (dfX - X_mean) / X_std


m = len(dfX)
iters = 100
batch_size = 10

def predict(X, t0, t1):
    return t0 + t1 * X

def compute_cost(X, y, t0, t1):
    preds = predict(X, t0, t1)
    return np.sum((preds - y) ** 2) / (2 * len(X))  # Question 2: Cost is averaged by len(X)


p = np.sum(1)
type(p)


# Question 1: Batch Gradient Descent with learning rate 0.5
def batch_gd(X, y, lr=0.5):
    t0, t1 = 0, 0
    costs = []
    
    for _ in range(iters):
        preds = predict(X, t0, t1)
        d_t0 = np.sum(preds - y) / m
        d_t1 = np.sum((preds - y) * X) / m
        t0 -= lr * d_t0
        t1 -= lr * d_t1
        costs.append(compute_cost(X, y, t0, t1))
    
    return t0, t1, costs



# Question 6: Stochastic Gradient Descent
def sgd(X, y, lr=0.5):
    t0, t1 = 0, 0
    costs = []
    
    for _ in range(iters):
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]
        
        for i in range(m):
            pred = predict(X[i], t0, t1)
            d_t0 = pred - y[i]
            d_t1 = (pred - y[i]) * X[i]
            t0 -= lr * d_t0
            t1 -= lr * d_t1
            
        costs.append(compute_cost(X, y, t0, t1))
    
    return t0, t1, costs



# Question 6: Mini-Batch Gradient Descent
def mini_batch_gd(X, y, lr=0.5):
    t0, t1 = 0, 0
    costs = []
    
    for _ in range(iters):
        indices = np.random.permutation(m)
        X_shuf = X[indices]
        y_shuf = y[indices]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuf[i:i+batch_size]
            y_batch = y_shuf[i:i+batch_size]
            preds = predict(X_batch, t0, t1)
            d_t0 = np.sum(preds - y_batch) / batch_size
            d_t1 = np.sum((preds - y_batch) * X_batch) / batch_size
            t0 -= lr * d_t0
            t1 -= lr * d_t1
            
        costs.append(compute_cost(X, y, t0, t1))
    
    return t0, t1, costs



# Question 1: Results
t0_b, t1_b, costs_b = batch_gd(X_norm, dfY)
print(f"After convergence: Theta_0: {t0_b:.4f}, Theta_1: {t1_b:.4f}")
print(f"Final Cost: {costs_b[-1]:.9f}")

# Question 3: Plot cost function vs iterations
plt.figure(figsize=(10, 6))
plt.plot(range(iters), costs_b)
plt.title('Cost Function vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()



# Question 4: Plot dataset and regression line
plt.figure(figsize=(10, 6))
plt.scatter(dfX, dfY, color='blue', label='Data Points')
regression_line = predict(X_norm, t0_b, t1_b)
plt.plot(dfX, regression_line, color='red', label='Regression Line')
plt.title('Linear Regression Fit')
plt.xlabel('Predictor')
plt.ylabel('Response')
plt.legend()
plt.show()



# Question 5: Learning rate comparison
plt.figure(figsize=(10, 6))
learning_rates = [0.005, 0.5, 5]
for lr in learning_rates:
    t0, t1 = 0, 0
    costs = []
    for _ in range(iters):
        preds = predict(X_norm, t0, t1)
        d_t0 = np.sum(preds - dfY) / m
        d_t1 = np.sum((preds - dfY) * X_norm) / m
        t0 -= lr * d_t0
        t1 -= lr * d_t1
        costs.append(compute_cost(X_norm, dfY, t0, t1))
    plt.plot(range(iters), costs, label=f'lr={lr}')

plt.title('Cost Function vs Iterations for Different Learning Rates')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.legend()
plt.show()



# Question 6: Run different gradient descent methods and cost plots
t0_s, t1_s, costs_s = sgd(X_norm, dfY)
t0_m, t1_m, costs_m = mini_batch_gd(X_norm, dfY)
plt.figure(figsize=(10, 6))
plt.plot(range(iters), costs_b, label='Batch GD', color='blue')
plt.plot(range(iters), costs_s, label='Stochastic GD', color='green')
plt.plot(range(iters), costs_m, label='Mini-Batch GD', color='red')
plt.title('Cost Function Comparison')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.legend()
plt.show()

