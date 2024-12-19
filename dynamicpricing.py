import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import linprog
import numpy as np

# Load the DataFrame
df = pd.read_csv("data-2 - copia.csv").dropna()

# Preprocess DataFrame
grouped = df.groupby('StockCode').agg({'Quantity': 'sum', 'UnitPrice': 'mean'}).reset_index()
grouped.columns = ['Product', 'Demand', 'Price']

# Assume fixed production cost
grouped['Cost'] = 5.0
base_prices = grouped['Price'].values
base_demand = grouped['Demand'].values

# Setup
alpha = 0.1
epsilon = 0.1  # Adjusted elasticity
beta = 0.02  # Adjusted competitor price adjustment rate
iterations = 5

# Initialize competitor prices dynamically
grouped['Competitor'] = np.maximum(grouped['Demand'] * 0.02 + 10, grouped['Cost'] + 1.5)

# Track results
results = []
ingresos_per_iteration = []

for iteration in range(iterations):
    grouped['Competitor'] = np.maximum(grouped['Competitor'], grouped['Cost'] + 1.5)
    D = grouped['Demand'].values
    C = grouped['Competitor'].values
    X = grouped['Cost'].values
    c = -D

    A_ub = [[1 if j == i else 0 for j in range(len(D))] for i in range(len(D))]
    b_ub = list(C)
    A_ub += [[-1 if j == i else 0 for j in range(len(D))] for i in range(len(D))]
    b_ub += [-X[i] for i in range(len(X))]
    A_ub += [[1 + alpha if j == i else 0 for j in range(len(D))] for i in range(len(D))]
    b_ub += list(np.maximum(C + alpha * X, X + 0.1))

    bounds = [(X[i] + 0.1, C[i] + 10) for i in range(len(D))]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if result.success:
        grouped['Optimized_Price'] = result.x
        results.append(grouped[['Product', 'Demand', 'Competitor', 'Optimized_Price']].copy())
        ingresos = (grouped['Demand'] * grouped['Optimized_Price']).sum()
        print(f"Ingressos at iteration {iteration + 1}: {ingresos:.2f}")
        ingresos_per_iteration.append(ingresos)
        grouped['Demand'] = np.maximum(grouped['Demand'] * (1 - epsilon * (grouped['Optimized_Price'] - base_prices) / base_prices), 5)
        grouped['Competitor'] += beta * (grouped['Demand'] - base_demand)
    else:
        print(f"Optimization failed at iteration {iteration + 1}")
        break

# Plot ingresos per iteration
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(ingresos_per_iteration) + 1), ingresos_per_iteration, marker='o', color='purple')
plt.title("Ingresos per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Ingresos")
plt.xticks(range(1, len(ingresos_per_iteration) + 1))
plt.grid(True)
plt.tight_layout()
plt.show()