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
epsilon = 0.3
beta = 0.05
iterations = 5

# Initialize competitor prices dynamically
grouped['Competitor'] = np.maximum(grouped['Demand'] * 0.02 + 10, grouped['Cost'] + 1.5)

# Track results
results = []

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
        grouped['Demand'] = np.maximum(grouped['Demand'] * (1 - epsilon * (grouped['Optimized_Price'] - base_prices) / base_prices), 5)
        grouped['Competitor'] += beta * (grouped['Demand'] - base_demand)
    else:
        print(f"Optimization failed at iteration {iteration + 1}")
        break

# Plot Iteration-Level Graphs
for iteration, res in enumerate(results):
    top_products = res.head(5)  # Focus on top 5 products
    fig, ax = plt.subplots(figsize=(10, 6))

    # Width and position of bars
    width = 0.25
    indices = range(len(top_products))

    # Plot bars for Original, Competitor, and Optimized Prices
    bars1 = ax.bar(indices, base_prices[:5], width, label='Original Price', color='blue')
    bars2 = ax.bar([i + width for i in indices], top_products['Competitor'], width, label='Competitor Price', color='red')
    bars3 = ax.bar([i + 2 * width for i in indices], top_products['Optimized_Price'], width, label='Optimized Price', color='green')

    # Add labels (price values) on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

    for bar in bars3:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

    # Customize plot
    ax.set_xticks([i + width for i in indices])
    ax.set_xticklabels(top_products['Product'], rotation=45, ha='right')
    ax.set_title(f"Iteration {iteration + 1}: Original, Competitor, and Optimized Prices")
    ax.set_xlabel('Product')
    ax.set_ylabel('Price')
    ax.legend()
    plt.tight_layout()
    plt.show()


# Plot Product-Level Graphs
unique_products = results[0]['Product'].head(5)  # Focus on first 5 products
for product in unique_products:
    fig, ax = plt.subplots(figsize=(10, 6))
    demand = [res.loc[res['Product'] == product, 'Demand'].values[0] for res in results]
    competitor = [res.loc[res['Product'] == product, 'Competitor'].values[0] for res in results]
    optimized = [res.loc[res['Product'] == product, 'Optimized_Price'].values[0] for res in results]
    iterations_range = range(1, len(results) + 1)

    ax.plot(iterations_range, demand, label='Demand', marker='o', color='blue')
    ax.plot(iterations_range, competitor, label='Competitor Price', marker='o', color='red')
    ax.plot(iterations_range, optimized, label='Optimized Price', marker='o', color='green')

    ax.set_title(f"Product {product}: Demand, Competitor Price, and Optimized Price Over Iterations")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.legend()
    plt.tight_layout()
    plt.show()


