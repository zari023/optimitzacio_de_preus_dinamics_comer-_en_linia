
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import linprog
import numpy as np

# Load the DataFrame
df = pd.read_csv("data-2 - copia.csv").dropna()

# Preprocess DataFrame
grouped = df.groupby('StockCode').agg({'Quantity': 'sum', 'UnitPrice': 'mean'}).reset_index()
grouped.columns = ['Product', 'Demand', 'Price']

# Assume fixed production cost and competitor price
grouped['Cost'] = 5.0  # Fixed production cost
grouped['Competitor'] = grouped['Demand'] * 0.02 + 10  # Competitor price
grouped = grouped[grouped['Cost'] <= grouped['Competitor']]  # Feasibility check

# Elasticity-based constraint setup
alpha = 0.1  # Elasticity factor: higher alpha reduces optimized price variability

# Optimization inputs
D = grouped['Demand'].values  # Demand
C = grouped['Competitor'].values  # Competitor prices
X = grouped['Cost'].values  # Production costs

# Objective Function: Maximize sum(D_i * P_i) --> Minimize -sum(D_i * P_i)
c = -D  # Negate for minimization

# Constraints
A_ub = []  # Upper bound constraint matrix
b_ub = []  # Upper bound values

# Constraint: P_i <= C_i (competitor prices)
A_ub.extend([[1 if j == i else 0 for j in range(len(D))] for i in range(len(D))])
b_ub.extend(C)

# Constraint: P_i >= X_i --> -P_i <= -X_i
A_ub.extend([[-1 if j == i else 0 for j in range(len(D))] for i in range(len(D))])
b_ub.extend(-X)

# Elastic Demand Constraint: (1 + alpha) * P_i <= C_i + alpha * X_i
A_ub.extend([[1 + alpha if j == i else 0 for j in range(len(D))] for i in range(len(D))])
b_ub.extend(C + alpha * X)

# Bounds for P_i (prices must be non-negative)
bounds = [(0, None) for _ in range(len(D))]

# Solve the linear program
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# Results
if result.success:
    grouped['Optimized_Price'] = result.x
    grouped = grouped.head(5)  # Use only the first 5 products for display
    print(grouped[['Product', 'Demand', 'Price', 'Cost', 'Competitor', 'Optimized_Price']])

    # Plotting a bar chart
    fig, ax = plt.subplots(figsize=(12, 7))

    # Bar chart for Original Price, Competitor Price, and Optimized Price
    width = 0.25
    index = range(len(grouped))

    ax.bar(index, grouped['Price'], width, label='Original Price', color='blue')
    ax.bar([i + width for i in index], grouped['Competitor'], width, label='Competitor Price', color='red')
    ax.bar([i + 2 * width for i in index], grouped['Optimized_Price'], width, label='Optimized Price', color='green')

    # Add labels and title
    ax.set_xlabel('Product')
    ax.set_ylabel('Price')
    ax.set_title('Comparison of Prices (First 5 Products): Original, Competitor, and Optimized')

    # Set x-ticks
    ax.set_xticks([i + width for i in index])
    ax.set_xticklabels(grouped['Product'], rotation=45, ha='right')

    ax.legend()
    plt.tight_layout()
    plt.show()

else:
    print("Optimization failed:", result.message)

