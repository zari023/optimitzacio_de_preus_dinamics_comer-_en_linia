import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import linprog

# Example DataFrame for illustration
df = pd.read_csv("data-2 - copia.csv")  # Make sure to load your DataFrame here

# Preprocess DataFrame
df = df.dropna()  # Remove missing values
grouped = df.groupby('StockCode').agg({'Quantity': 'sum', 'UnitPrice': 'mean'}).reset_index()
grouped.columns = ['Product', 'Demand', 'Price']

# Assume fixed production cost and competitor price (not dependent on our price)
grouped['Cost'] = 5.0  # Fixed production cost, e.g., $5
grouped['Competitor'] = grouped['Demand'] * 0.02 + 10  # Competitor price, e.g., based on demand and a base price

# Ensure feasibility by filtering rows where Cost > Competitor
grouped = grouped[grouped['Cost'] <= grouped['Competitor']]

# Optimization inputs
D = grouped['Demand'].values  # Demand
C = grouped['Competitor'].values  # Competitor prices
X = grouped['Cost'].values  # Production costs

# Objective Function: Maximize sum(D_i * P_i) --> Minimize -sum(D_i * P_i)
c = -D  # Negate for minimization

# Constraints
A_ub = []
b_ub = []

# Constraint: P_i <= C_i (competitor prices)
A_ub.extend([[1 if j == i else 0 for j in range(len(D))] for i in range(len(D))])
b_ub.extend(C)

# Constraint: P_i >= X_i (production cost) --> -P_i <= -X_i
A_ub.extend([[-1 if j == i else 0 for j in range(len(D))] for i in range(len(D))])
b_ub.extend(-X)

# Bounds for P_i (prices must be non-negative)
bounds = [(0, None) for _ in range(len(D))]

# Solve the linear program
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# Results
if result.success:
    grouped['Optimized_Price'] = result.x
    grouped = grouped.head(5)  # Use only the first 5 products
    print(grouped[['Product', 'Demand', 'Price', 'Cost', 'Competitor', 'Optimized_Price']])

    # Plotting a bar chart
    fig, ax = plt.subplots(figsize=(12, 7))

    # Bar chart for Original Price, Competitor Price, and Optimized Price
    width = 0.25  # Width of each bar
    index = range(len(grouped))  # Position of each group of bars

    # Plot bars for each type of price
    ax.bar(index, grouped['Price'], width, label='Original Price', color='blue')
    ax.bar([i + width for i in index], grouped['Competitor'], width, label='Competitor Price', color='red')
    ax.bar([i + 2 * width for i in index], grouped['Optimized_Price'], width, label='Optimized Price', color='green')

    # Add labels and title
    ax.set_xlabel('Product')
    ax.set_ylabel('Price')
    ax.set_title('Comparison of Prices (First 5 Products): Original, Competitor, and Optimized')

    # Set x-ticks to the product names and rotate them
    ax.set_xticks([i + width for i in index])  # Center the x-tick labels
    ax.set_xticklabels(grouped['Product'], rotation=45, ha='right')

    # Add legend
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

else:
    print("Optimization failed:", result.message)
