import pandas as pd
from scipy.optimize import linprog
import kagglehub
import os

# Load the csv file into a pandas DataFrame
df = pd.read_csv("data-2 - copia.csv")
df.head()


# Preprocessing
df = df.dropna()  # Remove missing values
grouped = df.groupby('StockCode').agg({'Quantity': 'sum', 'UnitPrice': 'mean'}).reset_index()
grouped.columns = ['Product', 'Demand', 'Price']


# Assume production cost and competitor price
grouped['Cost'] = grouped['Price'] * 0.6  # Production cost = 60% of average price
grouped['Competitor'] = grouped['Price'] * 1.1  # Competitor price = 10% higher



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
    print(grouped[['Product', 'Demand', 'Price', 'Cost', 'Competitor', 'Optimized_Price']])
else:
    print("Optimization failed:", result.message)
