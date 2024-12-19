import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import linprog
import numpy as np

# Load the DataFrame
df = pd.read_csv("data-2 - copia.csv").dropna()

# Preprocess DataFrame
grouped = df.groupby('StockCode').agg({'Quantity': 'sum', 'UnitPrice': 'mean'}).reset_index()
grouped.columns = ['Product', 'Demand', 'Price']
# Adding day-based features to the dataset
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])  # Ensure InvoiceDate is in datetime format
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek  # Extract day of the week (0=Monday, 6=Sunday)
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)  # Flag for weekends

# Aggregate data by StockCode and day of the week
grouped_by_day = df.groupby(['StockCode', 'DayOfWeek']).agg({
    'Quantity': 'sum',
    'UnitPrice': 'mean',
    'IsWeekend': 'mean'  # Average to retain weekend vs. weekday insight
}).reset_index()

# Rename columns for clarity
grouped_by_day.columns = ['Product', 'DayOfWeek', 'Demand', 'Price', 'IsWeekend']

# Check the updated DataFrame
grouped_by_day.head()
# Assume fixed production cost
# Add a dynamic pricing adjustment for each product by day
grouped_by_day['Cost'] = 5.0  # Fixed production cost
base_prices_day = grouped_by_day['Price'].values
base_demand_day = grouped_by_day['Demand'].values

# Initialize dynamic competitor prices
grouped_by_day['Competitor'] = np.maximum(grouped_by_day['Demand'] * 0.02 + 10, grouped_by_day['Cost'] + 1.5)

# Adjusting parameters for daily pricing
iterations = 5
alpha = 0.1
epsilon = 0.1
beta = 0.02

# Track results for each day and product
results_daily = []
ingresos_per_iteration_daily = []

for iteration in range(iterations):
    grouped_by_day['Competitor'] = np.maximum(grouped_by_day['Competitor'], grouped_by_day['Cost'] + 1.5)
    D = grouped_by_day['Demand'].values
    C = grouped_by_day['Competitor'].values
    X = grouped_by_day['Cost'].values
    c = -D  # Negative demand to maximize revenue

    # Constraints for linear programming
    A_ub = [[1 if j == i else 0 for j in range(len(D))] for i in range(len(D))]
    b_ub = list(C)
    A_ub += [[-1 if j == i else 0 for j in range(len(D))] for i in range(len(D))]
    b_ub += [-X[i] for i in range(len(X))]
    A_ub += [[1 + alpha if j == i else 0 for j in range(len(D))] for i in range(len(D))]
    b_ub += list(np.maximum(C + alpha * X, X + 0.1))

    # Bounds for prices
    bounds = [(X[i] + 0.1, C[i] + 10) for i in range(len(D))]

    # Linear programming optimization
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if result.success:
        grouped_by_day['Optimized_Price'] = result.x
        results_daily.append(grouped_by_day[['Product', 'DayOfWeek', 'Demand', 'Competitor', 'Optimized_Price']].copy())

        # Calculate revenue
        ingresos = (grouped_by_day['Demand'] * grouped_by_day['Optimized_Price']).sum()
        ingresos_per_iteration_daily.append(ingresos)

        # Update demand and competitor prices for the next iteration
        grouped_by_day['Demand'] = np.maximum(
            grouped_by_day['Demand'] * (
                        1 - epsilon * (grouped_by_day['Optimized_Price'] - base_prices_day) / base_prices_day),
            5
        )
        grouped_by_day['Competitor'] += beta * (grouped_by_day['Demand'] - base_demand_day)
    else:
        print(f"Optimization failed at iteration {iteration + 1}")
        break

# Plot ingresos per iteration for daily pricing
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(ingresos_per_iteration_daily) + 1), ingresos_per_iteration_daily, marker='o', color='blue')
plt.title("Daily Ingresos per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Ingresos")
plt.xticks(range(1, len(ingresos_per_iteration_daily) + 1))
plt.grid(True)
plt.tight_layout()
plt.show()

# Display final optimized results for daily pricing
print(grouped_by_day[['Product', 'DayOfWeek', 'Demand', 'Competitor', 'Optimized_Price']])
