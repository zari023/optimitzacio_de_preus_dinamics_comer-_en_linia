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
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# Aggregate data by StockCode and day of the week
grouped_by_day = df.groupby(['StockCode', 'DayOfWeek']).agg({
    'Quantity': 'sum',
    'UnitPrice': 'mean',
    'IsWeekend': 'mean'
}).reset_index()

grouped_by_day.columns = ['Product', 'DayOfWeek', 'Demand', 'Price', 'IsWeekend']

# Add cost and competitor price
base_prices_day = grouped_by_day['Price'].values
base_demand_day = grouped_by_day['Demand'].values
grouped_by_day['Cost'] = grouped_by_day['Price'] * 0.2
grouped_by_day['Competitor'] = np.maximum(grouped_by_day['Demand'] * 0.02 + 10, grouped_by_day['Cost'] + 1.5)

# Adjusting parameters for daily pricing
iterations = 50  # Allow more iterations for thorough adjustments
alpha = 0.1
epsilon = 0.02  # Further reduced for stability
beta = 0.005  # Further reduced for competitor adjustment damping

# Track results for each day and product
results_daily = []
ingresos_per_iteration_daily = []

# Convergence threshold
convergence_threshold = 1e-4  # Stricter threshold for convergence
previous_ingresos = 0

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
        grouped_by_day['Ingresos'] = grouped_by_day['Demand'] * grouped_by_day['Optimized_Price']
        ingresos = grouped_by_day['Ingresos'].sum()
        ingresos_per_iteration_daily.append(ingresos)

        # Check for convergence
        if abs(ingresos - previous_ingresos) < convergence_threshold:
            print(f"Converged at iteration {iteration + 1} with ingresos: {ingresos:.2f}")
            break
        previous_ingresos = ingresos

        print(f"Ingressos at iteration {iteration + 1}: {ingresos:.2f}")

        # Update demand and competitor prices with stronger damping
        grouped_by_day['Demand'] = np.maximum(
            grouped_by_day['Demand'] * (
                1 - epsilon * (grouped_by_day['Optimized_Price'] - base_prices_day) / base_prices_day
            ),
            5
        )
        grouped_by_day['Competitor'] += beta * (grouped_by_day['Demand'] - base_demand_day)
    else:
        print(f"Optimization failed at iteration {iteration + 1}")
        break

# Aggregate ingresos by day of the week
ingresos_by_day = grouped_by_day.groupby('DayOfWeek')['Ingresos'].sum().reset_index()

# Plot ingresos per day of the week
plt.figure(figsize=(10, 6))
plt.bar(ingresos_by_day['DayOfWeek'], ingresos_by_day['Ingresos'], color='blue')
plt.title("Ingresos por Día de la Semana")
plt.xlabel("Día de la Semana")
plt.ylabel("Ingresos")
plt.xticks(ticks=range(7), labels=['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'])
plt.grid(True)
plt.tight_layout()
plt.show()

# Add original price to the final DataFrame
grouped_by_day['Original_Price'] = base_prices_day

# Save the final DataFrame to a CSV file
grouped_by_day.to_csv("optimized_prices_daily.csv", index=False)

# Display final optimized results for daily pricing
print(grouped_by_day[['Product', 'DayOfWeek', 'Demand', 'Competitor', 'Original_Price', 'Optimized_Price', 'Ingresos']])
