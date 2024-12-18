import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.optimize import minimize

# Simulate a DataFrame for the example
data = {
    'StockCode': [f'P{i}' for i in range(1, 21)],
    'Quantity': np.random.randint(10, 100, 20),
    'UnitPrice': np.random.uniform(5, 20, 20)
}
df = pd.DataFrame(data)

# Preprocess DataFrame
df = df.dropna()  # Remove missing values
grouped = df.groupby('StockCode').agg({'Quantity': 'sum', 'UnitPrice': 'mean'}).reset_index()
grouped.columns = ['Product', 'Demand', 'Price']

# Function definitions used in the code
def modelo_demanda(precio, gradient=1.5, intercept=100):
    return max(intercept * (1 / (1 + gradient * precio)), 0)

def ingresos_totales(precio, modelo_func, costo_min, precio_comp):
    cantidad = modelo_func(precio)
    retorn = (cantidad * precio) - (costo_min * cantidad)  # Profits
    return -1 * retorn

def ingressos_optims(preu_optim, demanda, cost_min):
    # Calcular la quantitat a partir del model de demanda
    quantitat = modelo_demanda(preu_optim)
    # Calcular els ingressos com la quantitat per preu, menys el cost de producció
    return quantitat * preu_optim - quantitat * cost_min

# Iterate over different costs to simulate new scenarios
costs = np.linspace(3, 7, 5)  # Example: Testing for different costs from $3 to $7
for cost in costs:
    grouped['Cost'] = cost  # Assign new cost
    grouped['Competitor'] = grouped['Demand'] * 0.02 + 10  # Competitor price, e.g., based on demand and a base price

    # Ensure feasibility by filtering rows where Cost > Competitor
    feasible_group = grouped[grouped['Cost'] <= grouped['Competitor']]

    # Optimization inputs
    D = feasible_group['Demand'].values  # Demand
    C = feasible_group['Competitor'].values  # Competitor prices
    X = feasible_group['Cost'].values  # Production costs

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
        feasible_group['Optimized_Price'] = result.x
        feasible_group = feasible_group.head(10)  # Use only the first 10 products
        print(feasible_group[['Product', 'Demand', 'Price', 'Cost', 'Competitor', 'Optimized_Price']])

        # Non-linear optimization for total optimal price
        costo_min = feasible_group['Cost'].min()
        precio_comp = feasible_group['Competitor'].max()
        precio_inicial = (costo_min + precio_comp) / 2

        resultado = minimize(
            ingresos_totales,
            [precio_inicial], 
            args=(modelo_demanda, costo_min, precio_comp),
            bounds=[(costo_min, precio_comp)]
        )

        precio_optimo = resultado.x[0]
        print(f"Costo actual: {cost:.2f}, Precio óptimo (no lineal): {precio_optimo:.2f}")

        # Calculate total revenue as the sum of individual optimal revenues
        total_ingresos_optims = sum(
            ingressos_optims(preu, demanda, cost_min)
            for preu, demanda, cost_min in zip(feasible_group['Optimized_Price'], feasible_group['Demand'], feasible_group['Cost'])
        )

        print(f"Costo actual: {cost:.2f}, Suma de tots els ingressos òptims dels productes: {total_ingresos_optims:.2f}")

        # Plotting a bar chart
        fig, ax = plt.subplots(figsize=(12, 7))

        # Bar chart for Original Price, Competitor Price, and Optimized Price
        width = 0.25  # Width of each bar
        index = range(len(feasible_group))  # Position of each group of bars

        # Plot bars for each type of price
        ax.bar(index, feasible_group['Price'], width, label='Original Price', color='blue')
        ax.bar([i + width for i in index], feasible_group['Competitor'], width, label='Competitor Price', color='red')
        ax.bar([i + 2 * width for i in index], feasible_group['Optimized_Price'], width, label='Optimized Price', color='green')
        ax.axhline(precio_optimo, color='purple', linestyle='--', label=f'Non-linear Optimal Price: {precio_optimo:.2f}')

        # Add labels and title
        ax.set_xlabel('Product')
        ax.set_ylabel('Price')
        ax.set_title(f'Comparison of Prices (Cost: {cost:.2f})')
        ax.set_xticks([i + width for i in index])  # Center the x-tick labels
        ax.set_xticklabels(feasible_group['Product'], rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()
        plt.show()
    else:
        print(f"Optimization failed for cost: {cost:.2f}", result.message)
