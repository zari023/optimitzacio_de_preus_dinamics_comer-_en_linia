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
grouped_by_day['Competitor'] = np.maximum(grouped_by_day['Demand'] * 0.02 + grouped_by_day['Price'] * 1.02, grouped_by_day['Cost'] + 1.5)

# Adjusting parameters for daily pricing
iterations = 50
alpha = 0.1
epsilon = 0.02
beta = 0.005

# Track results for each day and product
results_daily = []
ingresos_per_iteration_daily = []

# DataFrame to store optimized prices and demand per iteration
optimized_data = {product: {'Optimized_Price': [], 'Demand': [], 'Competitor_Price': []} for product in grouped_by_day['Product'].unique()}

# Convergence threshold
convergence_threshold = 1e-4
previous_ingresos = 0

for iteration in range(iterations):
    grouped_by_day['Competitor'] = np.maximum(grouped_by_day['Competitor'], grouped_by_day['Cost'] + 1.5)
    D = grouped_by_day['Demand'].values
    C = grouped_by_day['Competitor'].values
    X = grouped_by_day['Cost'].values
    c = -D

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

        # Store optimized prices, demand, and competitor prices per iteration
        for product in grouped_by_day['Product'].unique():
            product_data = grouped_by_day[grouped_by_day['Product'] == product]
            optimized_data[product]['Optimized_Price'].append(product_data['Optimized_Price'].values[0])
            optimized_data[product]['Demand'].append(product_data['Demand'].values[0])
            optimized_data[product]['Competitor_Price'].append(product_data['Competitor'].values[0])

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

# Add original price to the final DataFrame
grouped_by_day['Original_Price'] = base_prices_day

# Save the final DataFrame to a CSV file
grouped_by_day.to_csv("optimized_prices_daily.csv", index=False)

# Display final optimized results for daily pricing
print(grouped_by_day[['Product', 'DayOfWeek', 'Demand', 'Competitor', 'Original_Price', 'Optimized_Price', 'Ingresos']])

# Calculate real and optimized revenues
grouped_by_day['Ingresos_Reales'] = grouped_by_day['Demand'] * grouped_by_day['Original_Price']
grouped_by_day['Ingresos_Optimizados'] = grouped_by_day['Demand'] * grouped_by_day['Optimized_Price']

# Calculate real revenues per iteration
ingresos_reales_per_iteration = [grouped_by_day['Ingresos_Reales'].sum()] * len(results_daily)

# Plot real and optimized revenues per iteration
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(ingresos_per_iteration_daily) + 1), ingresos_per_iteration_daily, label='Ingresos Optimizados', marker='o')
plt.plot(range(1, len(ingresos_reales_per_iteration) + 1), ingresos_reales_per_iteration, label='Ingresos Reales', marker='x')
plt.title("Comparación de Ingresos Reales y Optimizados por Iteración")
plt.xlabel("Iteración")
plt.ylabel("Ingresos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Convert optimized_data to DataFrame for visualization
optimized_df = pd.DataFrame({
    'Product': [],
    'Iteration': [],
    'Optimized_Price': [],
    'Demand': [],
    'Competitor_Price': []
})

for product, data in optimized_data.items():
    for i in range(len(data['Optimized_Price'])):
        new_row = pd.DataFrame({
            'Product': [product],
            'Iteration': [i + 1],
            'Optimized_Price': [data['Optimized_Price'][i]],
            'Demand': [data['Demand'][i]],
            'Competitor_Price': [data['Competitor_Price'][i]]
        })
        optimized_df = pd.concat([optimized_df, new_row], ignore_index=True)

# Filter the DataFrame for the selected products
Productlist = ["15056BL", "15056N", "2068", "20725", "10002"]
filtered_df = optimized_df[optimized_df['Product'].isin(Productlist)]

# Visualize the data for the selected products
for product in Productlist:
    product_data = filtered_df[filtered_df['Product'] == product]
    if not product_data.empty:
        plt.figure(figsize=(12, 6))
        iterations_to_plot = np.linspace(1, product_data['Iteration'].max(), 5, dtype=int)
        bar_width = 0.25
        index = np.arange(len(iterations_to_plot))

        plt.bar(index, product_data[product_data['Iteration'].isin(iterations_to_plot)]['Optimized_Price'], bar_width, label='Precio Optimizado')
        plt.bar(index + bar_width, product_data[product_data['Iteration'].isin(iterations_to_plot)]['Competitor_Price'], bar_width, label='Precio Competidor')
        real_price = grouped[grouped['Product'] == product]['Price'].values
        if real_price.size > 0:
            plt.bar(index + 2 * bar_width, [real_price[0]] * len(iterations_to_plot), bar_width, label='Precio Real')

        plt.xlabel('Iteración')
        plt.ylabel('Precio')
        plt.title(f'Precio Optimizado, Competidor y Real para {product}')
        plt.xticks(index + bar_width, iterations_to_plot)
        plt.legend()
        plt.tight_layout()
        plt.show()
#valores de comprobacion
#ingresos reales
ingresos_reales = grouped_by_day['Ingresos_Reales'].sum()
print(f"Ingresos Reales: {ingresos_reales:.2f}")
#ingresos optimizados
ingresos_optimizados = grouped_by_day['Ingresos_Optimizados'].sum()
print(f"Ingresos Optimizados: {ingresos_optimizados:.2f}")
#ingresos reales menos coste
ingresos_reales_menos_coste = ingresos_reales - grouped_by_day['Cost'].sum()
#ingresos optimizados menos coste
ingresos_optimizados_menos_coste = ingresos_optimizados - grouped_by_day['Cost'].sum()
