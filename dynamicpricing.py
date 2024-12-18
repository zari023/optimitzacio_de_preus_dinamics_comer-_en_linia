import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import linprog
import numpy as np

# Load and prepare market data
market_data = pd.read_csv("data-2 - copia.csv").dropna()

product_metrics = market_data.groupby('StockCode').agg({
    'Quantity': 'sum', 
    'UnitPrice': 'mean'
}).reset_index()
product_metrics.columns = ['product_id', 'initial_demand', 'initial_price']

# Market parameters
PRODUCTION_COST = 5.0
PRICE_SENSITIVITY = 0.1  # Former alpha
DEMAND_ELASTICITY = 0.3  # Former epsilon
COMPETITOR_RESPONSE_RATE = 0.05  # Former beta
SIMULATION_ITERATIONS = 5

product_metrics['production_cost'] = PRODUCTION_COST
initial_prices = product_metrics['initial_price'].values
initial_demand = product_metrics['initial_demand'].values

# Initialize competitor pricing strategy
product_metrics['competitor_price'] = np.maximum(
    product_metrics['initial_demand'] * 0.02 + 10, 
    product_metrics['production_cost'] + 1.5
)

simulation_results = []

for iteration in range(SIMULATION_ITERATIONS):
    # Ensure competitor prices stay above minimum viable price
    product_metrics['competitor_price'] = np.maximum(
        product_metrics['competitor_price'], 
        product_metrics['production_cost'] + 1.5
    )
    
    current_demand = product_metrics['initial_demand'].values
    competitor_prices = product_metrics['competitor_price'].values
    production_costs = product_metrics['production_cost'].values
    
    # Optimization objective: maximize revenue
    objective_coefficients = -current_demand

    # Define price constraints matrix
    price_constraints = []
    constraint_bounds = []
    
    # Price must be below competitor price
    price_constraints.extend([[1 if j == i else 0 for j in range(len(current_demand))] 
                            for i in range(len(current_demand))])
    constraint_bounds.extend(competitor_prices)
    
    # Price must be above production cost
    price_constraints.extend([[-1 if j == i else 0 for j in range(len(current_demand))] 
                            for i in range(len(current_demand))])
    constraint_bounds.extend([-cost for cost in production_costs])
    
    # Price markup constraints
    price_constraints.extend([[1 + PRICE_SENSITIVITY if j == i else 0 
                             for j in range(len(current_demand))] 
                            for i in range(len(current_demand))])
    constraint_bounds.extend(
        list(np.maximum(competitor_prices + PRICE_SENSITIVITY * production_costs, 
                       production_costs + 0.1))
    )

    # Define valid price ranges
    price_bounds = [(cost + 0.1, comp_price + 10) 
                   for cost, comp_price in zip(production_costs, competitor_prices)]

    result = linprog(objective_coefficients, A_ub=price_constraints, b_ub=constraint_bounds, bounds=price_bounds, method='highs')
    if result.success:
        product_metrics['optimized_price'] = result.x
        simulation_results.append(product_metrics[['product_id', 'initial_demand', 'competitor_price', 'optimized_price']].copy())
        product_metrics['initial_demand'] = np.maximum(product_metrics['initial_demand'] * (1 - DEMAND_ELASTICITY * (product_metrics['optimized_price'] - initial_prices) / initial_prices), 5)
        product_metrics['competitor_price'] += COMPETITOR_RESPONSE_RATE * (product_metrics['initial_demand'] - initial_demand)
    else:
        print(f"Optimization failed at iteration {iteration + 1}")
        break

# Plot Iteration-Level Graphs
for iteration, res in enumerate(simulation_results):
    top_products = res.head(5)  # Focus on top 5 products
    fig, ax = plt.subplots(figsize=(10, 6))

    # Width and position of bars
    width = 0.25
    indices = range(len(top_products))

    # Plot bars for Original, Competitor, and Optimized Prices
    bars1 = ax.bar(indices, initial_prices[:5], width, label='Original Price', color='blue')
    bars2 = ax.bar([i + width for i in indices], top_products['competitor_price'], width, label='Competitor Price', color='red')
    bars3 = ax.bar([i + 2 * width for i in indices], top_products['optimized_price'], width, label='Optimized Price', color='green')

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
    ax.set_xticklabels(top_products['product_id'], rotation=45, ha='right')
    ax.set_title(f"Iteration {iteration + 1}: Original, Competitor, and Optimized Prices")
    ax.set_xlabel('Product')
    ax.set_ylabel('Price')
    ax.legend()
    plt.tight_layout()
    plt.show()


# Plot Product-Level Graphs
unique_products = simulation_results[0]['product_id'].head(5)  # Focus on first 5 products
for product in unique_products:
    fig, ax = plt.subplots(figsize=(10, 6))
    demand = [res.loc[res['product_id'] == product, 'initial_demand'].values[0] for res in simulation_results]
    competitor = [res.loc[res['product_id'] == product, 'competitor_price'].values[0] for res in simulation_results]
    optimized = [res.loc[res['product_id'] == product, 'optimized_price'].values[0] for res in simulation_results]
    iterations_range = range(1, len(simulation_results) + 1)

    ax.plot(iterations_range, demand, label='Demand', marker='o', color='blue')
    ax.plot(iterations_range, competitor, label='Competitor Price', marker='o', color='red')
    ax.plot(iterations_range, optimized, label='Optimized Price', marker='o', color='green')

    ax.set_title(f"Product {product}: Demand, Competitor Price, and Optimized Price Over Iterations")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.legend()
    plt.tight_layout()
    plt.show()


