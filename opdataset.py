import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import linprog

# Cargar el DataFrame
df = pd.read_csv("data-2 - copia.csv")  # Asegúrate de que el archivo exista

# Preprocesar DataFrame
df = df.dropna()  # Eliminar valores faltantes
grouped = df.groupby('StockCode').agg({'Quantity': 'sum', 'UnitPrice': 'mean'}).reset_index()
grouped.columns = ['Product', 'Demand', 'Price']

# Añadir costos de producción y precios del competidor
grouped['Cost'] = 5.0  # Coste de producción fijo
grouped['Competitor'] = grouped['Demand'] * 0.02 + 10  # Precio del competidor

# Filtrar productos donde el coste no supera al precio del competidor
grouped = grouped[grouped['Cost'] <= grouped['Competitor']]

# Variables de optimización
D = grouped['Demand'].values  # Demanda
C = grouped['Competitor'].values  # Precios del competidor
X = grouped['Cost'].values  # Costes de producción

# Margen competitivo (5% por debajo del precio del competidor)
margin = 0.05
C_adjusted = C * (1 - margin)  # Ajustar el precio del competidor

# Función objetivo: Maximizar ingresos --> Minimize -sum(D_i * P_i)
c = -D  # Negativo para minimizar

# Restricciones
A_ub = []
b_ub = []

# Restricción: P_i <= C_adjusted (competitor price con margen)
A_ub.extend([[1 if j == i else 0 for j in range(len(D))] for i in range(len(D))])
b_ub.extend(C_adjusted)

# Restricción: P_i >= X_i --> -P_i <= -X_i
A_ub.extend([[-1 if j == i else 0 for j in range(len(D))] for i in range(len(D))])
b_ub.extend(-X)

# Limitar P_i a valores no negativos
bounds = [(0, None) for _ in range(len(D))]

# Resolver la optimización
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# Resultados y verificación
if result.success:
    grouped['Optimized_Price'] = result.x

    # Comprobación de restricciones
    grouped['Constraint_Cost'] = grouped['Optimized_Price'] >= grouped['Cost']
    grouped['Constraint_Competitor'] = grouped['Optimized_Price'] <= C_adjusted
    
    print("Resultados de optimización con comprobación de restricciones:")
    print(grouped[['Product', 'Demand', 'Cost', 'Competitor', 'Optimized_Price', 
                  'Constraint_Cost', 'Constraint_Competitor']])

    # Mostrar los primeros 5 productos
    grouped = grouped.head(5)

    # Gráfico
    fig, ax = plt.subplots(figsize=(12, 7))
    width = 0.25  # Ancho de las barras
    index = range(len(grouped))  # Posiciones de los productos

    # Barras de precios
    ax.bar(index, grouped['Price'], width, label='Original Price', color='blue')
    ax.bar([i + width for i in index], grouped['Competitor'], width, label='Competitor Price', color='red')
    ax.bar([i + 2 * width for i in index], grouped['Optimized_Price'], width, label='Optimized Price', color='green')

    # Añadir valores exactos en las barras
    for i, row in grouped.iterrows():
        ax.text(i, row['Price'] + 0.1, f"{row['Price']:.2f}", color='blue', ha='center')
        ax.text(i + width, row['Competitor'] + 0.1, f"{row['Competitor']:.2f}", color='red', ha='center')
        ax.text(i + 2 * width, row['Optimized_Price'] + 0.1, f"{row['Optimized_Price']:.2f}", color='green', ha='center')

    # Etiquetas y título
    ax.set_xlabel('Product')
    ax.set_ylabel('Price')
    ax.set_title('Comparison of Prices (First 5 Products): Original, Competitor, and Optimized')
    ax.set_xticks([i + width for i in index])  # Centrar las etiquetas
    ax.set_xticklabels(grouped['Product'], rotation=45, ha='right')

    # Leyenda
    ax.legend()

    # Mostrar la gráfica
    plt.tight_layout()
    plt.show()

else:
    print("Optimization failed:", result.message)
