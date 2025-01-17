import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import lu

# ----- Definindo o problema -----
# Análise de um sistema de transporte urbano, otimizando o fluxo entre pontos de uma cidade.

# Dados de entrada: Matriz de adjacência representando a conectividade entre pontos
np.random.seed(42)
num_points = 5
adj_matrix = np.random.randint(1, 10, size=(num_points, num_points))
np.fill_diagonal(adj_matrix, 0)  # Sem laços

# Criando um dataframe para visualização
df_adj_matrix = pd.DataFrame(adj_matrix, columns=[f"P{i+1}" for i in range(num_points)],
                             index=[f"P{i+1}" for i in range(num_points)])
print("Matriz de adjacência:")
print(df_adj_matrix)

# ----- Decomposição LU -----
# Resolvendo um sistema linear associado ao fluxo de transporte
b = np.random.randint(1, 20, size=num_points)  # Vetor de demandas
P, L, U = lu(adj_matrix)

# Resolução do sistema LUx = b
y = np.linalg.solve(L, np.dot(P.T, b))
x = np.linalg.solve(U, y)

print("\nResultados do sistema linear:")
print(f"Demanda: {b}")
print(f"Fluxos otimizados: {x}")

# ----- Visualização dos resultados -----
# Matriz de adjacência como heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(adj_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=df_adj_matrix.columns,
            yticklabels=df_adj_matrix.index)
plt.title("Matriz de Adjacência - Conectividade entre Pontos")
plt.xlabel("Pontos de Destino")
plt.ylabel("Pontos de Origem")
plt.show()

# Fluxos otimizados
plt.figure(figsize=(8, 6))
plt.bar(range(1, num_points + 1), x, color="skyblue")
plt.title("Fluxos Otimizados")
plt.xlabel("Ponto")
plt.ylabel("Fluxo")
plt.xticks(range(1, num_points + 1), labels=[f"P{i}" for i in range(1, num_points + 1)])
plt.show()

# ----- Estrutura de Dados: Fila de Prioridade -----
# Simulação de uma fila de prioridade para transporte
class PriorityQueue:
    def __init__(self):
        self.queue = []

    def insert(self, item):
        self.queue.append(item)
        self.queue.sort(key=lambda x: x[1])  # Ordenar por prioridade

    def pop(self):
        if not self.is_empty():
            return self.queue.pop(0)
        else:
            return None

    def is_empty(self):
        return len(self.queue) == 0

# Fila de prioridade simulando transporte
pq = PriorityQueue()
for i in range(num_points):
    pq.insert((f"P{i+1}", x[i]))

print("\nFila de prioridade de transporte (ordem de atendimento):")
while not pq.is_empty():
    print(pq.pop())
