Clustering con K-Means
Este proyecto implementa el algoritmo de clustering no supervisado K-Means utilizando Scikit-Learn. El objetivo es demostrar cómo se pueden identificar y visualizar patrones en un conjunto de datos sin etiquetas, agrupándolos en clusters para su análisis.

Objetivo del Proyecto
El propósito de este proyecto es aplicar el algoritmo K-Means para agrupar un conjunto de datos sintético generado con make_blobs. Usamos K-Means para descubrir clusters naturales dentro de los datos y analizar su comportamiento visualizando los resultados de cada grupo y su centroide. Este enfoque es útil en aplicaciones prácticas, como la segmentación de clientes, análisis de patrones y clasificación.

Algoritmo Seleccionado: K-Means Clustering
El algoritmo K-Means es una técnica popular de clustering no supervisado que divide los datos en k clusters en función de la proximidad a los centroides. Es especialmente adecuado para datos con estructuras de clusters compactas. En este proyecto, se seleccionaron 4 clusters para agrupar los datos generados, representando distintos patrones dentro del conjunto.

Justificación de K-Means
K-Means fue elegido por su:
Simplicidad y velocidad para agrupar puntos de datos.
Capacidad para identificar estructuras claras de agrupamiento en datos generados alrededor de puntos centrales.
Escalabilidad para manejar grandes volúmenes de datos.

Implementación en Google Colab:

Requisitos Previos
Python 3.x: Asegúrate de que Google Colab utiliza una versión actualizada de Python.
Scikit-Learn y Matplotlib: Las librerías Scikit-Learn y Matplotlib se instalan por defecto en Google Colab, por lo que no es necesario instalarlas adicionalmente.
Pasos para Ejecutar el Código en Google Colab
Abrir Google Colab: Ve a Google Colab y selecciona "Nuevo Notebook".

Importar Librerías: Asegúrate de que el código incluye las importaciones necesarias:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

Copia el Código en el Notebook: Utiliza el siguiente código para ejecutar el proyecto, puedes modificarlo a tu gusto si deseas:
# Generar conjunto de datos
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-Means
kmeans = KMeans(n_clusters=4)
kmeans.fit(X_scaled)
y_kmeans = kmeans.predict(X_scaled)
centers = kmeans.cluster_centers_

# Visualización de Resultados
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

Ejecutar las Celdas: Ejecuta cada celda de código en el notebook para generar el conjunto de datos, aplicar el clustering y visualizar los resultados.

Resultados
El algoritmo K-Means agrupa los puntos en función de su proximidad a cada centroide, representado por marcadores X rojos en el gráfico. Cada color representa un cluster, lo cual facilita la identificación visual de los grupos naturales en los datos.

Conclusión
K-Means es efectivo en conjuntos de datos donde se esperan agrupaciones claras y es útil para diversas aplicaciones en el análisis de datos. La visualización muestra cómo los puntos se agrupan alrededor de los centroides, lo que valida la elección de este algoritmo para el conjunto de datos.
