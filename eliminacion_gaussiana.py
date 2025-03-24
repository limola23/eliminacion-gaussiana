import numpy as np

def gaussian_elimination_scaled_partial_pivoting(A, b):
    """
    Implementa la eliminación gaussiana con pivotaje parcial escalado
    para resolver el sistema de ecuaciones Ax = b.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)
    
    # Vector de escala para cada fila
    scale = np.max(np.abs(A), axis=1)
    
    for k in range(n - 1):
        # Determinar la fila con el mayor valor relativo
        ratios = np.abs(A[k:n, k]) / scale[k:n]
        max_index = k + np.argmax(ratios)
        
        # Intercambiar filas si es necesario
        if max_index != k:
            A[[k, max_index]] = A[[max_index, k]]
            b[[k, max_index]] = b[[max_index, k]]
            scale[[k, max_index]] = scale[[max_index, k]]
        
        # Eliminación gaussiana
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    
    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:n], x[i + 1:n])) / A[i, i]
    
    return x

# Prueba con una matriz de tamaño 4x4
A = np.array([
    [2, 3, -1, 2],
    [4, 4, -3, 3],
    [-2, 3, 2, -1],
    [3, -1, 2, 5]
])

b = np.array([5, 3, 2, -1])
solucion = gaussian_elimination_scaled_partial_pivoting(A, b)
print("Solución del sistema:", solucion)