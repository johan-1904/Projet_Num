import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from matplotlib.animation import FuncAnimation

# Constantes physiques (unités atomiques)
ħ = 1.0
m = 1.0

# Paramètres spatiaux
L = 200.0
N = 2048
dx = L / N
x = np.linspace(-L/2, L/2, N)
dx2 = dx**2

# Potentiel gaussien (puits)
V0 = 1.0
a = 5.0
V = -V0 * np.exp(-(x / a)**2)

# Paquet d'ondes initial
x0 = -50.0
k0 = 1.5
sigma = 10.0
psi0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx)  # Normalisation

# Laplacien discret
laplacian = (-2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)) / dx2

# Hamiltonien
H = -ħ**2 / (2 * m) * laplacian + np.diag(V)

# Crank-Nicolson
dt = 0.05
A = np.eye(N) + 1j * dt / (2 * ħ) * H
B = np.eye(N) - 1j * dt / (2 * ħ) * H
lu, piv = lu_factor(A)

# Préparation de l'animation
psi = psi0.copy()
fig, ax = plt.subplots(figsize=(10, 5))
line_psi, = ax.plot(x, np.abs(psi)**2, label="|ψ(x,t)|²")
line_V, = ax.plot(x, V / np.abs(V).max() * 0.5, 'r--', label="V(x) (échelle réduite)")
ax.set_xlim(-L/2, L/2)
ax.set_ylim(0, 0.6)
ax.set_xlabel("x")
ax.set_ylabel("Amplitude")
ax.set_title("Propagation d’un paquet d’ondes dans un potentiel gaussien")
ax.legend()
ax.grid(True)

def update(frame):
    global psi
    for _ in range(5):  # Plus de stabilité : plusieurs pas internes par frame
        rhs = B @ psi
        psi = lu_solve((lu, piv), rhs)
    line_psi.set_ydata(np.abs(psi)**2)
    return line_psi, line_V

ani = FuncAnimation(fig, update, frames=300, blit=True, interval=30)
plt.show()
