
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# --- Paramètres physiques et numériques ---
dx = 0.001                  # Pas d'espace
nx = int(2 / dx)            # Nombre de points (domaine de 0 à 2)
x = np.linspace(0, (nx - 1) * dx, nx)  # Axe spatial

v0 = -50
V = np.zeros(nx)
V[int(0.5 / dx):int(1.5 / dx)] = v0

# --- Hamiltonien discretisé ---
diag = 1.0 / dx**2 + V                    # Terme diagonal
off_diag = -0.5 / dx**2 * np.ones(nx - 1)  # Terme hors-diagonal

# --- Résolution de l'équation de Schrödinger ---
n_states = 20  # On en demande 20 pour être sûr d'inclure tous les états liés
energies, wavefuncs = eigh_tridiagonal(diag, off_diag, select='i', select_range=(0, n_states-1))

# --- Filtrage des états liés (E < 0) ---
mask_lies = energies < 0
energies_lies = energies[mask_lies]
wavefuncs_lies = wavefuncs[:, mask_lies]

# --- Normalisation des fonctions d’onde ---
norms = np.sqrt(np.sum(np.abs(wavefuncs_lies)**2, axis=0) * dx)
wavefuncs_lies /= norms

# --- Affichage ---
plt.figure(figsize=(10, 6))
for i in range(len(energies_lies)):
    plt.plot(x, wavefuncs_lies[:, i]**2 + energies_lies[i], label=f"E{i} = {energies_lies[i]:.2f} eV")

plt.plot(x, V, 'k', label="Potentiel V(x)")
plt.title("États stationnaires liés (E < 0)")
plt.xlabel("x")
plt.ylabel("Énergie / |ψ(x)|²")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Impression des énergies liées ---
print("Énergies des états liés (E < 0) :")
for i, E in enumerate(energies_lies):
    print(f"  État {i} : E = {E:.4f} eV")

print(f"\nNombre total d'états liés : {len(energies_lies)}")
