# Projet Numérique en Physique Moderne
# Effet Ramsauer-Townsend
# CY Tech 2024-2025

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import identity, csc_matrix
from scipy.sparse.linalg import splu
from matplotlib.animation import FuncAnimation

###########################################
# 1. PARAMETRES PHYSIQUES ET NUMERIQUES  #
###########################################

# Constantes physiques (unités naturelles)
hbar = 1.0  # constante de Planck réduite
m = 1.0     # masse de la particule

# Paramètres du puits de potentiel
V0 = 50     # profondeur du puits
a = 1.0     # demi-largeur du puits

# Domaine spatial
x_min, x_max = -5.0, 5.0
N = 1000
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]

# Potentiel
V = np.zeros_like(x)
V[np.abs(x) < a] = -V0

######################################
# 2. RESOLUTION DE SCHRODINGER STATIQUE
######################################

def solve_stationary_states(V, dx, Nstates=5):
    diag = 1.0 / dx**2 + V
    off_diag = -0.5 / dx**2 * np.ones(N - 1)
    energies, wavefuncs = eigh_tridiagonal(diag, off_diag)
    return energies[:Nstates], wavefuncs[:, :Nstates]

energies, wavefuncs = solve_stationary_states(V, dx)

# Normalisation
wavefuncs /= np.sqrt(dx) * np.linalg.norm(wavefuncs, axis=0)

# Affichage
plt.figure()
for i in range(len(energies)):
    plt.plot(x, wavefuncs[:, i]**2 + energies[i], label=f"E={energies[i]:.2f}")
plt.plot(x, V, color='black', label="Potentiel")
plt.title("Fonctions d'onde stationnaires")
plt.legend()
plt.grid()
plt.xlabel("x")
plt.ylabel("Energie")
plt.show()

######################################
# 3. PAQUET D'ONDES - DYNAMIQUE TEMPORELLE
######################################

# Paquet initial : gaussien centré en x0 avec impulsion p0
x0 = -2.0
sigma = 0.3
p0 = 10.0

def psi_0(x):
    return (1/(sigma * np.sqrt(np.pi)))**0.5 * np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * p0 * x)

psi = psi_0(x)

# Discrétisation de la dérivée seconde (Hamiltonien)
Lap = (-2*np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)) / dx**2
H = -0.5 * Lap + np.diag(V)

# Paramètres temporels
dt = 0.005
T = 1.0
Nt = int(T/dt)

# Résolution temporelle par Crank-Nicolson
H_sparse = csc_matrix(H)
I_sparse = identity(N, format="csc", dtype=complex)
A = (I_sparse + 1j * dt / 2 * H_sparse)
B = (I_sparse - 1j * dt / 2 * H_sparse)

lu = splu(A)

psis = [psi.copy()]
for _ in range(Nt):
    rhs = B @ psi
    psi = lu.solve(rhs)
    psis.append(psi.copy())

psis = np.array(psis)

# Animation simple (à convertir si besoin en vidéo)
fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(psis[0])**2)
ax.set_ylim(0, 2)
ax.set_title("Propagation du paquet d'ondes")

def update(frame):
    line.set_ydata(np.abs(psis[frame])**2)
    return line,

ani = FuncAnimation(fig, update, frames=range(0, Nt, 10), interval=50)
plt.show()

######################################
# 4. COMPARAISON AVEC L'EFFET RAMSAUER
######################################

# On observera numériquement que pour certaines énergies (ou p0), le paquet traverse le puits sans forte déviation,
# simulant l'effet Ramsauer (faible section efficace). À approfondir avec des profils de transmission/réflexion.

# Pour une vraie comparaison avec l'expérience, il faudrait calculer la section efficace σ(E)
# à partir des coefficients de transmission/reflexion du paquet incident en fonction de l'énergie.

# Ceci peut être fait en analysant les flux incidents/réfléchis à partir de |psi(x,t)|^2

######################################
# 5. ETAPES SUIVANTES (OPTIONNEL)
######################################

# Implémenter un potentiel plus réaliste (ex. potentiel de Lennard-Jones)
# Comparer les résultats avec l'article [G76] (He + He)
# Étudier l'effet du choix de p0 et de la largeur du paquet sur la diffusion
