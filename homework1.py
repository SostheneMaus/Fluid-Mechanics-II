import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def read_single_file(filepath):
    """
    Lit un fichier et retourne les composantes u, v, w de la vitesse
    
    Args:
        filepath: Chemin vers le fichier à lire
        
    Returns:
        u, v, w: Les trois composantes de la vitesse sous forme de tableaux numpy
    """
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1], data[:, 2]  # u, v, w

def get_file_path(dimension, i, j):
    """
    Construit le chemin vers un fichier de données
    
    Args:
        dimension: 'x', 'y' ou 'z'
        i, j: indices de position
    
    Returns:
        Chemin absolu vers le fichier
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, f'pencils_{dimension}', f'{dimension}_{i}_{j}.txt')



# Definitions of turbulence parameters
# k : kinetic energy
# e : dissipation rate with the relation (valid in HIT) e=15*nu*(du/dx^2)_ta
# L : integral length scale
# Re_L : Reynolds number based on the integral length scale L
# eta : Kolmogorov scale
# Lambda : Taylor microscale
# Re_lambda : Reynolds number based on the Taylor microscale Lambda



L = 2*np.pi  # domain size
nu = 1.10555e-5    # kinematic viscosity


# Plot must be in function of r/eta in loglog scale

def load_all_data():
    """
    Charge toutes les données de tous les fichiers et calcule les moyennes globales
    """
    # Listes pour stocker toutes les composantes
    all_u = []
    all_v = []
    all_w = []
    
    # Pour chaque dimension (x, y, z)
    for dim in ['x', 'y', 'z']:
        # Pour chaque position i, j
        for i in range(4):
            for j in range(4):
                filepath = get_file_path(dim, i, j)
                try:
                    u, v, w = read_single_file(filepath)
                    all_u.extend(u)  # Ajouter toutes les valeurs de u
                    all_v.extend(v)  # Ajouter toutes les valeurs de v
                    all_w.extend(w)  # Ajouter toutes les valeurs de w
                except Exception as e:
                    print(f"Erreur avec le fichier {filepath}: {e}")
    
    # Convertir en tableaux numpy pour les calculs
    all_u = np.array(all_u)
    all_v = np.array(all_v)
    all_w = np.array(all_w)
    
    return all_u, all_v, all_w


# 1 : Calcul de l'énergie cinétique moyenne

def compute_kinetic_energy(u, v, w):
    """Calcule l'énergie cinétique"""
    return 0.5 * (u**2 + v**2 + w**2)

# Charger toutes les données
print("Chargement de toutes les données...")
u, v, w = load_all_data()

# Calculer l'énergie cinétique moyenne
k = compute_kinetic_energy(u, v, w)
k_mean = np.mean(k)

print("\nStatistiques globales:")
print(f"Nombre total de points: {len(u)}")
print(f"Moyennes des composantes:")
print(f"u_mean = {np.mean(u):.6f}")
print(f"v_mean = {np.mean(v):.6f}")
print(f"w_mean = {np.mean(w):.6f}")
print(f"\nÉnergie cinétique moyenne: {k_mean:.6f}")


# 2 : Calcul du taux de dissipation moyen

def compute_dissipation_rate(u, nu):
    """Calcule le taux de dissipation moyen"""
    dx = L / len(u)  # Espacement entre les points
    du_dx = (-np.roll(u, -2) + 8*np.roll(u, -1) - 8*np.roll(u, 1) + np.roll(u, 2)) / (12 * dx)
    dissipation = 15 * nu * np.mean(du_dx**2)
    return np.mean(dissipation)

e_mean = compute_dissipation_rate(u, nu)
print(f"Taux de dissipation moyen: {e_mean:.6f}")

# 3 : Calcul de l'échelle de Kolmogorov

def compute_kolmogorov_scale(nu, e):
    """Calcule l'échelle de Kolmogorov"""
    return (nu**3 / e)**0.25

eta = compute_kolmogorov_scale(nu, e_mean)
print(f"Échelle de Kolmogorov : {eta:.6f}")

# 4 : Calcul de l'échelle intégrale
def compute_integral_scale(k, e):
    """Calcule l'échelle intégrale"""
    return (k**1.5) / e
L_integral = compute_integral_scale(k_mean, e_mean)
print(f"Échelle intégrale : {L_integral:.6f}")
# 5 : Calcul de l'échelle de Taylor

def compute_taylor_microscale(k, nu, e):
    return (10*nu*k / e)**0.5
Lambda = compute_taylor_microscale(k_mean, nu, e_mean)
print(f"Échelle de Taylor : {Lambda:.6f}")

# 6 : Calcul des nombres de Reynolds

def compute_reynolds_numbers(k, Lambda, nu,e):
    """Calcule les nombres de Reynolds basés sur L et Lambda"""
    
    Re_lambda = (k**0.5) * Lambda / nu
    return Re_lambda

Re_lambda = compute_reynolds_numbers(k_mean, Lambda, nu, e_mean)
print(f"Nombre de Reynolds basé sur Lambda : {Re_lambda:.6f}")


# STRUCTURE FUNCTIONS 

# Longitudinal structure functions D_11(r) = <(u(x+r)-u(x))^2>
# Obtain the longitudinal and transverse structure functions, D11(r eˆx) and D22(r eˆx),
# for r/η up to 5×10**3
# Since D33(r eˆx)) = D22(r eˆx) in HIT, you have twice as much
# data available per pencil to obtain the transverse function.


def compute_structure_functions(u, max_r):
    """Calcule les fonctions de structure D11 et D22 jusqu'à max_r"""
    D11 = np.zeros(max_r)
    D22 = np.zeros(max_r)
    N = len(u)
    
    for r in range(1, max_r + 1):
        diffs = u[r:] - u[:-r]
        D11[r-1] = np.mean(diffs**2)
        # Pour D22, on utilise la même approche en supposant isotropie
        D22[r-1] = np.mean(diffs**2)  # En HIT, D22 est similaire à D11 pour les statistiques globales
    
    return D11, D22


def compansated_compute_structure_functions(u, max_r):
    print("For eta >> r but not too large : D11(r) ~ C2 (e r)^{2/3} with C2 ~ 2.1 and D22(r) ~ 4/3 C2 (e r)^{2/3}")
    D11 = 2.1*(e_mean * np.arange(1, max_r+1))**(2/3)
    D22 = (4/3)*2.1*(e_mean * np.arange(1, max_r+1))**(2/3)
    return D11, D22

max_r = int(5e2)
D11, D22 = compute_structure_functions(u, max_r)
D11_comp, D22_comp = compansated_compute_structure_functions(u, max_r)

print(f"\nFonctions de structure calculées jusqu'à r = {max_r}")
# Tracer les fonctions de structure
r_values = np.arange(1, max_r+1)
plt.loglog(r_values, D11, label='D11(r)')
plt.loglog(r_values, D22, label='D22(r)')
plt.loglog(r_values, D11_comp, label='D11 compensée', linestyle='--')
plt.loglog(r_values, D22_comp, label='D22 compensée', linestyle='--')
plt.xlabel('r')
plt.ylabel('Structure Functions')
plt.title('Fonctions de structure D11 et D22')
plt.legend()
plt.show()

# compute slope in the inertial range

def compute_inertial_range_slope(D, r_values, r_min, r_max):
    """Calcule la pente dans la plage inertielle"""
    mask = (r_values >= r_min) & (r_values <= r_max)
    log_r = np.log(r_values[mask])
    log_D = np.log(D[mask])
    slope, intercept = np.polyfit(log_r, log_D, 1)
    return slope
slope_D11 = compute_inertial_range_slope(D11, r_values, 10, 1000)
slope_D22 = compute_inertial_range_slope(D22, r_values, 10, 1000)
print(f"Pente de D11 dans la plage inertielle : {slope_D11:.6f}")
print(f"Pente de D22 dans la plage inertielle : {slope_D22:.6f}")
