import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fft import fft
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter

###################################################################################################
# PART 1 : Open and read the files 

def read_single_file(filepath):
    """
    Reads a single pencil file and returns the velocity components.

    Args:
        filepath: Path to the pencil file (.txt)

    Returns:
        u, v, w: NumPy arrays containing the velocity components in x, y, and z directions
    """
    data = np.loadtxt(filepath)
    u, v, w = data[:, 0], data[:, 1], data[:, 2]
    return u, v, w


def get_file_path(dimension, i, j):
    """
    Builds the path to a data file

    Args:
        dimension: 'x', 'y', or 'z' — the direction of the pencil
        i, j: position indices

    Returns:
        Path to the file
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, f'pencils_{dimension}', f'{dimension}_{i}_{j}.txt')


###################################################################################################
# PART 2 : Global quantities

"""
Definitions of turbulence parameters
- k : kinetic energy
- e : dissipation rate with the relation (valid in HIT) e=15*nu*(du/dx^2)_ta
- L : integral length scale
- Re_L : Reynolds number based on the integral length scale L
- eta : Kolmogorov scale
- Lambda : Taylor microscale
- Re_lambda : Reynolds number based on the Taylor microscale Lambda
"""


L = 2*np.pi         # domain size
nu = 1.10555e-5     # kinematic viscosity


# 1 : Turbulent kinetic energy k_bar
def compute_k_bar():
    """
    Computes the average turbulent kinetic energy over all 48 pencils.

    Returns:
        k_bar: Scalar value of the turbulent kinetic energy
    """
    total_k = 0
    count = 0
    for dimension in ['x', 'y', 'z']:
        for i in range(4):
            for j in range(4):
                filepath = get_file_path(dimension, i, j)
                u, v, w = read_single_file(filepath)
                k_local = 0.5 * (np.mean(u**2) + np.mean(v**2) + np.mean(w**2))
                total_k += k_local
                count += 1
    return total_k / count

k_bar = compute_k_bar()
print(f"Turbulent kinetic energy k_bar = {k_bar:.6e} m^2/s^2")


# 2 : Dissipation rate epsilon_bar
def deriv_4th_order(f, dx):
    """
    Computes the spatial derivative df/dx using fourth-order finite differences.

    Args:
        f: 1D NumPy array of function values
        dx: Grid spacing

    Returns:
        df: 1D NumPy array of derivative values
    """
    df = np.zeros_like(f)
    df[2:-2] = (-f[4:] + 8*f[3:-1] - 8*f[1:-3] + f[0:-4]) / (12 * dx)
    df[0]  = (-25*f[0] + 48*f[1] - 36*f[2] + 16*f[3] - 3*f[4]) / (12*dx)
    df[1]  = (-3*f[0] - 10*f[1] + 18*f[2] - 6*f[3] + f[4]) / (12*dx)
    df[-2] = (3*f[-5] - 16*f[-4] + 36*f[-3] - 48*f[-2] + 25*f[-1]) / (12*dx)
    df[-1] = (-f[-5] + 6*f[-4] - 18*f[-3] + 10*f[-2] + 3*f[-1]) / (12*dx)
    return df

def compute_epsilon_bar():
    """
    Computes the average dissipation rate epsilon over all 48 pencils.

    Returns:
        epsilon_bar: Scalar value of the dissipation rate
    """
    total_eps = 0
    count = 0
    dx = L / 32768  # grid spacing

    for dimension in ['x', 'y', 'z']:
        for i in range(4):
            for j in range(4):
                filepath = get_file_path(dimension, i, j)
                u, v, w = read_single_file(filepath)

                if dimension == 'x':
                    du_dx = deriv_4th_order(u, dx)
                    eps_local = 15 * nu * np.mean(du_dx**2)
                elif dimension == 'y':
                    dv_dy = deriv_4th_order(v, dx)
                    eps_local = 15 * nu * np.mean(dv_dy**2)
                elif dimension == 'z':
                    dw_dz = deriv_4th_order(w, dx)
                    eps_local = 15 * nu * np.mean(dw_dz**2)

                total_eps += eps_local
                count += 1

    return total_eps / count

epsilon_bar = compute_epsilon_bar()
print(f"Dissipation rate epsilon = {epsilon_bar:.6e} m^2/s^3")


# 3 : Integral scale L_bar
def compute_integral_scale(k_bar, epsilon_bar):
    """
    Computes the integral length scale L_bar using the HIT relation.

    Args:
        k_bar: Average turbulent kinetic energy
        epsilon_bar: Average dissipation rate

    Returns:
        L_bar: Integral length scale
    """
    return k_bar**1.5 / epsilon_bar

L_bar = compute_integral_scale(k_bar, epsilon_bar)
print(f"Integral length scale L = {L_bar:.6e} m")


# 4 : Reynolds number
def compute_rms_velocities():
    """
    Computes RMS velocities using directional pencils:
    - u from x-pencils
    - v from y-pencils
    - w from z-pencils

    Returns:
        u_rms, v_rms, w_rms: RMS velocities in x, y, and z directions
    """
    # u from x-pencils
    u_sq_sum = 0
    count = 0
    for i in range(4):
        for j in range(4):
            filepath = get_file_path('x', i, j)
            u, _, _ = read_single_file(filepath)
            u_sq_sum += np.mean(u**2)
            count += 1
    u_rms = np.sqrt(u_sq_sum / count)

    # v from y-pencils
    v_sq_sum = 0
    count = 0
    for i in range(4):
        for j in range(4):
            filepath = get_file_path('y', i, j)
            _, v, _ = read_single_file(filepath)
            v_sq_sum += np.mean(v**2)
            count += 1
    v_rms = np.sqrt(v_sq_sum / count)

    # w from z-pencils
    w_sq_sum = 0
    count = 0
    for i in range(4):
        for j in range(4):
            filepath = get_file_path('z', i, j)
            _, _, w = read_single_file(filepath)
            w_sq_sum += np.mean(w**2)
            count += 1
    w_rms = np.sqrt(w_sq_sum / count)

    return u_rms, v_rms, w_rms

u_rms, v_rms, w_rms = compute_rms_velocities()

def compute_reynolds_number(u_rms, L_bar, nu):
    """
    Computes the Reynolds number based on the integral scale.

    Args:
        u_rms: RMS velocity (can be averaged over u', v', w')
        L_bar: Integral length scale
        nu: Kinematic viscosity

    Returns:
        Re_L: Reynolds number
    """
    return k_bar**2 / (nu * epsilon_bar)
    # return u_rms * L_bar / nu

u_rms, v_rms, w_rms = compute_rms_velocities()
rms_velocities_avg = (u_rms + v_rms + w_rms) / 3
Re = compute_reynolds_number(rms_velocities_avg, L_bar, nu)
print(f"Reynolds number Re = {Re:.2f}")


# 5 : Kolmogorov scale eta 
def compute_kolmogorov_scale(nu, epsilon_bar):
    """
    Computes the Kolmogorov scale eta.

    Args:
        nu: Kinematic viscosity
        epsilon_bar: Average dissipation rate

    Returns:
        eta: Kolmogorov length scale
    """
    return (nu**3 / epsilon_bar)**0.25

eta = compute_kolmogorov_scale(nu, epsilon_bar)
print(f"Kolmogorov scale eta = {eta:.6e} m")


# 6 : Taylor micro-scale lambda
def compute_taylor_microscale_directional(k_bar, epsilon_bar, nu):
    """
    Computes Taylor microscale using the formula:
    λ = sqrt(10 * ν * k_bar / epsilon_bar)

    Args:
        k_bar: Turbulent kinetic energy
        epsilon_bar: Dissipation rate
        nu: Kinematic viscosity

    Returns:
        lambda_ke: Scalar Taylor microscale
    """
    return np.sqrt(10 * nu * k_bar / epsilon_bar)

lambda_taylor = compute_taylor_microscale_directional(k_bar, epsilon_bar, nu)
print(f"Taylor microscale lambda = {lambda_taylor:.6e} m")


# 7 : Reynolds number Re_lambda
def compute_re_lambda(Re):
    """
    Computes the Taylor-scale Reynolds number using:
    Re_lambda = sqrt((20/3) * Re)

    Args:
        Re: Integral-scale Reynolds number

    Returns:
        Re_lambda: Taylor-scale Reynolds number
    """
    return np.sqrt((20 / 3) * Re)

Re_lambda = compute_re_lambda(Re)
print(f"Taylor-scale Reynolds number Re_lambda = {Re_lambda:.2f}")


###################################################################################################
# PART 3 : Structure functions 

def compute_structure_functions(f, max_r):
    """
    Computes the second-order structure function D(r) for a 1D velocity signal f.

    Args:
        f: 1D NumPy array of velocity values
        max_r: maximum separation in grid points

    Returns:
        r_vals: array of separation distances
        D_vals: array of structure function values
    """
    r_vals = np.arange(1, max_r + 1)
    D_vals = np.zeros_like(r_vals, dtype=np.float64)

    for idx, r in enumerate(r_vals):
        diffs = f[r:] - f[:-r]
        D_vals[idx] = np.mean(diffs**2)

    return r_vals, D_vals


def average_structure_functions(dimension, component, max_r):
    """
    Averages structure functions over all 48 pencils (x, y, z directions × 16 pencils each).

    Args:
        dimension: 'all' to average over x, y, z; or 'x', 'y', 'z' for single direction
        component: 0 for u, 1 for v, 2 for w
        max_r: maximum separation in grid points

    Returns:
        r_vals, D_avg: averaged structure function
    """
    D_total = np.zeros(max_r)
    count = 0


    for i in range(4):
        for j in range(4):
            filepath = get_file_path(dimension, i, j)
            u, v, w = read_single_file(filepath)
            f = [u, v, w][component]
            r_vals, D_vals = compute_structure_functions(f, max_r)
            D_total += D_vals
            count += 1

    return r_vals, D_total / count

def average_transverse_structure_function(dimension, max_r):
    """
    Averages transverse structure functions D22 and D33 over all 16 pencils in the given direction.
    Returns the average of both components together.
    """
    D_total = np.zeros(max_r)
    count = 0

    for i in range(4):
        for j in range(4):
            filepath = get_file_path(dimension, i, j)
            u, v, w = read_single_file(filepath)

            _, D_v = compute_structure_functions(v, max_r)
            _, D_w = compute_structure_functions(w, max_r)

            D_total += D_v + D_w  # accumulate both transverse components
            count += 2            # count both v and w

    r_vals = np.arange(1, max_r + 1)
    return r_vals, D_total / count


dx = L / 32768
max_r = int(5000 * eta / dx) # convert r/η to grid points
  
# Longitudinal: u from x-pencils
r_vals, D11 = average_structure_functions('x', 0, max_r)

# Transverse: v and w from x-pencils (twice the data)
r_vals, D22 = average_transverse_structure_function('x', max_r)

# Convert r to r/η
r_eta = r_vals * (L / 32768) / eta

r_phys = r_vals * (L / 32768)
D11_comp = D11 / ((epsilon_bar * r_phys)**(2/3))
D22_comp = D22 / ((epsilon_bar * r_phys)**(2/3))


plt.figure(figsize=(10, 5))
plt.loglog(r_eta, D11, label=r'$D_{11}$ (Longitudinal)')
plt.loglog(r_eta, D22, label=r'$D_{22}$ (Transverse)')
plt.xlabel(r'$r / \eta$')
plt.ylabel(r'$D_{jj}(r\, \hat{e}_x)$')
plt.legend()
plt.grid(True, which='major', linestyle='-', linewidth=0.7)
plt.grid(True, which='minor', linestyle=':', linewidth=0.7, alpha=0.6)
plt.title(r'Structure functions')
plt.gca().xaxis.set_minor_locator(ticker.LogLocator(subs='all'))
plt.gca().yaxis.set_minor_locator(ticker.LogLocator(subs='all'))
plt.show()

mask = (r_eta > 100) & (r_eta < 1000)
C2_meas = np.mean(D11_comp[mask])
C2p_meas = np.mean(D22_comp[mask])

plt.figure(figsize=(10, 5))
line_D11, = plt.semilogx(r_eta, D11_comp, label=r'Compensated $D_{11}$')
line_D22, = plt.semilogx(r_eta, D22_comp, label=r'Compensated $D_{22}$')
plt.axhline(2.10, color=line_D11.get_color(), linestyle='--', label=r'$C_2 \approx 2.10$')
plt.axhline(2.793, color=line_D22.get_color(), linestyle='--', label=r"$C_2' \approx 1.33C_2$")
plt.axhline(C2_meas, color=line_D11.get_color(), linestyle=':', label=fr'$C_2^{{\text{{measured}}}} \approx {C2_meas:.2f}$')
plt.axhline(C2p_meas, color=line_D22.get_color(), linestyle=':', label=fr"$C_2'^{{\text{{measured}}}} \approx {C2p_meas:.2f}$")
plt.axvspan(100, 1000, color='gray', alpha=0.15, label='"Pseudo-plateau" range')
plt.xlabel(r'$r / \eta$')
plt.ylabel(r'$(\bar{\epsilon} r)^{-2/3} D_{11,22}(r\, \hat{e}_x)$')
plt.legend()
plt.grid(True, which='major', linestyle='-', linewidth=0.7)
plt.grid(True, which='minor', linestyle=':', linewidth=0.7, alpha=0.6)
plt.title(r'Compensated structure functions')
plt.show()


###################################################################################################
# PART 4 : One-dimensional energy spectra


def compute_spectrum(f, dx):
    f = f - np.mean(f)
    N = f.size
    f_hat = np.fft.rfft(f)
    k = 2 * np.pi * np.fft.rfftfreq(N, d=dx)
    E_k = (dx / (2*np.pi*N)) * np.abs(f_hat)**2
    if N % 2 == 0:
        E_k[1:-1] *= 2
    else:
        E_k[1:] *= 2
    return k, E_k

def average_energy_spectra():
    """
    Calcule les spectres 1D moyens E11 (longitudinal) et E22 (transverse)
    à partir des 16 x-pencils (comme dans tes structure functions).
    """
    # Initialisation
    E11_acc = None
    E22_acc = None
    count = 0

    for i in range(4):
        for j in range(4):
            filepath = get_file_path('x', i, j)
            u, v, w = read_single_file(filepath)

            k, E_u = compute_spectrum(u, dx)
            _, E_v = compute_spectrum(v, dx)
            _, E_w = compute_spectrum(w, dx)

            E22_local = 0.5 * (E_v + E_w)  # moyenne transverse

            if E11_acc is None:
                E11_acc = np.zeros_like(E_u)
                E22_acc = np.zeros_like(E_u)

            E11_acc += E_u
            E22_acc += E22_local
            count += 1

    # Moyenne sur tous les pencils
    E11_avg = E11_acc / count
    E22_avg = E22_acc / count

    return k, E11_avg, E22_avg
# Compute spectra
N  = 32768
dx = L / N

# k, E11, E22 viennent de la version corrigée d'average_energy_spectra()
k, E11, E22 = average_energy_spectra()      # k en rad/m (déjà un-sided, DC retiré)
k_eta = k * eta

# ---- Inner scaling (Kolmogorov) ----
prefactor = (epsilon_bar * nu**5)**0.25     # = u_η^2 * η  avec u_η=(ν ε)^{1/4}, η=(ν^3/ε)^{1/4}
E11_dimless = E11 / prefactor
E22_dimless = E22 / prefactor

# Theory plot C_1k^-5/3 line
C1 = 0.52
C1_ = 0.70
k_theory = np.linspace(1e-4, 5, 100)
E11_theory = C1 * (k_theory)**(-5/3)   
E22_theory = C1_ * (k_theory)**(-5/3)
# masque: on garde kη >= 8e-3
mask = k_eta >= 2e-5

plt.figure(figsize=(10,5))
plt.xlim(1e-4, 5)
plt.ylim(1e-13, 1e7)

# données mesurées (masquées)
plt.loglog(k_eta[mask], E11_dimless[mask], label=r'$E_{11}/(\varepsilon\nu^5)^{1/4}$', color='C0')
plt.loglog(k_eta[mask], E22_dimless[mask], label=r'$E_{22}/(\varepsilon\nu^5)^{1/4}$', color='C1')

# lignes théoriques: commence aussi à 8e-3 pour cohérence visuelle
E11_theory = 0.52 * k_theory**(-5/3)
E22_theory = 0.70 * k_theory**(-5/3)
plt.loglog(k_theory, E11_theory, '--', color='k', lw=2, label=r'$C_1 k^{-5/3}$, $C_1\approx0.52$', zorder=10)
plt.loglog(k_theory, E22_theory, ':',  color='k', lw=2, label=r"$C_1' k^{-5/3}$, $C_1'\approx0.70$", zorder=10)
plt.xlabel(r'$k_1\eta$')
plt.ylabel(r'$E_{11,22}(k_1) / \left( \bar{\epsilon} \nu^5 \right)^{1/4}$')
plt.title('Dimensionless 1D energy spectra')
plt.grid(True, which='both', ls=':', alpha=0.6)
plt.legend()
plt.show()


# Kolmogorov inner scaling prefactor
prefactor = (epsilon_bar * nu**5)**0.25
k_eta = k * eta

# Compensated dimensionless spectra
E11_comp = (k_eta**(5/3)) * E11 / prefactor
E22_comp = (k_eta**(5/3)) * E22 / prefactor

# Mask for plotting and measuring
plot_mask = k_eta > 1e-4
plateau_mask = (k_eta >= 1e-3) & (k_eta <= 1e-2)

# Measure pseudo-plateau values
C1_meas = np.mean(E11_comp[plateau_mask])
C2_meas = np.mean(E22_comp[plateau_mask])

# Plotting
plt.figure(figsize=(10, 5))
plt.xlim(1e-4, 5)
plt.ylim(1e-12, 10)

# Raw compensated spectra
line_E11, = plt.loglog(k_eta[plot_mask], E11_comp[plot_mask], label=r'Compensated $E_{11}$', color='C0')
line_E22, = plt.loglog(k_eta[plot_mask], E22_comp[plot_mask], label=r'Compensated $E_{22}$', color='C1')

# Reference theoretical plateaus
plt.axhline(0.52, linestyle='--', color='k', linewidth=2, label=r'Theory plateau $C_1 = 0.52$')
plt.axhline(0.70, linestyle=':', color='k', linewidth=2, label=r'Theory plateau $C_2 = 0.70$')

# Measured pseudo-plateaus
plt.axhline(C1_meas, color=line_E11.get_color(), linestyle=':', label=fr'$C_1^{{\text{{measured}}}} \approx {C1_meas:.3f}$')
plt.axhline(C2_meas, color=line_E22.get_color(), linestyle=':', label=fr"$C_2^{{\text{{measured}}}} \approx {C2_meas:.3f}$")

# Highlight pseudo-plateau range
plt.axvspan(1e-3, 1e-2, color='gray', alpha=0.15, label='"Pseudo-plateau" range')

# Labels and layout
plt.xlabel(r'$k_1\eta$')
plt.ylabel(r'$(k_1\eta)^{5/3} \cdot E_{ii}(k_1) / (\varepsilon \nu^5)^{1/4}$')
plt.title('Compensated Dimensionless 1D Energy Spectra')
plt.legend()
plt.grid(True, which='major', linestyle='-', linewidth=0.7)
plt.grid(True, which='minor', linestyle=':', linewidth=0.7, alpha=0.6)
plt.minorticks_on()
plt.tight_layout()
plt.show()
