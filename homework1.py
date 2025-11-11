import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fft import fft


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
    Averages structure functions over all 16 pencils in a given direction.

    Args:
        dimension: 'x', 'y', or 'z'
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


max_r = int(5e3 * eta / (L / 32768))  # convert r/η to grid points

# Longitudinal: u from x-pencils
r_vals, D11 = average_structure_functions('x', 0, max_r)

# Transverse: v and w from x-pencils (twice the data)
r_vals_v, D22_v = average_structure_functions('x', 1, max_r)
r_vals_w, D22_w = average_structure_functions('x', 2, max_r)
D22 = (D22_v + D22_w) / 2

# Convert r to r/η
r_eta = r_vals * (L / 32768) / eta

# Compensated structure functions
C2 = (18/55) * 1.6  # longitudinal
C2_prime = (4/3) * C2  # transverse

D11_comp = D11 / ((epsilon_bar * r_vals * (L / 32768))**(2/3))
D22_comp = D22 / ((epsilon_bar * r_vals * (L / 32768))**(2/3))


plt.figure(figsize=(10, 5))
plt.loglog(r_eta, D11, label='D11 (longitudinal)')
plt.loglog(r_eta, D22, label='D22 (transverse)')
plt.xlabel('r / η')
plt.ylabel('Structure function D(r)')
plt.legend()
plt.grid(True)
plt.title('Structure functions D11 and D22')
plt.show()

plt.figure(figsize=(10, 5))
plt.semilogx(r_eta, D11_comp, label='Compensated D11')
plt.semilogx(r_eta, D22_comp, label='Compensated D22')
plt.axhline(C2, color='gray', linestyle='--', label='C2 ≈ 0.52')
plt.axhline(C2_prime, color='gray', linestyle=':', label="C2' ≈ 0.70")
plt.xlabel('r / η')
plt.ylabel('Compensated structure function')
plt.legend()
plt.grid(True)
plt.title('Compensated structure functions')
plt.show()


###################################################################################################
# PART 4 : One-dimensional energy spectra

def compute_spectrum(f):
    f_hat = np.fft.fft(f)
    E_k = 0.5 * np.abs(f_hat)**2
    return E_k[:N//2]

def average_energy_spectra():
    E11_total = np.zeros(N//2)
    E22_total = np.zeros(N//2)
    count = 0

    for i in range(4):
        for j in range(4):
            filepath = get_file_path('x', i, j)
            u, v, w = read_single_file(filepath)
            E11_total += compute_spectrum(u)
            E22_total += compute_spectrum(v)
            E22_total += compute_spectrum(w)
            count += 1

    E11_avg = E11_total / count
    E22_avg = E22_total / (2 * count)
    return E11_avg, E22_avg

# Compute spectra
N = 32768
dx = L / N
E11, E22 = average_energy_spectra()
k_vals = np.fft.fftfreq(N, dx)[:N//2]
k_eta = k_vals * eta
prefactor = (epsilon_bar * nu**5)**0.25

# Dimensionless spectra
E11_dimless = E11 / prefactor
E22_dimless = E22 / prefactor

# Compensated spectra
E11_comp = (k_eta**(5/3)) * E11_dimless
E22_comp = (k_eta**(5/3)) * E22_dimless

# Plotting
plt.figure(figsize=(10, 5))
plt.loglog(k_eta, E11_dimless, label='E11 / (ε ν⁵)¹/⁴')
plt.loglog(k_eta, E22_dimless, label='E22 / (ε ν⁵)¹/⁴')
plt.xlabel('k η')
plt.ylabel('Dimensionless Energy Spectrum')
plt.title('Dimensionless Energy Spectra')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.semilogx(k_eta, E11_comp, label='Compensated E11')
plt.semilogx(k_eta, E22_comp, label='Compensated E22')
plt.axhline(0.52, color='gray', linestyle='--', label='C1 ≈ 0.52')
plt.axhline(0.70, color='gray', linestyle=':', label='C2 ≈ 0.70')
plt.xlabel('k η')
plt.ylabel('Compensated Spectrum')
plt.title('Compensated Energy Spectra')
plt.legend()
plt.grid(True)
plt.show()


