import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import scipy.constants as const
from arc import Rubidium87

#
Nlvl = 3
g = basis(Nlvl, 0)   # |1> = |g> = 5s
e = basis(Nlvl, 1)   # |2> = |e> = 5p
r = basis(Nlvl, 2)   # |3> = |r> = 46s

Pg, Pe, Pr = [k*k.dag() for k in (g, e, r)]

# parameters
Omega_p_MHz = 0.001
Omega_p_SI   = 2*const.pi*Omega_p_MHz*1e6  # rad/s

Omega_c_MHz  = 5.5
Delta_c_MHz  = -1.9        # 耦合失谐

gamma_e_MHz  = 6.0
gamma_r_MHz  = 0.3

#
e_charge = const.e
a0       = const.physical_constants["Bohr radius"][0]
d21 = (1/np.sqrt(3.0))*5.177*e_charge*a0   # [C·m]

lambda_p = 780e-9
k_p = 2*const.pi/lambda_p

CD = 1.5e13

coef = 2 * CD * d21**2 * k_p / (const.hbar * const.epsilon_0 * Omega_p_SI)

# Δp
detunings_MHz = np.linspace(-8.0, 8.0, 801)
T_list = []

for Delta_p_MHz in detunings_MHz:
    H0 = (- Delta_p_MHz * Pe
          - (Delta_p_MHz + Delta_c_MHz) * Pr)

    H_int = (0.5*Omega_p_MHz * (g*e.dag() + e*g.dag())
             + 0.5*Omega_c_MHz * (e*r.dag() + r*e.dag()))

    H = H0 + H_int

# master eq
    c_ops = []
    c_ops.append(np.sqrt(gamma_e_MHz) * g * e.dag())
    c_ops.append(np.sqrt(gamma_r_MHz) * Pr)

    rho_ss = steadystate(H, c_ops)

    # ρ21 = <2|ρ|1>
    rho21 = rho_ss[1, 0]
    im_rho21 = np.imag(rho21)

    T = np.exp(coef * im_rho21)
    T_list.append(T)

T_list = np.array(T_list)

plt.figure(figsize=(7,5))
plt.plot(detunings_MHz, T_list)
plt.xlabel("Probe detuning Δp (MHz)")
plt.ylabel("Transmission")
plt.title("EIT, Ω_μ = 0, using Eq.(9)")
plt.grid(True)
plt.tight_layout()
plt.show()

