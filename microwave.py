import numpy as np
import matplotlib.pyplot as plt
from qutip import (basis, liouvillian, spre, spost, steadystate, mesolve,rand_dm)
import scipy.constants as const
from sympy.physics.wigner import wigner_3j, wigner_6j
import scipy.sparse.linalg as sp_linalg
from math import sqrt

#1-10
Nlvl = 10
g = basis(Nlvl, 0)   # |1> = |g> = 5s
e = basis(Nlvl, 1)   # |2> = |e> = 5p
r = basis(Nlvl, 2)   # |3> = |r> = 46S1/2, F=2,mF=2
m = basis(Nlvl, 3)   # |4> = 45P...
x = basis(Nlvl, 4)   # |5>
y = basis(Nlvl, 5)   # |6>
z = basis(Nlvl, 6)   # |7>
a = basis(Nlvl, 7)   # |8>
b = basis(Nlvl, 8)   # |9>
c = basis(Nlvl, 9)   # |10>

Pg, Pe, Pr, Pm, Px, Py, Pz, Pa, Pb, Pc = [k*k.dag() for k in (g, e, r, m, x, y, z, a, b, c)]
basis_list = [g, e, r, m, x, y, z, a, b, c]

#L,J,F,mF
level_info = {
    1: dict(L=0, J=0.5, F=3, mF=3),   # |1> 5S1/2(F=2,mF=2)
    2: dict(L=1, J=1.5, F=2, mF=2),   # |2> 5P3/2(F=3,mF=3)

    # 46S1/2(|3>,|6>,|7>,|10>)
    3:  dict(L=0, J=0.5, F=2, mF=2),
    6:  dict(L=0, J=0.5, F=2, mF=1),
    7:  dict(L=0, J=0.5, F=1, mF=1),
    10: dict(L=0, J=0.5, F=2, mF=0),

    # 45P1/2(|4>,|5>)
    4: dict(L=1, J=0.5, F=2, mF=2),#2
    5: dict(L=1, J=0.5, F=1, mF=1),#

    # 45P3/2(|8>,|9>)
    8: dict(L=1, J=1.5, F=2, mF=1),
    9: dict(L=1, J=1.5, F=1, mF=1),
}

#C_ij
def C_ij(i, j):
    info_i = level_info[i]
    info_j = level_info[j]

    Li, Ji, Fi, mFi = info_i['L'], info_i['J'], info_i['F'], info_i['mF']
    Lj, Jj, Fj, mFj = info_j['L'], info_j['J'], info_j['F'], info_j['mF']

    sixj1 = wigner_6j(Ji, Jj, 1, Fj, Fi, 3/2)   # {Ji Jj 1; Fj Fi I}
    sixj2 = wigner_6j(Li, Lj, 1, Jj, Ji, 1/2)   # {Li Lj 1; Jj Ji S}
    threej_L = wigner_3j(Lj, 1, Li, 0, 0, 0)    # (Lj 1 Li; 0 0 0)

    term_qm1 = wigner_3j(Fj, 1, Fi, mFj, -1, -mFi)  # q=-1
    term_qp1 = wigner_3j(Fj, 1, Fi, mFj, +1, -mFi)  # q=+1

    phase = (-1)**(-mFi)
    pref = phase * sqrt((2*Fi+1)*(2*Fj+1))

    C = pref * float( sixj1 * sixj2 * threej_L *(term_qm1 - term_qp1) )
    return float(C)

#sum_mu
# i={3,5,7,9} → i,i+1
mw_edges1 = [(3,4), (5,6), (7,8), (9,10)]
# i=3..8 → i,i+2
mw_edges2 = [(3,5), (4,6), (5,7), (6,8), (7,9), (8,10)]
# i={4,6} → i,i+3
mw_edges3 = [(4,7), (6,9)]

#parameters
Omega_p_MHz = 0.01
Omega_p_SI  = 2 * const.pi * Omega_p_MHz * 1e6  # rad/s

Omega_c_MHz = 5.5
Delta_c_MHz = -1.9                    # 耦合失谐

Omega_mu_MHz = 0.001*const.pi
Delta_mu_MHz = -0.05                   # 微波失谐（可微调）

gamma_p_MHz   = 0.33                  # 0.33
gamma_rel_MHz = 0.08                  # 0.14
gamma_Ry_MHz  = 0.18                  # 0.3
gamma_e_MHz   = 6                   # 6

# eq.9
e_charge = const.e
a0       = const.physical_constants["Bohr radius"][0]
d21 = (1/np.sqrt(3.0))*5.177*e_charge*a0   # [C·m]

lambda_p = 780e-9
k_p = 2 * const.pi / lambda_p
CD  = 1.5e13   # [m^-2]

coef = 2 * CD * d21**2 * k_p / (const.hbar * const.epsilon_0 * Omega_p_SI)

#H_mu
def build_H_mu(Omega_mu_MHz):
    H_mu1 = 0
    for (i, j) in mw_edges1:
        C = C_ij(i, j)
        Omega_ij = Omega_mu_MHz * C
        ki = basis_list[i-1]
        kj = basis_list[j-1]
        H_mu1 += 0.5 * Omega_ij * (ki * kj.dag() + kj * ki.dag())


    H_mu2 = 0
    for (i, j) in mw_edges2:
        C = C_ij(i, j)
        Omega_ij = Omega_mu_MHz * C
        ki = basis_list[i-1]
        kj = basis_list[j-1]
        H_mu2 += 0.5 * Omega_ij * (ki * kj.dag() + kj * ki.dag())


    H_mu3 = 0
    for (i, j) in mw_edges3:
        C = C_ij(i, j)
        Omega_ij = Omega_mu_MHz * C  
        ki = basis_list[i-1]
        kj = basis_list[j-1]
        H_mu3 += 0.5 * Omega_ij * (ki * kj.dag() + kj * ki.dag())


    H_mu=H_mu1+H_mu2+H_mu3

    return H_mu

H_mu = build_H_mu(Omega_mu_MHz)

#gamma
gamma_mat = np.array([
    [0,            gamma_p_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz],
    [gamma_p_MHz, 0,            gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz],
    [gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, 0, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz],
    [gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_Ry_MHz, 0, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz],
    [gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, 0, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz],
    [gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, 0, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz],
    [gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, 0, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz],
    [gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, 0, gamma_Ry_MHz, gamma_Ry_MHz],
    [gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, 0, gamma_Ry_MHz],
    [gamma_rel_MHz + gamma_Ry_MHz, gamma_rel_MHz + gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, gamma_Ry_MHz, 0]
])



#L_d
def build_Ld_super(gamma_mat):
    Ld = 0
    for i in range(Nlvl):
        Pi = basis(Nlvl, i) * basis(Nlvl, i).dag()
        for j in range(Nlvl):
            if i == j:
                continue
            gamma_ij = gamma_mat[i, j]
            if gamma_ij == 0:
                continue
            Pj = basis(Nlvl, j) * basis(Nlvl, j).dag()
            Ld += -gamma_ij * spre(Pi) * spost(Pj)
    return Ld

Ld_super = build_Ld_super(gamma_mat)

#L(ρ)
c_ops = []
c_ops.append(np.sqrt(gamma_e_MHz) * g * e.dag())   # 5P -> 5S

#Δp
detunings_MHz = np.linspace(-10.0, 10.0, 501)
T_list = []

for Delta_p_MHz in detunings_MHz:
    H0 = (- Delta_p_MHz * Pe
          - (Delta_p_MHz + Delta_c_MHz) * (Pr + Py + Pz + Pc)
          - (Delta_p_MHz + Delta_c_MHz - Delta_mu_MHz) * (Pm + Px + Pa + Pb))

    H_int = (0.5 * Omega_p_MHz * (g * e.dag() + e * g.dag())
             + 0.5 * Omega_c_MHz * (e * r.dag() + r * e.dag()))

    H = H0 + H_int + H_mu

#master eq
    L = liouvillian(H, c_ops) + Ld_super

    rho_ss = steadystate(L, method='svd')

    rho21 = rho_ss[1, 0]
    im_rho21 = np.imag(rho21)
    arg = coef * im_rho21
    #arg = np.clip(arg, -5.0, -0.5)
    T = np.exp(arg)

    T_list.append(T)

T_list = np.array(T_list)

plt.figure(figsize=(8, 6))
plt.plot(detunings_MHz, T_list)
plt.xlabel("Probe detuning(MHz)")
plt.ylabel("Transmission")
plt.title("Ω_μ/2π = 0 MHz")
plt.grid(True)
plt.tight_layout()
plt.show()