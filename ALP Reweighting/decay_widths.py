import numpy as np
from particle import Particle
from scipy import constants

# Constants
hbar = 4.135667696e-21 # MeV⋅Hz−1

m_e = Particle.from_pdgid(11).mass # +- 0.00000000015 MeV, https://pdglive.lbl.gov/Particle.action?node=S003
m_mu = Particle.from_pdgid(13).mass # MeV
m_tau = Particle.from_pdgid(15).mass # +- 0.00000000015 MeV, https://pdglive.lbl.gov/Particle.action?node=S003

m_u = Particle.from_pdgid(2).mass
m_d = Particle.from_pdgid(1).mass 
m_c = Particle.from_pdgid(4).mass # +- 22 MeV, https://pdglive.lbl.gov/DataBlock.action?node=Q004M
m_s = Particle.from_pdgid(3).mass # +- 2 MeV, https://pdglive.lbl.gov/DataBlock.action?node=Q004M
m_b = Particle.from_pdgid(5).mass  # +30,-20 MeV, https://pdglive.lbl.gov/Particle.action?node=Q005
m_t = Particle.from_pdgid(6).mass  # +- 300 MeV, https://pdglive.lbl.gov/Particle.action?node=Q007
m_Z = Particle.from_pdgid(23).mass  # +- 2.0 MeV, https://pdglive.lbl.gov/Particle.action?node=S043
m_W = Particle.from_pdgid(24).mass  # +- 12 MeV https://pdglive.lbl.gov/Particle.action?node=S043
m_pi = Particle.from_pdgid(211).mass  # +- 0.00018 MeV https://pdglive.lbl.gov/Particle.action?node=S008
m_pi0 = Particle.from_pdgid(111).mass  # +- 0.00018 MeV https://pdglive.lbl.gov/Particle.action?node=S008
m_K = Particle.from_pdgid(321).mass   # +- 0.016 MeV https://pdglive.lbl.gov/Particle.action?node=S010
m_K0 = Particle.from_pdgid(130).mass   # +- 0.016 MeV https://pdglive.lbl.gov/Particle.action?node=S010

LAMBDA = 10**6 # MeV, energy scale (1 TeV, used in https://arxiv.org/pdf/2202.03447.pdf)

PI = np.pi
f_A = LAMBDA

# v = 246220 # MeV, https://pdg.lbl.gov/2023/reviews/rpp2022-rev-standard-model.pdf
y_t = 0.95 # top yukawa coupling, measured in https://pdglive.lbl.gov/DataBlock.action?node=S126YTC
alpha_t = y_t/(4*PI)


# CKM matrix
def deg2rad(deg):
    return deg * np.pi / 180

theta_12 = 13.04 # degrees https://en.wikipedia.org/wiki/Cabibbo%E2%80%93Kobayashi%E2%80%93Maskawa_matrix#%22Standard%22_parameters
theta_13 = 0.201 # degrees
theta_23 = 2.38 # degrees
delta_13 = 68.8 # degrees

theta_12 = deg2rad(theta_12)
theta_13 = deg2rad(theta_13)
theta_23 = deg2rad(theta_23)    
delta_13 = deg2rad(delta_13)

exp = np.exp(-1j * delta_13)

s12 = np.sin(theta_12)
s13 = np.sin(theta_13)
s23 = np.sin(theta_23)

c12 = np.cos(theta_12)
c13 = np.cos(theta_13)
c23 = np.cos(theta_23)

V_td = s12*s23 - c12*c23*s13*exp
V_ts = -s23*c12 - s12*c23*s13*exp

# Using the results of Schwartz (Quantum field theory and the Standard Model) using the masses of 
# the W and Z bosons as renormalization conditions.

# Using definitions given in https://arxiv.org/pdf/2202.03447.pdf, p. 6
e = 0.303
s2w = 1-(m_W/m_Z)**2
c2w = (m_W/m_Z)**2
alpha = e**2/(4*np.pi)
g2_muw = e**2/s2w

v = m_W /g2_muw 

m_t = y_t*v/2**0.5
x_t = m_t**2/m_W**2
MU = m_t


# Useful functions, defined in https://arxiv.org/pdf/2202.03447.pdf
def h(x):
    return (1-x+x*np.log(x))/(1-x**2)

def A(mu, Lambda):
    return (9/(64*PI**2)*g2_muw)*x_t*np.log(Lambda**2/mu**2)

def g(x):
    return x*h(x)

def lambda_function(a,b,c):
    return a**2 + b**2 + c**2 -2*a*b - 2*a*c - 2*b*c 

def kQ_sd(c_w, c_phi, c_B):
    # Evaluated at LAMBDA = 1 TeV (https://arxiv.org/pdf/2202.03447.pdf)
    return np.conjugate(V_td)*V_ts*(-c_w*9.7e-3+c_phi*8.2e-3-c_B*3.5e-5)

def tau_p(m, m_a):
    return 4*m**2/m_a**2

def f2(tau):
    if tau >= 1:
        return np.arcsin(1/tau**0.5)**2
    
    # imag = 1/2*np.log((1+(1-tau)**0.5)/(1-(1-tau)**0.5))
    # return (PI/2)**2 + imag**2

    comp = PI/2 + 1/2*(np.log((1+(1-tau)**0.5)/(1-(1-tau)**0.5)))*1j
    comp2 = comp**2
    return comp2

def B_1(tau):
    return 1 - tau*f2(tau)

def B_2(tau):
    return 1 - (tau-1)*f2(tau)

def B_0(m_a):
    tau_u = tau_p(m_u, m_a)
    tau_d = tau_p(m_d, m_a)
    tau_c = tau_p(m_c, m_a)
    tau_s = tau_p(m_s, m_a)
    tau_t = tau_p(m_t, m_a)
    tau_b = tau_p(m_b, m_a)
    tau_e = tau_p(m_e, m_a)
    tau_mu = tau_p(m_mu, m_a)
    
    e_f = 1
    # up, charm, top
    # sum_1 = 3 * (2/3*e_f)**2 * B_1(tau_u) + 3 * (2/3*e_f)**2 * B_1(tau_c) + 3 * (2/3*e_f)**2 * B_1(tau_t)
    sum_1 = 3 * (2/3*e_f)**2 * B_1(tau_c) + 3 * (2/3*e_f)**2 * B_1(tau_t)
    # down, strange, bottom, electron, muon
    # sum_2 = 3 * (-1/3*e_f)**2 * B_1(tau_d) + 3 * (-1/3*e_f)**2 * B_1(tau_s) + 3 * (-1/3*e_f)**2 * B_1(tau_b) + e_f**2 * B_1(tau_e) + e_f**2 * B_1(tau_mu)
    sum_2 = 3 * (-1/3*e_f)**2 * B_1(tau_s) + 3 * (-1/3*e_f)**2 * B_1(tau_b) + e_f**2 * B_1(tau_e) + e_f**2 * B_1(tau_mu)
    return sum_1 - sum_2

def c_yy(c_w, c_phi, c_B,m_a):    
    tau_W = tau_p(m_W, m_a)
    part_cw = c_w*(s2w+2*alpha/PI*B_2(tau_W))
    part_cb = c_B*c2w
    part_cphi = c_phi*alpha/(4*PI)*(B_0(m_a) - (m_a**2)/(m_pi0**2-m_a**2))
    return part_cw + part_cb - part_cphi

def c_yy_tree(c_w, c_phi, c_B,m_a):
    return c_B * c2w + c_w * s2w

def c_ll(c_w, c_phi, c_B, lambda_E, lepton='e'):
    if lepton == 'e':
        m_l = m_e
    elif lepton == 'mu':
        m_l = m_mu
    elif lepton == 'tau':
        m_l = m_tau
    
    f_a = lambda_E

    # c_w only
    part_cw = 3*alpha/(4*PI)*np.log(f_a/m_W)*3*c_w/s2w + 6*alpha/PI*c_w*s2w*np.log(m_W/m_l)
    # c_B only
    part_cb = 3*alpha/(4*PI)*np.log(f_a/m_W)*5*c_B/c2w + 6*alpha/PI*c_B*c2w*np.log(m_W/m_l)

    return c_phi + part_cw + part_cb

# Scalar production partial widths
def decay_width_pi_a(c_w, c_phi, c_B, m_a, lambda_E):
    f_a = lambda_E # Condition used in https://arxiv.org/pdf/2202.03447.pdf
    lambda_pi_a = lambda_function(1, m_a**2/m_K**2, m_pi**2/m_K**2)
    kQ2 = np.abs(kQ_sd(c_w, c_phi, c_B))**2
    return m_K**3*kQ2/(64*PI*f_a**2)*lambda_pi_a**0.5*(1-m_pi**2/m_K**2)**2

def decay_width_pi0_a(c_w, c_phi, c_B, m_a, lambda_E):
    f_a = lambda_E # Condition used in https://arxiv.org/pdf/2202.03447.pdf
    lambda_pi_a = lambda_function(1, m_a**2/m_K0**2, m_pi0**2/m_K0**2)
    kQ2 = np.imag(kQ_sd(c_w, c_phi, c_B))**2
    return m_K**3*kQ2/(64*PI*f_a**2)*lambda_pi_a**0.5*(1-m_pi0**2/m_K0**2)**2

def decay_width_pi_S(theta, m_S):
    part_a = theta**2/(16*PI*m_K)
    part_b = np.abs((3*np.conjugate(V_td)*V_ts*m_t**2*m_K**2)/(32*PI**2*v**3))**2
    return part_a*part_b*lambda_function(1, m_S**2/m_K**2, m_pi**2/m_K**2)**0.5

def decay_width_pi0_S(theta, m_S):
    part_a = theta**2/(16*PI*m_K0)
    part_b = np.abs((3*np.conjugate(V_td)*V_ts*m_t**2*m_K0**2)/(32*PI**2*v**3))**2
    return part_a*part_b*lambda_function(1, m_S**2/m_K0**2, m_pi0**2/m_K0**2)**0.5

def decay_width_pi0_a(c_w, c_phi, c_B, m_a, lambda_E):
    f_a = lambda_E # Condition used in https://arxiv.org/pdf/2202.03447.pdf
    lambda_pi_a = lambda_function(1, m_a**2/m_K0**2, m_pi0**2/m_K0**2)
    kQ2 = np.imag(kQ_sd(c_w, c_phi, c_B))**2
    return m_K**3*kQ2/(64*PI*f_a**2)*lambda_pi_a**0.5*(1-m_pi0**2/m_K0**2)**2

def decay_width_kShort_a(c_w, c_phi, c_B, m_a, lambda_E):
    f_a = lambda_E # Condition used in https://arxiv.org/pdf/2202.03447.pdf
    lambda_pi_a = lambda_function(1, m_a**2/m_K0**2, m_pi0**2/m_K0**2)
    kQ2 = np.real(kQ_sd(c_w, c_phi, c_B))**2
    return m_K**3*kQ2/(64*PI*f_a**2)*lambda_pi_a**0.5*(1-m_pi0**2/m_K0**2)**2

def decay_width_pi0_S(theta, m_S):
    part_a = theta**2/(16*PI*m_K0)
    part_b = np.abs((3*np.conjugate(V_td)*V_ts*m_t**2*m_K0**2)/(32*PI**2*v**3))**2
    return part_a*part_b*lambda_function(1, m_S**2/m_K0**2, m_pi0**2/m_K0**2)**0.5

# Scalar decay widths
def decay_width_S_ee(theta, M_S):
    return theta**2*(m_e**2*M_S)/(8*PI*v**2)*(1-4*m_e**2/M_S**2)**1.5

def decay_width_a_ee(c_w, c_phi, c_B, m_a, lambda_E):
    f_a = lambda_E
    if m_a > 2*m_e:
        return c_ll(c_w, c_phi, c_B, lambda_E, lepton='e')**2*m_a*m_e**2/(8*PI*f_a**2)*(1-4*m_e**2/m_a**2)**0.5
    else:
        return 0
    


def decay_width_a_mumu(c_w, c_phi, c_B, m_a, lambda_E):
    f_a = lambda_E
    if m_a > 2*m_mu:
        return c_ll(c_w, c_phi, c_B, lambda_E, lepton='mu')**2*m_a*m_mu**2/(8*PI*f_a**2)*(1-4*m_mu**2/m_a**2)**0.5
    else:
        return 0
    
def decay_width_a_tautau(c_w, c_phi, c_B, m_a, lambda_E):
    f_a = lambda_E
    if m_a > 2*m_tau:
        return c_ll(c_w, c_phi, c_B, lambda_E, lepton='tau')**2*m_a*m_tau**2/(8*PI*f_a**2)*(1-4*m_tau**2/m_a**2)**0.5
    else:
        return 0

def decay_width_a_yy(c_w, c_phi, c_B, m_a, lambda_E):
    f_a = lambda_E
    c = c_yy(c_w, c_phi, c_B, m_a)
    c2 = c * np.conjugate(c)
    c2 = np.abs(c)**2
    return c2*m_a**3/(4*PI*f_a**2)

def decay_width_a_yy_tree(c_w, c_phi, c_B, m_a, lambda_E):
    f_a = lambda_E
    c2 = c_yy_tree(c_w, c_phi, c_B, m_a)**2

    return c2*m_a**3/(4*PI*f_a**2)

def reweighting(tof, m_S, theta, c_w, c_phi, c_B, klong):
    
    m_S = float(m_S)
    
    m_a = m_S
    lambda_E = LAMBDA

    # Using the correct decay width, using the array of bools 'klong' to determine which function to use.
    gamma_pi_a = np.where(klong, decay_width_pi0_a(c_w, c_phi, c_B, m_a, lambda_E), decay_width_pi_a(c_w, c_phi, c_B, m_a, lambda_E))
    gamma_pi_S = np.where(klong, decay_width_pi0_S(theta, m_S), decay_width_pi_S(theta, m_S))

    gamma_S = decay_width_S_ee(theta, m_S)
    gamma_a = decay_width_a_ee(c_w, c_phi, c_B, m_a, lambda_E) + decay_width_a_yy(c_w, c_phi, c_B, m_a, lambda_E)
    gamma_a_ee = decay_width_a_ee(c_w, c_phi, c_B, m_a, lambda_E)

    prod_ratio = gamma_pi_a/gamma_pi_S
    branching_ratio = gamma_a_ee/gamma_a

    return prod_ratio*np.exp(-tof*(gamma_a-gamma_S)/hbar)*branching_ratio

# Functions used to match run, subrun and event numbers between a pkl file and a root file.
# This is verified by comparing the difference in vertex positions between the two file.
def match_RSE_indices(pkl_data, uproot_file,columns=[0,1,2]):
    test_run = pkl_data['run_ls'].to_numpy()
    test_subrun = pkl_data['sub_ls'].to_numpy()
    test_event = pkl_data['evt_ls'].to_numpy()
    
    keys = uproot_file.keys()
    data = uproot_file[keys[0]]
    keys = data.keys()

    runs = data['run'].array()
    subruns = data['sub'].array()
    event = data['evt'].array()
    mc_primary_pdg = data['mc_primary_pdg'].array()

    indices = []
    for i, run in enumerate(test_run):
        run_args = np.where(runs==run)
        subrun_args = np.where(subruns==test_subrun[i])
        event_arg = np.where(event==test_event[i])
        shared_run_sub_arg = np.intersect1d(run_args, subrun_args)
        shared_arg = np.intersect1d(shared_run_sub_arg, event_arg)
        indices.append(shared_arg[0])

    return indices


def check_matching_vtx(vtx_1, vtx_2):
    diff = np.abs(vtx_1-vtx_2)
    mag_diff = diff/np.mean([vtx_1,vtx_2])
    warning_indices = np.where(mag_diff>10**-7)
    return warning_indices