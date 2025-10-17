import numpy as np
from scipy.constants import g

def normal_flow(A, S_0, n, B):
    R_ = R(A, B)
    
    Q = A * R_**(2/3) * np.abs(S_0)**0.5 / n
    if S_0 < 0:
        Q = -Q
                    
    return Q
    
def normal_area(Q, A_guess, S_0, n, B, tolerance = 1e-3):
    Q_guess = normal_flow(A_guess, S_0, n, B)
        
    while abs(Q_guess - Q) >= tolerance:
        error = (Q_guess - Q) / Q
        A_guess -= 0.1 * error * A_guess
        Q_guess = normal_flow(A_guess, S_0, n, B)
        
    return A_guess

def effective_roughness(depth: float, steepness, wet_roughness, dry_roughness, wet_depth):
    transition_depth = steepness * wet_depth
    
    if depth <= wet_depth:
        return wet_roughness
    if transition_depth == 0 or depth - wet_depth > transition_depth:
        return dry_roughness
    else:
        return wet_roughness + (dry_roughness - wet_roughness) * (depth - wet_depth) / transition_depth
    
def Sf(A: float, Q: float, n: float, B: float) -> float:
    """
    Computes the friction slope using Manning's equation.
    
    Parameters
    ----------
    A : float
        The cross-sectional flow area.
    Q : float
        The discharge.
        
    Returnsr
    -------
    float
        The computed friction slope.
        
    """
    R_ = R(A, B)
    return n**2 * A**-2 * R_**(-4/3) * Q * np.abs(Q)

def Sc(A: float, Q: float, n: float, B: float, rc: float) -> float:
    """
    Computes the energy gradient due to transverse circulation.
    
    Parameters
    ----------
    A : float
        The cross-sectional flow area.
    Q : float
        The discharge.
        
    Returns
    -------
    float
        The computed friction slope.
        
    """
    h = A/B
    Fr = froude_num(A, Q, B)
    R_ = R(A, B)
    C = R_**(1/6) / n
    f = 8 * g / C**2
    
    numerator = (2.86 * np.sqrt(f) + 2.07 * f) * h**2 * Fr**2
    denominator = (0.565 + np.sqrt(f)) * rc**2
    Sc = numerator/denominator
    return Sc

def dSc_dA(A, Q, n, B, rc):
    h = A / B
    Fr = froude_num(A, Q, B)
    R_ = R(A, B)
    dR_dA_ = dR_dA(A, B)
    C = R_**(1/6) / n
    f = 8 * g / C**2
    
    dh_dA = 1.0 / B
    dFr_dA = -1.5 * Fr / A
    df_dA = -(8.0/3.0) * g * n**2 * R_**(-4.0/3.0) * dR_dA_
    
    sqrtf = np.sqrt(f)
    num = (2.86*sqrtf + 2.07*f) * h**2 * Fr**2
    den = (0.565 + sqrtf) * rc**2
    
    dnum_dA = (2.86/(2*sqrtf)*df_dA + 2.07*df_dA) * h**2 * Fr**2 \
            + (2.86*sqrtf + 2.07*f) * (2*h*dh_dA * Fr**2 + h**2 * 2*Fr*dFr_dA)
    dden_dA = (1.0/(2*sqrtf) * df_dA) * rc**2
    
    return (dnum_dA*den - num*dden_dA) / (den**2)

def dSc_dQ(A, Q, n, B, rc):
    h = A / B
    Fr = froude_num(A, Q, B)
    R_ = R(A, B)
    C = R_**(1/6) / n
    f = 8 * g / C**2
    
    dFr_dQ = Fr / Q
    
    sqrtf = np.sqrt(f)
    num = (2.86*sqrtf + 2.07*f) * h**2 * Fr**2
    den = (0.565 + sqrtf) * rc**2
    
    dnum_dQ = (2.86*sqrtf + 2.07*f) * h**2 * 2*Fr*dFr_dQ
    dden_dQ = 0.0
    
    return (dnum_dQ*den - num*dden_dQ) / (den**2)

def dSc_dn(A, Q, n, B, rc):
    h = A / B
    Fr = froude_num(A, Q, B)
    R_ = R(A, B)
    C = R_**(1/6) / n
    f = 8 * g / C**2
    
    df_dn = 16.0 * g * n / R_**(1/3)
    
    sqrtf = np.sqrt(f)
    num = (2.86*sqrtf + 2.07*f) * h**2 * Fr**2
    den = (0.565 + sqrtf) * rc**2
    
    dnum_dn = (2.86/(2*sqrtf)*df_dn + 2.07*df_dn) * h**2 * Fr**2
    dden_dn = (1.0/(2*sqrtf) * df_dn) * rc**2
    
    return (dnum_dn*den - num*dden_dn) / (den**2)

def froude_num(A: float, Q: float, B: float):
    V = Q / A
    h = A / B

    return V / np.sqrt(g*h)

def dSf_dA(A: float, Q: float, n: float, B: float) -> float:
    """Computes the partial derivative of Sf w.r.t. A.

    Args:
        A (float): Cross-sectional flow area
        Q (float): Flow rate
        n (float): Manning's coefficient
        B (float): Cross-sectional width

    Returns:
        float: dSf/dA
    """
    R_ = R(A, B)
    dSf_dA = -2 * n**2 * A**-3 * R_**(-4/3) * Q * abs(Q)
    dSf_dR = (-4/3) * n**2 * A**-2 * R_**(-4/3 - 1) * Q * abs(Q)

    return dSf_dA + dSf_dR * dR_dA(A, B)

def dSf_dQ(A: float, Q: float, n: float, B: float) -> float:
    """Computes the partial derivative of Sf w.r.t. Q.

    Args:
        A (float): Cross-sectional flow area
        Q (float): Flow rate
        n (float): Manning's coefficient
        B (float): Cross-sectional width

    Returns:
        float: dSf/dQ
    """
    R_ = R(A, B)
    
    d_Sf = 2 * abs(Q) * (n / (A * R_**(2/3)))**2
    
    return d_Sf

def R(A, B, approx = False):
    if approx:
        P = B
    else:
        P = B + 2 * A/B
                
    return A / P

def dR_dA(A, B, approx = False):
    """
    dR/dA where R = A / P and P = B + 2A/B (unless approx=True, then P=B).
    Correct formula: dR/dA = (P - A*dP/dA) / P**2
    """
    if approx:
        P = B
        dP_dA = 0.0
    else:
        P = B + 2.0 * A / B
        dP_dA = 2.0 / B

    return (P - A * dP_dA) / (P**2)
    
def dSf_dn(A, Q, n, B):
    R_ = R(A, B)
    return 2 * n * A**-2 * R_**(-4/3) * Q * abs(Q)

def dn_dh(depth: float, steepness: float, roughness: float, dry_roughness: float, wet_depth: float):
    transition_depth = steepness * wet_depth
    
    if depth <= wet_depth or depth - wet_depth > transition_depth:
        return 0
    else:
        return (dry_roughness - roughness) / transition_depth
    
def dQn_dA(A, S, n, B):
    R_ = R(A, B)
    dQn_dR = (2/3) * A * R_**(2/3 - 1) * abs(S)**0.5 / n
    
    dQn_dA = R_**(2/3) * abs(S)**0.5 / n + dQn_dR * dR_dA(A, B)
    if S < 0:
        dQn_dA = -dQn_dA
        
    return dQn_dA

def dQn_dn(A, S_0, n, B):
    R_ = R(A, B)
    
    dQn_dn = -1 * A * R_**(2/3) * abs(S_0)**0.5 * n**-2
    
    if S_0 < 0:
        dQn_dn = -dQn_dn
        
    return dQn_dn
