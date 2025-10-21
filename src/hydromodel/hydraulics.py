import numpy as np
from scipy.constants import g

def normal_flow(A, S_0, n, R):
    Q = A * R**(2/3) * np.abs(S_0)**0.5 / n
    if S_0 < 0:
        Q = -Q
                    
    return Q
    
def normal_area(Q, A_guess, S_0, n, R, tolerance = 1e-3):
    raise ValueError("hydraulics.normal_area is WIP.")
    Q_guess = normal_flow(A=A_guess, S_0=S_0, n=n, R=R)
        
    while abs(Q_guess - Q) >= tolerance:
        error = (Q_guess - Q) / Q
        A_guess -= 0.1 * error * A_guess
        Q_guess = normal_flow(A=A_guess, S_0=S_0, n=n, R=R)
        
    return A_guess

def effective_roughness(depth: float, wet_depth: float, wet_roughness: float, dry_roughness: float, steepness: float):
    transition_depth = steepness * wet_depth
    
    if depth <= wet_depth:
        return wet_roughness
    if transition_depth == 0 or depth - wet_depth > transition_depth:
        return dry_roughness
    else:
        return wet_roughness + (dry_roughness - wet_roughness) * (depth - wet_depth) / transition_depth
    
def Sf(A: float, Q: float, n: float, R: float) -> float:
    """Computes friction slope using Manning's equation.

    Args:
        A (float): Cross-sectional flow area.
        Q (float): Flow rate
        n (float): Manning's roughness coefficient.
        R (float): Hydraulic radius.

    Returns:
        float: Friction slope.
    """
    return n**2 * A**-2 * R**(-4/3) * Q * np.abs(Q)

def Sc(h: float, A: float, Q: float, n: float, R: float, rc: float) -> float:
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
        The computed slope.
        
    """
    Fr = froude_num(h=h, A=A, Q=Q)
    f = darcey_weisbach_f(n=n, R=R)
    
    numerator = (2.86 * np.sqrt(f) + 2.07 * f) * h**2 * Fr**2
    denominator = (0.565 + np.sqrt(f)) * rc**2
    Sc = numerator/denominator
    return Sc

def darcey_weisbach_f(n: float, R: float):
    """Computes Darcey-Weisbach's friction factor.

    Args:
        n (float): Manning's roughness coefficient.
        R (float): Hydraulic radius.

    Returns:
        float: Darcey-Weisbach's friction factor.
    """
    C = R**(1/6) / n
    f = 8 * g / C**2
    return f

def dSc_dA(h, A, Q, n, R, rc, dR_dA, T):
    Fr = froude_num(h=h, A=A, Q=Q)
    
    C = R**(1/6) / n
    f = 8 * g / C**2
    
    dh_dA = 1./T
    dFr_dA_ = dFr_dA(h=h, A=A, Q=Q, T=T)
    df_dA = -(8.0/3.0) * g * n**2 * R**(-4.0/3.0) * dR_dA
    
    sqrtf = np.sqrt(f)
    num = (2.86*sqrtf + 2.07*f) * h**2 * Fr**2
    den = (0.565 + sqrtf) * rc**2
    
    dnum_dA = (2.86/(2*sqrtf)*df_dA + 2.07*df_dA) * h**2 * Fr**2 \
            + (2.86*sqrtf + 2.07*f) * (2*h*dh_dA * Fr**2 + h**2 * 2*Fr*dFr_dA_)
    dden_dA = (1.0/(2*sqrtf) * df_dA) * rc**2
    
    return (dnum_dA*den - num*dden_dA) / (den**2)

def dSc_dQ(h, A, Q, n, R, rc):
    Fr = froude_num(h=h, A=A, Q=Q)
    C = R**(1/6) / n
    f = 8 * g / C**2
    
    dFr_dQ_ = dFr_dQ(h=h, A=A, Q=Q)
    
    sqrtf = np.sqrt(f)
    num = (2.86*sqrtf + 2.07*f) * h**2 * Fr**2
    den = (0.565 + sqrtf) * rc**2
    
    dnum_dQ = (2.86*sqrtf + 2.07*f) * h**2 * 2*Fr*dFr_dQ_
    dden_dQ = 0.0
    
    return (dnum_dQ*den - num*dden_dQ) / (den**2)

def dSc_dn(h, A, Q, n, R, rc):
    Fr = froude_num(h=h, A=A, Q=Q)
    C = R**(1/6) / n
    f = 8 * g / C**2
    
    df_dn = 16.0 * g * n / R**(1/3)
    
    sqrtf = np.sqrt(f)
    num = (2.86*sqrtf + 2.07*f) * h**2 * Fr**2
    den = (0.565 + sqrtf) * rc**2
    
    dnum_dn = (2.86/(2*sqrtf)*df_dn + 2.07*df_dn) * h**2 * Fr**2
    dden_dn = (1.0/(2*sqrtf) * df_dn) * rc**2
    
    return (dnum_dn*den - num*dden_dn) / (den**2)

def froude_num(h: float, A: float, Q: float):
    V = Q/A
    return V / np.sqrt(g*h)

def dFr_dA(h: float, A: float, Q: float, T: float):
    V = Q/A
    dFr_dh = -0.5 * V * (g*h)**(-1.5) * g
    dh_dA = 1./T
    return dFr_dh * dh_dA

def dFr_dQ(h: float, A: float, Q: float):
    dFr_dV = 1. / np.sqrt(g*h)
    dV_dQ = 1./A
    return dFr_dV * dV_dQ

def dSf_dA(A: float, Q: float, n: float, R: float, dR_dA: float) -> float:
    """Computes the partial derivative of Sf w.r.t. A.

    Args:
        A (float): Cross-sectional flow area
        Q (float): Flow rate
        n (float): Manning's coefficient
        B (float): Cross-sectional width

    Returns:
        float: dSf/dA
    """
    dSf_dA = -2 * n**2 * A**-3 * R**(-4/3) * Q * abs(Q)
    dSf_dR = (-4/3) * n**2 * A**-2 * R**(-4/3 - 1) * Q * abs(Q)

    return dSf_dA + dSf_dR * dR_dA

def dSf_dQ(A: float, Q: float, n: float, R: float) -> float:
    """Computes the partial derivative of Sf w.r.t. Q.

    Args:
        A (float): Cross-sectional flow area
        Q (float): Flow rate
        n (float): Manning's coefficient
        B (float): Cross-sectional width

    Returns:
        float: dSf/dQ
    """
    return 2 * abs(Q) * (n / (A * R**(2/3)))**2
    
def dSf_dn(A, Q, n, R):
    return 2 * n * A**-2 * R**(-4/3) * Q * abs(Q)

def dn_dh(depth: float, steepness: float, roughness: float, dry_roughness: float, wet_depth: float):
    transition_depth = steepness * wet_depth
    
    if depth <= wet_depth or depth - wet_depth > transition_depth:
        return 0
    else:
        return (dry_roughness - roughness) / transition_depth
    
def dQn_dA(A, S, n, R, dR_dA):
    dQn_dR = (2/3) * A * R**(2/3 - 1) * abs(S)**0.5 / n
    
    dQn_dA = R**(2/3) * abs(S)**0.5 / n + dQn_dR * dR_dA
    if S < 0:
        dQn_dA = -dQn_dA
        
    return dQn_dA

def dQn_dn(A, S_0, n, R):
    dQn_dn = -1 * A * R**(2/3) * abs(S_0)**0.5 * n**-2
    
    if S_0 < 0:
        dQn_dn = -dQn_dn
        
    return dQn_dn
