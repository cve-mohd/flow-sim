import numpy as np
from scipy.constants import g

def normal_flow(bed_slope, area: float = None, roughness: float = None, hydraulic_radius: float = None, K: float = None):
    if K is None:
        K = conveyance(A=area, n=roughness, R=hydraulic_radius)
        
    Q = K * np.abs(bed_slope)**0.5
    
    if bed_slope < 0:
        Q = -Q
                    
    return Q

def conveyance(A: float, n: float, R: float) -> float:
    """Computes conveyance.

    Args:
        A (float): Flow area.
        n (float): Roughness.
        R (float): Hydraulic radius.

    Returns:
        float: K
    """
    return A * R**(2/3) / n

def dK_dA_(A, n, R, dR_dA):
    """Derivative of conveyance w.r.t. flow area.

    Args:
        A (float): Flow area.
        n (float): Roughness.
        R (float): Hydraulic radius.
        dR_dA (float): dR/dA.

    Returns:
        float: dK/dA
    """
    return (R**(2/3) + A * 2./3. * R**(2/3-1) * dR_dA) / n
    
def Sf(Q: float, A: float = None, n: float = None, R: float = None, K: float = None) -> float:
    """Computes friction slope using Manning's equation.

    Args:
        A (float): Cross-sectional flow area.
        Q (float): Flow rate
        n (float): Manning's roughness coefficient.
        R (float): Hydraulic radius.

    Returns:
        float: Friction slope.
    """
    if K is None:
        K = conveyance(A=A, n=n, R=R)
        
    return Q * np.abs(Q) / K**2
    
def dSf_dA(Q: float, A: float = None, n: float = None, R: float = None, dR_dA: float = None, K: float = None, dK_dA: float = None) -> float:
    """Computes the partial derivative of Sf w.r.t. A.

    Args:
        A (float): Cross-sectional flow area
        Q (float): Flow rate
        n (float): Manning's coefficient
        B (float): Cross-sectional width

    Returns:
        float: dSf/dA
    """
    if K is None or dK_dA is None:
        K = conveyance(A=A, n=n, R=R)
        dK_dA = dK_dA_(A=A, n=n, R=R, dR_dA=dR_dA)
        
    return -2 * Sf(Q=Q, K=K) * (dK_dA / K)

def dSf_dQ(Q: float, A: float = None, n: float = None, R: float = None, K: float = None) -> float:
    """Computes the partial derivative of Sf w.r.t. Q.

    Args:
        A (float): Cross-sectional flow area
        Q (float): Flow rate
        n (float): Manning's coefficient
        B (float): Cross-sectional width

    Returns:
        float: dSf/dQ
    """
    if K is None:
        K = conveyance(A=A, n=n, R=R)
        
    return 2 * abs(Q) / K**2

def Sc(h: float, T: float, A: float, Q: float, n: float, R: float, rc: float) -> float:
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
    Fr = froude_num(T=T, A=A, Q=Q)
    f = darcey_weisbach_f(n=n, R=R)
    
    numerator = (2.86 * np.sqrt(f) + 2.07 * f) * h**2 * Fr**2
    denominator = (0.565 + np.sqrt(f)) * rc**2
    Sc = numerator/denominator
    return Sc

def dSc_dA(h, A, Q, n, R, rc, dR_dA, T):
    Fr = froude_num(T=T, A=A, Q=Q)
    
    C = R**(1/6) / n
    f = 8 * g / C**2
    
    dh_dA = 1./T
    dFr_dA_ = dFr_dA(A=A, Q=Q, T=T)
    df_dA = -(8.0/3.0) * g * n**2 * R**(-4.0/3.0) * dR_dA
    
    sqrtf = np.sqrt(f)
    num = (2.86*sqrtf + 2.07*f) * h**2 * Fr**2
    den = (0.565 + sqrtf) * rc**2
    
    dnum_dA = (2.86/(2*sqrtf)*df_dA + 2.07*df_dA) * h**2 * Fr**2 \
            + (2.86*sqrtf + 2.07*f) * (2*h*dh_dA * Fr**2 + h**2 * 2*Fr*dFr_dA_)
    dden_dA = (1.0/(2*sqrtf) * df_dA) * rc**2
    
    return (dnum_dA*den - num*dden_dA) / (den**2)

def dSc_dQ(h, T, A, Q, n, R, rc):
    Fr = froude_num(T=T, A=A, Q=Q)
    C = R**(1/6) / n
    f = 8 * g / C**2
    
    dFr_dQ_ = dFr_dQ(T=T, A=A)
    
    sqrtf = np.sqrt(f)
    num = (2.86*sqrtf + 2.07*f) * h**2 * Fr**2
    den = (0.565 + sqrtf) * rc**2
    
    dnum_dQ = (2.86*sqrtf + 2.07*f) * h**2 * 2*Fr*dFr_dQ_
    dden_dQ = 0.0
    
    return (dnum_dQ*den - num*dden_dQ) / (den**2)

def froude_num(T: float, A: float, Q: float):
    """Computes the Froude number.

    Args:
        T (float): Top width.
        A (float): Flow area.
        Q (float): Flow rate.

    Returns:
        float: The Froude number.
    """
    V = Q/A
    D = A/T
    return V / np.sqrt(g*D)

def dFr_dA(T: float, A: float, Q: float) -> float:
    """Computes the derivative of the Froude number w.r.t. flow area.

    Args:
        T (float): Top width.
        A (float): Flow area.
        Q (float): Flow rate.

    Returns:
        float: dFr/dA.
    """
    V = Q/A
    D = A/T
    
    dV_dA = - Q / A**2
    dD_dA = 1.0 / T
    
    return -0.5 * V * (g*D)**(-1.5) * g*dD_dA + dV_dA * (g*D)**(-0.5)

def dFr_dQ(T: float, A: float):
    """Computes the derivative of the Froude number w.r.t. flow rate.

    Args:
        T (float): Top width.
        A (float): Flow area.
        Q (float): Flow rate.

    Returns:
        float: dFr/dQ.
    """
    D = A/T
    
    dV_dQ = 1.0 / A
    
    return dV_dQ * (g*D)**(-0.5)
    
def dQn_dA(A, S, n, R, dR_dA):
    dQn_dR = (2/3) * A * R**(2/3 - 1) * abs(S)**0.5 / n
    
    dQn_dA = R**(2/3) * abs(S)**0.5 / n + dQn_dR * dR_dA
    if S < 0:
        dQn_dA = -dQn_dA
        
    return dQn_dA

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