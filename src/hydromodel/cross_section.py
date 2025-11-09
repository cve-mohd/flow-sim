from . import hydraulics
import numpy as np
from scipy.optimize import brentq
from abc import ABC, abstractmethod

class CrossSection(ABC):
    """
    Abstract base class for hydraulic cross-sections.
    
    Provides a common interface for different cross-section geometries
    and shared hydraulic calculations (e.g., friction slope, normal flow).
    
    Subclasses must implement geometry-specific methods like
    properties(), conveyance(), and their derivatives.
    """
    def __init__(self, n=None, bed_slope=None, curvature=0.0):
        self.n_left = n
        self.n_main = n
        self.n_right = n
        
        self.left_fp_limit = 0.0
        self.right_fp_limit = 0.0
        
        self._last_hw = None
        self._last_res = None
        self._last_hw_n = None
        self._last_n = None
        
        self.curvature = curvature
        self.bed_slope = bed_slope

    ## ------------------------------------------------------------------
    ## Abstract Methods (Must be implemented by subclasses)
    ## ------------------------------------------------------------------

    @property
    @abstractmethod
    def z_min(self) -> float:
        """Lowest elevation of the cross-section bed."""
        pass

    @property
    @abstractmethod
    def width(self) -> float:
        """Total width of the cross-section."""
        pass

    @abstractmethod
    def properties(self, hw: float) -> tuple:
        """
        Return (A, P, R, T) for a scalar water surface elevation hw.
        (Area, Wetted Perimeter, Hydraulic Radius, Top Width)
        """
        pass

    @abstractmethod
    def get_equivalent_n(self, hw: float) -> float:
        """Compute equivalent Manning's n for the cross-section at hw."""
        pass
    
    @abstractmethod
    def conveyance(self, hw: float) -> float:
        """Compute conveyance (K) for the cross-section at hw."""
        pass
    
    @abstractmethod
    def dK_dA(self, hw: float) -> float:
        """Compute derivative of conveyance w.r.t. area (dK/dA) at hw."""
        pass
    
    @abstractmethod
    def dR_dA(self, hw: float) -> float:
        """Compute derivative of hydraulic radius w.r.t. area (dR/dA) at hw."""
        pass
    
    @abstractmethod
    def dh_dA(self, hw: float) -> float:
        """Compute derivative of depth w.r.t. area (dh/dA) at hw."""
        pass

    @abstractmethod
    def z_at(self, x: float) -> float:
        """Return the bed elevation z at a given lateral coordinate x."""
        pass

    ## ------------------------------------------------------------------
    ## Concrete Methods (Shared functionality)
    ## ------------------------------------------------------------------

    def area(self, hw: float) -> float:
        """Return wetted area (A)."""
        return self.properties(hw)[0]

    def wetted_perimeter(self, hw: float) -> float:
        """Return wetted perimeter (P)."""
        return self.properties(hw)[1]

    def hydraulic_radius(self, hw: float) -> float:
        """Return hydraulic radius (R)."""
        return self.properties(hw)[2]

    def top_width(self, hw: float) -> float:
        """Return top width (T)."""
        return self.properties(hw)[3]

    def get_roughness_para(self) -> tuple:
        """Get composite roughness parameters."""
        return (self.n_left, self.n_main, self.n_right, self.left_fp_limit, self.right_fp_limit)
    
    def set_roughness_para(self, parameters: tuple):
        """Set composite roughness parameters."""
        self.n_left, self.n_main, self.n_right, self.left_fp_limit, self.right_fp_limit = parameters

    def friction_slope(self, h: float, Q: float) -> float:
        """
        Compute friction slope (Sf).
        Assumes a single, contiguous wetted channel.
        Subclasses (like IrregularSection) override this for sub-channel logic.
        """
        hw = h + self.z_min
        K = self.conveyance(hw=hw)
        return hydraulics.Sf(Q=Q, K=K)
    
    def dSf_dA(self, h: float, Q: float) -> float:
        """
        Compute dSf/dA.
        Assumes a single, contiguous wetted channel.
        """
        hw = h + self.z_min
        K = self.conveyance(hw=hw)
        dK_dA = self.dK_dA(hw=hw)
        return hydraulics.dSf_dA(Q=Q, K=K, dK_dA=dK_dA)
    
    def dSf_dQ(self, h: float, Q: float) -> float:
        """
        Compute dSf/dQ.
        Assumes a single, contiguous wetted channel.
        """
        hw = h + self.z_min
        K = self.conveyance(hw=hw)
        return hydraulics.dSf_dQ(Q=Q, K=K)

    def curvature_slope(self, h: float, Q: float) -> float:
        """Compute curvature slope (Sc)."""
        if self.curvature == 0:
            return 0.0
        
        hw = h + self.z_min
        n = self.get_equivalent_n(hw=hw)
        A, P, R, T = self.properties(hw=hw)
        
        return hydraulics.Sc(h=h, T=T, A=A, Q=Q, n=n, R=R, rc=(1.0 / self.curvature))
        
    def dSc_dA(self, h: float, Q: float) -> float:
        """Compute dSc/dA."""
        if self.curvature == 0:
            return 0.0
        
        hw = h + self.z_min
        n = self.get_equivalent_n(hw=hw)
        A, P, R, T = self.properties(hw=hw)
        dR_dA = self.dR_dA(hw=hw)
        
        return hydraulics.dSc_dA(h=h, A=A, Q=Q, n=n, R=R, rc=(1.0 / self.curvature), dR_dA=dR_dA, T=T) / self.dh_dA(hw=hw)
        
    def dSc_dQ(self, h: float, Q: float) -> float:
        """Compute dSc/dQ."""
        if self.curvature == 0:
            return 0.0
        
        hw = h + self.z_min
        n = self.get_equivalent_n(hw=hw)
        A, P, R, T = self.properties(hw=hw)
        
        return hydraulics.dSc_dQ(h=h, T=T, A=A, Q=Q, n=n, R=R, rc=(1.0 / self.curvature))
    
    def normal_flow(self, hw: float) -> float:
        """Computes the normal flow rate for a given water level."""
        if self.bed_slope is None or self.bed_slope <= 0.0:
            return 0.0
        K = self.conveyance(hw=hw)        
        return hydraulics.normal_flow(bed_slope=self.bed_slope, K=K)
    
    def normal_depth(self, Q_target: float, hw_max = None) -> float:
        """Computes the normal flow depth for a given flow rate."""
        _z_min = self.z_min
        if hw_max is None:
            hw_max = _z_min + 100 # Default 100m max depth
            
        def f(hw):
            return Q_target - self.normal_flow(hw=hw)
        
        try:
            hw = brentq(f, _z_min, hw_max)
            return hw - _z_min
        except ValueError:
            # Handle cases where Q_target is out of bounds
            if f(_z_min) < 0: # Q_target is > 0
                return 0.0
            if f(hw_max) > 0: # Q_target is larger than capacity at hw_max
                return hw_max - _z_min
            return 0.0

## 2. 'IrregularSection' Class
## ------------------------------------------------------------------

class IrregularSection(CrossSection):
    """
    Cross-section defined by a polyline (x, z coordinates).
    
    Handles complex geometries including multiple, disconnected
    sub-channels (e.g., a main channel and flooded overbanks
    separated by a dry levee).
    """
    
    def __init__(self, x, z, n=None, **kwargs):
        super().__init__(n=n, **kwargs)

        x = np.ascontiguousarray(x, dtype=float)
        z = np.ascontiguousarray(z, dtype=float)
        if x.shape != z.shape:
            raise ValueError("x and z must have the same shape")
        if x.ndim != 1:
            raise ValueError("x and z must be 1-D arrays")

        # sort by x to guarantee increasing x
        idx = np.argsort(x)
        self.x = x[idx]
        self.z = z[idx]

        self._n_pts = self.x.size
        self._z_min = float(np.min(self.z))
        self._width = float(np.max(self.x) - np.min(self.x))

        # Set roughness limits to extents of section
        self.left_fp_limit = self.x[0]
        self.right_fp_limit = self.x[-1]

    @property
    def z_min(self) -> float:
        return self._z_min
    
    @property
    def width(self) -> float:
        return self._width

    def properties(self, hw: float) -> tuple:
        """Return (A, P, R, T) for scalar hw."""
        hw = float(hw)
        if self._last_hw is not None and hw == self._last_hw:
            return self._last_res

        z = self.z
        x = self.x
        
        # Check if water level is below bed
        if hw <= self._z_min:
            res = (0.0, 0.0, 0.0, 0.0)
            self._last_hw, self._last_res = hw, res
            return res
            
        h_nodes = hw - z
        below = h_nodes > 0.0
        if not np.any(below):
            res = (0.0, 0.0, 0.0, 0.0)
            self._last_hw, self._last_res = hw, res
            return res

        # Identify continuous wetted segments (sub-channels)
        wet_segments = []
        i = 0
        n = len(below)
        while i < n:
            if below[i]:
                start = i
                while i + 1 < n and below[i + 1]:
                    i += 1
                end = i
                wet_segments.append((start, end))
            i += 1

        A_total = 0.0
        P_total = 0.0
        T_total = 0.0

        for (i0, iN) in wet_segments:
            x_seg = x[i0:iN + 1]
            z_seg = z[i0:iN + 1]

            # Add intersection points at hw on both sides
            if i0 > 0 and z[i0 - 1] > hw:
                z0, z1 = z[i0 - 1], z[i0]
                x0, x1 = x[i0 - 1], x[i0]
                t = (hw - z0) / (z1 - z0) # Linear interpolation factor
                xl = x0 + t * (x1 - x0)
                x_seg = np.insert(x_seg, 0, xl)
                z_seg = np.insert(z_seg, 0, hw)

            if iN < n - 1 and z[iN + 1] > hw:
                z0, z1 = z[iN], z[iN + 1]
                x0, x1 = x[iN], x[iN + 1]
                t = (hw - z0) / (z1 - z0) # Linear interpolation factor
                xr = x0 + t * (x1 - x0)
                x_seg = np.append(x_seg, xr)
                z_seg = np.append(z_seg, hw)
            
            # Water depth at each node in the segment
            d_seg = np.maximum(hw - z_seg, 0.0)

            # Area (using trapezoidal rule on depths)
            A = np.sum(0.5 * (d_seg[:-1] + d_seg[1:]) * np.diff(x_seg))

            # Perimeter (sum of segment lengths)
            dz = np.diff(z_seg)
            dx = np.diff(x_seg)
            seg_len = np.sqrt(dx**2 + dz**2)
            P = np.sum(seg_len)

            # Top width
            T = x_seg[-1] - x_seg[0]

            A_total += A
            P_total += P
            T_total += T

        R_total = A_total / P_total if P_total > 0.0 else 0.0
        res = (float(A_total), float(P_total), float(R_total), float(T_total))
        self._last_hw, self._last_res = hw, res
        return res

    def get_subchannels(self, hw: float) -> list:
        """
        Identify contiguous wetted regions (subchannels) in the cross-section.
        """
        wet = self.z < hw
        subchannels = []
        i = 0
        n = len(wet)

        while i < n:
            # skip dry zones
            if not wet[i]:
                i += 1
                continue

            # start of a wetted segment
            start = i
            while i < n and wet[i]:
                i += 1
            end = i  # one past last wet index
            
            if (end - start) < 2: # Need at least 2 points
                continue

            # include boundary points for accurate integration
            x_seg = self.x[start:end]
            z_seg = self.z[start:end]
            
            # Add intersection points with water surface at edges
            if start > 0 and self.z[start-1] > hw:
                x0 = np.interp(hw, [self.z[start-1], self.z[start]], [self.x[start-1], self.x[start]])
                x_seg = np.insert(x_seg, 0, x0)
                z_seg = np.insert(z_seg, 0, hw)
            if end < n and self.z[end-1] < hw and self.z[end] > hw:
                x1 = np.interp(hw, [self.z[end-1], self.z[end]], [self.x[end-1], self.x[end]])
                x_seg = np.append(x_seg, x1)
                z_seg = np.append(z_seg, hw)

            subchannels.append({"x": x_seg, "z": z_seg})

        return subchannels

    def friction_slope(self, h: float, Q: float) -> float:
        """Compute friction slope (Sf), handling multiple sub-channels."""
        hw = h + self.z_min
        subchs = self.get_subchannels(hw)
        _n = len(subchs)
            
        if _n <= 1:
            # Fallback to single-channel logic
            return super().friction_slope(h, Q)

        # Multiple subchannels: combine conveyance
        K_sum = 0.0
        for sc in subchs:
            # Create temporary, simple cross-section for this sub-channel
            xs = IrregularSection(x=sc["x"], z=sc["z"])
            xs.set_roughness_para(parameters=self.get_roughness_para())
                        
            K_j = xs.conveyance(hw)
            K_sum += K_j ** 1.5
        
        K_total = K_sum ** (2.0 / 3.0)
        return hydraulics.Sf(K=K_total, Q=Q)
    
    def dSf_dA(self, h: float, Q: float) -> float:
        """Compute dSf/dA, handling multiple sub-channels."""
        hw = h + self.z_min
        subchs = self.get_subchannels(hw)
        _n = len(subchs)
            
        if _n <= 1:
            # Fallback to single-channel logic
            return super().dSf_dA(h, Q)

        K_sum = 0.0
        dK_dA_sum = 0.0
        for sc in subchs:
            xs = IrregularSection(x=sc["x"], z=sc["z"])
            xs.set_roughness_para(parameters=self.get_roughness_para())
                        
            K_j = xs.conveyance(hw=hw)
            K_sum += K_j ** 1.5
            dK_dA_j = xs.dK_dA(hw=hw)
            dK_dA_sum += 1.5 * (K_j ** 0.5) * dK_dA_j

        K_eq = K_sum ** (2.0 / 3.0)
        dK_dA_eq = (2.0 / 3.0) * K_sum ** (-1.0 / 3.0) * dK_dA_sum
        
        return hydraulics.dSf_dA(Q=Q, K=K_eq, dK_dA=dK_dA_eq)
    
    def dSf_dQ(self, h: float, Q: float) -> float:
        """Compute dSf/dQ, handling multiple sub-channels."""
        hw = h + self.z_min
        subchs = self.get_subchannels(hw)
        _n = len(subchs)
            
        if _n <= 1:
            # Fallback to single-channel logic
            return super().dSf_dQ(h, Q)

        K_sum = 0.0
        for sc in subchs:
            xs = IrregularSection(x=sc["x"], z=sc["z"])
            xs.set_roughness_para(parameters=self.get_roughness_para())
            K_j = xs.conveyance(hw=hw)
            
            K_sum += K_j ** 1.5

        K_eq = K_sum ** (2.0 / 3.0)
        return hydraulics.dSf_dQ(Q=Q, K=K_eq)

    def get_equivalent_n(self, hw: float) -> float:
        """
        Compute equivalent Manning's n using composite roughness.
        """
        if self._last_hw_n is not None and hw == self._last_hw_n:
            return self._last_n
        
        def subsection_props(x_min, x_max, n_value):
            # Find points within the subsection
            mask = (self.x >= x_min) & (self.x <= x_max)
            if mask.sum() < 2:
                return 0.0, 0.0, 0.0
                
            xs_nodes, zs_nodes = self.x[mask], self.z[mask]
            
            # Create a temporary section for this subsection
            sub_xs = IrregularSection(x=xs_nodes, z=zs_nodes)
                
            A = sub_xs.area(hw)
            if A <= 0:
                return 0.0, 0.0, 0.0
            
            # Note: This P is for the wetted bed, not including vertical walls
            # which is the correct approach for composite roughness.
            P = sub_xs.wetted_perimeter(hw)
            if P <= 0:
                return 0.0, 0.0, 0.0
                
            R = A / P
            K = hydraulics.conveyance(A=A, n=n_value, R=R)
            return A, R, K

        # Subsections
        left_A, left_R, left_K = subsection_props(self.x[0], self.left_fp_limit, self.n_left)
        main_A, main_R, main_K = subsection_props(self.left_fp_limit, self.right_fp_limit, self.n_main)
        right_A, right_R, right_K = subsection_props(self.right_fp_limit, self.x[-1], self.n_right)

        # Total area and hydraulic radius
        A_total = self.area(hw) # Use full section properties
        P_total = self.wetted_perimeter(hw)
        
        if A_total <= 0 or P_total <= 0:
            return self.n_main  # fallback

        R_total = A_total / P_total

        # Combine conveyances (Horton-Einstein method)
        K_total = (left_K**1.5 + main_K**1.5 + right_K**1.5) ** (2.0 / 3.0)
        
        if K_total <= 0.0:
            return self.n_main # fallback

        # Equivalent Manning's n
        n_eq = (A_total * (R_total ** (2.0 / 3.0))) / K_total
        
        self._last_hw_n = hw
        self._last_n = n_eq
        
        return n_eq
    
    def conveyance(self, hw: float) -> float:
        """Compute conveyance K."""
        A = self.area(hw=hw)
        if A <= 0.0: return 0.0
        
        n = self.get_equivalent_n(hw=hw)
        R = self.hydraulic_radius(hw=hw)
        
        return hydraulics.conveyance(A=A, n=n, R=R)
    
    def dK_dA(self, hw: float) -> float:
        """Compute dK/dA."""
        A = self.area(hw=hw)
        if A <= 0.0: return 0.0
        
        n = self.get_equivalent_n(hw=hw)
        R = self.hydraulic_radius(hw=hw)
        dR_dA = self.dR_dA(hw=hw)
        
        return hydraulics.dK_dA_(A=A, n=n, R=R, dR_dA=dR_dA)

    def dR_dA(self, hw: float, dh: float = 1e-6) -> float:
        """Compute dR/dA using finite difference."""
        A1 = self.area(hw - dh)
        A2 = self.area(hw + dh)
        if (A2 - A1) == 0.0: return 0.0
        
        R1 = self.hydraulic_radius(hw - dh)
        R2 = self.hydraulic_radius(hw + dh)
        return (R2 - R1) / (A2 - A1)
    
    def dh_dA(self, hw: float, dh: float = 1e-6) -> float:
        """Compute dh/dA using finite difference."""
        A1 = self.area(hw - dh)
        A2 = self.area(hw + dh)
        if (A2 - A1) == 0.0: return 0.0
                
        return 2 * dh / (A2 - A1)

    def z_at(self, x: float) -> float:
        """Interpolate bed elevation at lateral coordinate x."""
        return np.interp(x, self.x, self.z, left=self.z[0], right=self.z[-1])

## ------------------------------------------------------------------
## 3. 'TrapezoidalSection' Class
## ------------------------------------------------------------------

class TrapezoidalSection(CrossSection):
    """
    Cross-section with a trapezoidal geometry.
    
    Handles:
    1. Compound Trapezoid (main channel + trapezoidal floodplains)
    2. Simple Trapezoid
    3. Rectangle (simple trapezoid with m=0)
    
    Parameters:
    - z_bed: Main channel bed elevation [m]
    - b_main: Main channel bottom width [m]
    - m_main: Main channel side slope [m/m]
    - z_bank: Elevation of floodplain [m]. If None, creates a simple trap.
    - b_fp_left: Width of left floodplain bed [m]
    - b_fp_right: Width of right floodplain bed [m]
    - m_fp: Side slope of the outer floodplain walls [m/m]
    - n_main, n_left, n_right: Manning's n values
    """
    
    def __init__(self, z_bed, b_main, m_main, 
                 z_bank=None, b_fp_left=0.0, b_fp_right=0.0, m_fp=0.0,
                 n_main=0.03, n_left=0.03, n_right=0.03, **kwargs):
        
        super().__init__(n=n_main, **kwargs)
        
        self.z_bed = float(z_bed)
        self.b_main = float(b_main)
        self.m_main = float(m_main)
        
        self._z_min = self.z_bed
        
        if z_bank is not None:
            self._is_compound = True
            self.z_bank = float(z_bank)
            self.b_fp_left = float(b_fp_left)
            self.b_fp_right = float(b_fp_right)
            self.m_fp = float(m_fp)
            
            if self.z_bank <= self.z_bed:
                raise ValueError("Bank elevation z_bank must be above bed z_bed")
            
            self.bankfull_depth = self.z_bank - self.z_bed
            self.T_main_at_bank = self.b_main + 2.0 * self.m_main * self.bankfull_depth
            
            self.left_fp_limit = -self.T_main_at_bank / 2.0
            self.right_fp_limit = self.T_main_at_bank / 2.0
            
            self._width_at_bank = self.b_fp_left + self.T_main_at_bank + self.b_fp_right
            self._width = np.inf # Unbounded
            
        else:
            self._is_compound = False
            self.z_bank = None
            self.b_fp_left = 0.0
            self.b_fp_right = 0.0
            self.m_fp = 0.0
            self._width = np.inf # Unbounded
            
            self.left_fp_limit = -np.inf
            self.right_fp_limit = np.inf
            
        self._is_rect = not self._is_compound and self.m_main == 0.0
        
        self.set_roughness_para((n_left, n_main, n_right, self.left_fp_limit, self.right_fp_limit))

    @property
    def z_min(self) -> float:
        return self._z_min
    
    @property
    def width(self) -> float:
        return self._width

    def properties(self, hw: float) -> tuple:
        """Return (A, P, R, T) using analytical formulas."""
        hw = float(hw)
        if self._last_hw is not None and hw == self.last_hw:
            return self._last_res

        depth = max(0.0, hw - self.z_bed)
        if depth <= 0.0:
            res = (0.0, 0.0, 0.0, 0.0)
            self._last_hw, self._last_res = hw, res
            return res

        # Case 1: Rectangle
        if self._is_rect:
            A = self.b_main * depth
            P = self.b_main + 2.0 * depth
            T = self.b_main
        
        # Case 2: Simple Trapezoid
        elif not self._is_compound:
            T = self.b_main + 2.0 * self.m_main * depth
            A = (self.b_main + T) / 2.0 * depth
            P = self.b_main + 2.0 * depth * np.sqrt(1.0 + self.m_main**2)
            
        # Case 3: Compound Trapezoid
        else:
            if depth <= self.bankfull_depth:
                # Water is only in the main channel
                T = self.b_main + 2.0 * self.m_main * depth
                A = (self.b_main + T) / 2.0 * depth
                P = self.b_main + 2.0 * depth * np.sqrt(1.0 + self.m_main**2)
            
            else:
                # Water is in main channel AND floodplains
                depth_fp = depth - self.bankfull_depth
                
                # Main channel (full)
                A_main = (self.b_main + self.T_main_at_bank) / 2.0 * self.bankfull_depth
                P_main = self.b_main + 2.0 * self.bankfull_depth * np.sqrt(1.0 + self.m_main**2)
                
                # Left FP (trapezoidal)
                A_left = (self.b_fp_left + 0.5 * self.m_fp * depth_fp) * depth_fp
                P_left = self.b_fp_left + depth_fp * np.sqrt(1.0 + self.m_fp**2)
                
                # Right FP (trapezoidal)
                A_right = (self.b_fp_right + 0.5 * self.m_fp * depth_fp) * depth_fp
                P_right = self.b_fp_right + depth_fp * np.sqrt(1.0 + self.m_fp**2)

                # Totals
                A = A_main + A_left + A_right
                P = P_main + P_left + P_right
                T = self._width_at_bank + 2.0 * self.m_fp * depth_fp
        
        R = A / P if P > 0.0 else 0.0
        res = (A, P, R, T)
        self._last_hw, self._last_res = hw, res
        return res

    def _get_subsection_props(self, hw: float) -> tuple:
        """Helper to get (A, P, R) for left, main, right."""
        depth = max(0.0, hw - self.z_bed)
        if depth <= 0.0:
            return (0,0,0), (0,0,0), (0,0,0)
            
        if not self._is_compound or depth <= self.bankfull_depth:
            A, P, R, T = self.properties(hw)
            return (0,0,0), (A, P, R), (0,0,0)
        
        depth_fp = depth - self.bankfull_depth
        
        # Main channel (full)
        A_main = (self.b_main + self.T_main_at_bank) / 2.0 * self.bankfull_depth + self.T_main_at_bank * depth_fp
        P_main_bed = self.b_main + 2.0 * self.bankfull_depth * np.sqrt(1.0 + self.m_main**2)
        R_main = A_main / P_main_bed if P_main_bed > 0 else 0.0
        
        # Left FP
        A_left = (self.b_fp_left + 0.5 * self.m_fp * depth_fp) * depth_fp
        P_left_bed = self.b_fp_left + depth_fp * np.sqrt(1.0 + self.m_fp**2)
        R_left = A_left / P_left_bed if P_left_bed > 0 else 0.0
        
        # Right FP
        A_right = (self.b_fp_right + 0.5 * self.m_fp * depth_fp) * depth_fp
        P_right_bed = self.b_fp_right + depth_fp * np.sqrt(1.0 + self.m_fp**2)
        R_right = A_right / P_right_bed if P_right_bed > 0 else 0.0
        
        return (A_left, P_left_bed, R_left), (A_main, P_main_bed, R_main), (A_right, P_right_bed, R_right)

    def get_equivalent_n(self, hw: float) -> float:
        """Compute equivalent n, analytic version."""
        if self._last_hw_n is not None and hw == self._last_hw_n:
            return self._last_n
        
        if not self._is_compound:
            self._last_hw_n, self._last_n = hw, self.n_main
            return self.n_main
            
        (A_l, P_l, R_l), (A_m, P_m, R_m), (A_r, P_r, R_r) = self._get_subsection_props(hw)
        
        K_left = hydraulics.conveyance(A_l, self.n_left, R_l)
        K_main = hydraulics.conveyance(A_m, self.n_main, R_m)
        K_right = hydraulics.conveyance(A_r, self.n_right, R_r)
        
        A_total, P_total, R_total, T_total = self.properties(hw)
        
        if A_total <= 0 or R_total <= 0:
            return self.n_main

        K_total = (K_left**1.5 + K_main**1.5 + K_right**1.5) ** (2.0 / 3.0)
        
        if K_total <= 0.0:
            return self.n_main

        n_eq = (A_total * (R_total ** (2.0 / 3.0))) / K_total
        
        self._last_hw_n = hw
        self._last_n = n_eq
        return n_eq
        
    def conveyance(self, hw: float) -> float:
        """Compute conveyance K, analytic version."""
        if not self._is_compound:
            A, P, R, T = self.properties(hw)
            return hydraulics.conveyance(A, self.n_main, R)
            
        (A_l, P_l, R_l), (A_m, P_m, R_m), (A_r, P_r, R_r) = self._get_subsection_props(hw)
        
        K_left = hydraulics.conveyance(A_l, self.n_left, R_l)
        K_main = hydraulics.conveyance(A_m, self.n_main, R_m)
        K_right = hydraulics.conveyance(A_r, self.n_right, R_r)
        
        K_total = (K_left**1.5 + K_main**1.5 + K_right**1.5) ** (2.0 / 3.0)
        return K_total

    def dK_dA(self, hw: float) -> float:
        """Compute dK/dA, analytic version."""
        A, P, R, T = self.properties(hw)
        if A <= 0.0: return 0.0
        
        n = self.get_equivalent_n(hw)
        dR_dA = self.dR_dA(hw)
        
        return hydraulics.dK_dA_(A=A, n=n, R=R, dR_dA=dR_dA)

    def dR_dA(self, hw: float) -> float:
        """Compute dR/dA analytically."""
        A, P, R, T = self.properties(hw)
        if P <= 0.0 or T <= 0.0:
            return 0.0
            
        depth = max(0.0, hw - self.z_bed)
        
        if self._is_rect:
            dP_dh = 2.0
        elif not self._is_compound:
            dP_dh = 2.0 * np.sqrt(1.0 + self.m_main**2)
        else:
            # Compound
            if depth <= self.bankfull_depth:
                dP_dh = 2.0 * np.sqrt(1.0 + self.m_main**2)
            else:
                # P = C + 2 * (P_fp_bed) = C + 2 * (b_fp + depth_fp * sqrt(1+m_fp^2))
                # dP/dh = dP/d(depth_fp) = 2 * sqrt(1 + m_fp^2)
                dP_dh = 2.0 * np.sqrt(1.0 + self.m_fp**2)
        
        dh_dA = 1.0 / T
        dP_dA = dP_dh * dh_dA
        
        return (P - A * dP_dA) / (P**2)

    def dh_dA(self, hw: float) -> float:
        """Compute dh/dA analytically (1/T)."""
        T = self.top_width(hw)
        return 1.0 / T if T > 0.0 else 0.0

    def z_at(self, x: float) -> float:
        """Return bed elevation z at lateral coordinate x."""
        x = float(x)
        
        if self._is_rect:
            if x > -self.b_main / 2.0 and x < self.b_main / 2.0:
                return self.z_bed
            else:
                return np.inf
        
        if not self._is_compound:
            # Simple trapezoid
            if x >= -self.b_main / 2.0 and x <= self.b_main / 2.0:
                return self.z_bed
            elif x > self.b_main / 2.0:
                dist = x - self.b_main / 2.0
                return self.z_bed + dist / self.m_main
            else: # x < -self.b_main / 2.0
                dist = -x - self.b_main / 2.0
                return self.z_bed + dist / self.m_main
        
        # Compound
        if x >= self.left_fp_limit and x <= self.right_fp_limit:
            # In main channel
            if x >= -self.b_main / 2.0 and x <= self.b_main / 2.0:
                return self.z_bed
            elif x > self.b_main / 2.0: # right bank
                dist = x - self.b_main / 2.0
                return self.z_bed + dist / self.m_main
            else: # left bank
                dist = -x - self.b_main / 2.0
                return self.z_bed + dist / self.m_main
                
        elif x < self.left_fp_limit:
            # Left floodplain
            x_left_bed_outer = self.left_fp_limit - self.b_fp_left
            if x >= x_left_bed_outer:
                return self.z_bank # On floodplain bed
            else:
                # On outer inclined wall
                dist = x_left_bed_outer - x
                return self.z_bank + dist / self.m_fp
                
        else: # x > self.right_fp_limit
            # Right floodplain
            x_right_bed_outer = self.right_fp_limit + self.b_fp_right
            if x <= x_right_bed_outer:
                return self.z_bank # On floodplain bed
            else:
                # On outer inclined wall
                dist = x - x_right_bed_outer
                return self.z_bank + dist / self.m_fp