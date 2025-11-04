import numpy as np
from . import hydraulics
from scipy.optimize import brentq

class CrossSection:
    """
    Efficient cross-section utilities.

    Constructor options (mutually exclusive):
      1) Provide x and z as 1D numpy arrays of equal length. x need not be strictly
         increasing; it will be sorted internally.
      2) Provide width and bed to create a rectangular cross-section with vertical
         walls.

    Methods (all accept a scalar water surface elevation hw):
      - area(hw) -> float
      - wetted_perimeter(hw) -> float
      - top_width(hw) -> float
      - properties(hw) -> (A, P, R, T)  # single fast call returning all four

    The implementation is vectorized and precomputes segment geometry for speed.
    """

    def __init__(self, x=None, z=None, width=None, bed=None, n=None):
        # Rectangular shortcut
        if width is not None and bed is not None:
            self._is_rect = True
            self.width = float(width)
            self.z_min = float(bed)

            # no arrays required
            self._n = 2

        else:
            # General polyline cross-section
            if x is None or z is None:
                raise ValueError("Either (x,z) or (width,bed) must be provided")

            x = np.ascontiguousarray(x, dtype=float)
            z = np.ascontiguousarray(z, dtype=float)
            if x.shape != z.shape:
                raise ValueError("x and z must have the same shape")
            if x.ndim != 1:
                raise ValueError("x and z must be 1-D arrays")

            # sort by x to guarantee increasing x
            idx = np.argsort(x)
            x = x[idx]
            z = z[idx]

            self._is_rect = False
            self.x = x
            self.z = z
            self._n = x.size
            self.z_min = float(np.min(self.z))
            self.width = float(np.max(self.x) - np.min(self.x))

            # Precompute segment geometry
            self._dx = np.diff(self.x)                # dx between consecutive points
            self._dz = np.diff(self.z)                # dz between consecutive points
            self._seg_len = np.hypot(self._dx, self._dz)
            self._x0 = self.x[:-1]
            self._x1 = self.x[1:]
            self._z0 = self.z[:-1]
            self._z1 = self.z[1:]
            self._denom = self._z1 - self._z0        # used for intersection param t
        
        # Composite n        
        self.n_left = n   # Manning's n for left floodplain
        self.n_main = n   # Manning's n for main channel
        self.n_right = n  # Manning's n for right floodplain
        self.left_fp_limit = 0 if self._is_rect else self.x[0]   # x-coordinate dividing left floodplain and main channel
        self.right_fp_limit = self.width if self._is_rect else self.x[-1]  # x-coordinate dividing main channel and right floodplain
        
        # cache
        self._last_hw = None
        self._last_res = None
        
        self._last_hw_n = None
        self._last_n = None
        
        # other
        self.curvature = 0.0
        self.bed_slope = None

    def properties(self, hw):
        """Return (A, P, R, T) for scalar hw."""
        hw = float(hw)
        if self._last_hw is not None and hw == self._last_hw:
            return self._last_res

        if self._is_rect:
            depth = max(0.0, hw - self.z_min)
            A = self.width * depth
            if depth <= 0.0:
                res = (0.0, 0.0, 0.0, 0.0)
            else:
                P = self.width + 2.0 * depth
                T = self.width
                R = A / P
                res = (A, P, R, T)
                
            self._last_hw, self._last_res = hw, res
            return res

        z = self.z
        x = self.x
        h = hw - z
        below = h > 0.0
        if not np.any(below):
            res = (0.0, 0.0, 0.0, 0.0)
            self._last_hw, self._last_res = hw, res
            return res

        # Identify continuous wetted segments
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
                t = (hw - z0) / (z1 - z0)
                xl = x0 + t * (x1 - x0)
                x_seg = np.insert(x_seg, 0, xl)
                z_seg = np.insert(z_seg, 0, hw)

            if iN < n - 1 and z[iN + 1] > hw:
                z0, z1 = z[iN], z[iN + 1]
                x0, x1 = x[iN], x[iN + 1]
                t = (hw - z0) / (z1 - z0)
                xr = x0 + t * (x1 - x0)
                x_seg = np.append(x_seg, xr)
                z_seg = np.append(z_seg, hw)

            d_seg = np.maximum(hw - z_seg, 0.0)

            # Area (trapezoidal)
            A = np.sum(0.5 * (d_seg[:-1] + d_seg[1:]) * np.diff(x_seg))

            # Perimeter
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

    def area(self, hw):
        return self.properties(hw)[0]

    def wetted_perimeter(self, hw):
        return self.properties(hw)[1]

    def top_width(self, hw):
        return self.properties(hw)[3]
    
    def hydraulic_radius(self, hw):
        return self.properties(hw)[2]
    
    def friction_slope(self, h, Q):
        hw = h + self.z_min
        
        if not self._is_rect:
            subchs = self.get_subchannels(hw)
            _n = len(subchs)
        else:
            _n = 1
            
        if _n == 1:
            K = self.conveyance(hw=hw)
            return hydraulics.Sf(Q=Q, K=K)

        # multiple subchannels
        K_sum = 0.0
        for sc in subchs:
            xs = CrossSection(x=sc["x"], z=sc["z"])
            xs.set_roughness_para(parameters=self.get_roughness_para())
                        
            K_j = xs.conveyance(hw)
            K_sum += K_j ** 1.5
        
        K_total = K_sum ** (2 / 3)
        return hydraulics.Sf(K=K_total, Q=Q)
    
    def dSf_dA(self, h, Q):
        hw = h + self.z_min
        
        if not self._is_rect:
            subchs = self.get_subchannels(hw)
            _n = len(subchs)
        else:
            _n = 1
            
        if _n == 1:
            K = self.conveyance(hw=hw)
            dK_dA = self.dK_dA(hw=hw)
            return hydraulics.dSf_dA(Q=Q, K=K, dK_dA=dK_dA)

        K_sum = 0.0
        dK_dA_sum = 0.0
        for sc in subchs:
            xs = CrossSection(x=sc["x"], z=sc["z"])
            xs.set_roughness_para(parameters=self.get_roughness_para())
                        
            K_j = xs.conveyance(hw=hw)
            K_sum += K_j ** 1.5
            dK_dA_j = xs.dK_dA(hw=hw)
            dK_dA_sum += 1.5 * (K_j ** 0.5) * dK_dA_j

        K_eq = K_sum ** (2 / 3)
        dK_dA_eq = (2 / 3) * K_sum ** (-1 / 3) * dK_dA_sum
        
        return hydraulics.dSf_dA(Q=Q, K=K_eq, dK_dA=dK_dA_eq)
    
    def dSf_dQ(self, h: float, Q: float) -> float:
        hw = h + self.z_min
        
        if not self._is_rect:
            subchs = self.get_subchannels(hw)
            _n = len(subchs)
        else:
            _n = 1
            
        if _n == 1:
            K = self.conveyance(hw=hw)
            return hydraulics.dSf_dQ(Q=Q, K=K)

        K_sum = 0.0
        for sc in subchs:
            xs = CrossSection(x=sc["x"], z=sc["z"])
            xs.set_roughness_para(parameters=self.get_roughness_para())
            K_j = xs.conveyance(hw=hw)
            
            K_sum += K_j ** 1.5

        K_eq = K_sum ** (2 / 3)
        return hydraulics.dSf_dQ(Q=Q, K=K_eq)
    
    def curvature_slope(self, h: float, Q: float) -> float:
        if self.curvature == 0:
            return 0.0
        
        hw = h + self.z_min
        
        n = self.get_equivalent_n(hw=hw)
        R = self.hydraulic_radius(hw=hw)
        T = self.top_width(hw=hw)
        A = self.area(hw=hw)
        
        return hydraulics.Sc(h=h, T=T, A=A, Q=Q, n=n, R=R, rc=(1.0 / self.curvature))
        
    def get_equivalent_n(self, hw):
        """
        Compute equivalent Manning's n for node i at flow depth h.
        Uses separate roughness values for left floodplain, main channel, and right floodplain.
        """
        if self._last_hw_n is not None and hw == self._last_hw_n:
            return self._last_n
        
        if self._is_rect:
            n_eq = self.n_main
            self._last_hw_n = hw
            self._last_n = n_eq
            return n_eq
        
        def subsection_props(x_min, x_max, n_value):
            mask = (self.x >= x_min) & (self.x <= x_max)
            if mask.sum() < 2:
                return 0.0, 0.0, 0.0
            xs, zs = self.x[mask], self.z[mask]
            sub_xs = CrossSection(x=xs, z=zs)
                
            A = sub_xs.area(hw)
            if A <= 0:
                return 0.0, 0.0, 0.0
            R = sub_xs.hydraulic_radius(hw)
            K = hydraulics.conveyance(A=A, n=n_value, R=R)
            return A, R, K

        # Subsections
        left_A, left_R, left_K = subsection_props(self.x[0], self.left_fp_limit, self.n_left)
        main_A, main_R, main_K = subsection_props(self.left_fp_limit, self.right_fp_limit, self.n_main)
        right_A, right_R, right_K = subsection_props(self.right_fp_limit, self.x[-1], self.n_right)

        # Total area and hydraulic radius
        A_total = left_A + main_A + right_A
        P_total = self.wetted_perimeter(hw)
        if A_total <= 0 or P_total <= 0:
            return self.n_main  # fallback

        R_total = A_total / P_total

        # Combine conveyances
        K_total = (left_K**1.5 + main_K**1.5 + right_K**1.5) ** (2.0 / 3.0)

        # Equivalent Manning's n
        n_eq = (A_total * (R_total ** (2.0 / 3.0))) / K_total
        
        self._last_hw_n = hw
        self._last_n = n_eq
        
        return n_eq

    def dSc_dA(self, h: float, Q: float) -> float:
        if self.curvature == 0:
            return 0.0
        
        hw = h + self.z_min
        
        n = self.get_equivalent_n(hw=hw)
        R = self.hydraulic_radius(hw=hw)
        T = self.top_width(hw=hw)
        A = self.area(hw=hw)
        dR_dA = self.dR_dA(hw=hw)
        
        return hydraulics.dSc_dA(h=h, A=A, Q=Q, n=n, R=R, rc=(1.0 / self.curvature), dR_dA=dR_dA, T=T) / self.dh_dA
        
    def dSc_dQ(self, h: float, Q: float) -> float:
        if self.curvature == 0:
            return 0.0
        
        hw = h + self.z_min
        
        n = self.get_equivalent_n(hw=hw)
        R = self.hydraulic_radius(hw=hw)
        T = self.top_width(hw=hw)
        A = self.area(hw=hw)
        
        return hydraulics.dSc_dQ(h=h, T=T, A=A, Q=Q, n=n, R=R, rc=(1.0 / self.curvature))

    def dR_dA(self, hw, dh=1e-6):
        if self._is_rect:
            A = self.area(hw=hw)
            P = self.wetted_perimeter(hw=hw)
            
            dR_dA = 1.0/P
            dR_dP = -A / P**2
            dP_dA = 2 * self.dh_dA(hw=hw)
            
            return dR_dA + dR_dP * dP_dA
            
        A1 = self.area(hw - dh)
        A2 = self.area(hw + dh)
        R1 = self.hydraulic_radius(hw - dh)
        R2 = self.hydraulic_radius(hw + dh)
        return (R2 - R1) / (A2 - A1)
    
    def dh_dA(self, hw, dh=1e-6):
        A1 = self.area(hw - dh)
        A2 = self.area(hw + dh)
                
        return 2 * dh / (A2 - A1)
    
    def normal_flow(self, hw: float) -> float:
        """Computes the normal flow rate for a given water level.

        Args:
            A (float): Water level.

        Returns:
            float: Normal flow rate.
        """
        K = self.conveyance(hw=hw)        
        return hydraulics.normal_flow(bed_slope=self.bed_slope, K=K)
    
    def normal_depth(self, Q_target: float, hw_max = None) -> float:
        """Computes the normal flow area for a given flow rate.

        Args:
            Q (float): Flow rate.

        Returns:
            float: Normal flow area.
        """
        if hw_max is None:
            hw_max = self.z_min + 100
            
        def f(hw):
            return Q_target - self.normal_flow(hw=hw)
        
        hw = brentq(f, self.z_min, hw_max)
        return hw - self.z_min

    def get_subchannels(self, hw):
        """
        Identify contiguous wetted regions (subchannels) in the cross-section.
        Each subchannel is defined by consecutive x,z points where z < h.
        Returns a list of dicts, each containing x and z arrays for the segment.
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

            # include boundary points for accurate integration
            x_seg = self.x[start:end]
            z_seg = self.z[start:end]
            # Add intersection points with water surface at edges
            if start > 0 and z_seg[0] > hw:
                x0 = np.interp(hw, [self.z[start-1], self.z[start]], [self.x[start-1], self.x[start]])
                x_seg = np.insert(x_seg, 0, x0)
                z_seg = np.insert(z_seg, 0, hw)
            if end < n and self.z[end-1] < hw and self.z[end] > hw:
                x1 = np.interp(hw, [self.z[end-1], self.z[end]], [self.x[end-1], self.x[end]])
                x_seg = np.append(x_seg, x1)
                z_seg = np.append(z_seg, hw)

            subchannels.append({"x": x_seg, "z": z_seg})

        return subchannels

    def conveyance(self, hw):
        A = self.area(hw=hw)
        n = self.get_equivalent_n(hw=hw)
        R = self.hydraulic_radius(hw=hw)
        
        return hydraulics.conveyance(A=A, n=n, R=R)
    
    def dK_dA(self, hw):
        A = self.area(hw=hw)
        n = self.get_equivalent_n(hw=hw)
        R = self.hydraulic_radius(hw=hw)
        dR_dA = self.dR_dA(hw=hw)
        
        return hydraulics.dK_dA_(A=A, n=n, R=R, dR_dA=dR_dA)
    
    def get_roughness_para(self):
        return (self.n_left, self.n_main, self.n_right, self.left_fp_limit, self.right_fp_limit)
    
    def set_roughness_para(self, parameters: tuple):
        self.n_left, self.n_main, self.n_right, self.left_fp_limit, self.right_fp_limit = parameters
        
    def depth_weighted_bed(self, hw: float) -> float:
        if self._is_rect:
            return self.z_min

        dx = np.diff(self.x)
        z = self.z

        # depth at nodes
        d = np.maximum(0.0, hw - z)
        # trapezoidal integral of z(x)*d(x) dx
        z_d_mid = 0.5 * (z[:-1] * d[:-1] + z[1:] * d[1:])
        A = np.dot(0.5 * (d[:-1] + d[1:]), dx)  # same as cs.area(hw)
        if A <= 0:
            # no wetted area, fallback to mean bed
            return float(np.dot(0.5*(z[:-1]+z[1:]), dx) / np.sum(dx))
        return float(np.sum(z_d_mid * dx) / A)

    def z_at(self, x):
        if self._is_rect:
            if x <=0 or x >= self.width:
                return np.inf
            else:
                return self.z_min
        else:
            return np.interp(x, self.x, self.z, left=self.z[0], right=self.z[-1])