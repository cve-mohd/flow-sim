import numpy as np
from . import hydraulics

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
            self.bed = float(bed)

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
            self.bed = float(np.min(self.z))
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
        """Return (A, P, R, T) for scalar hw. Results are cached for repeated calls
        with the same hw to avoid recomputation.
        """
        hw = float(hw)
        if self._last_hw is not None and hw == self._last_hw:
            return self._last_res

        if self._is_rect:
            depth = max(0.0, hw - self.bed)
            A = self.width * depth
            if depth <= 0.0:
                P = 0.0
                T = 0.0
                R = 0.0
            else:
                # vertical walls assumed
                P = self.width + 2.0 * depth
                T = self.width
                R = A / P
            res = (A, P, R, T)
            self._last_hw = hw
            self._last_res = res
            return res

        # General polyline
        # Depths at nodes
        d = hw - self.z
        d_clip = np.maximum(d, 0.0)

        # Area using trapezoidal rule on depths vs x
        # A = sum( (d_i + d_{i+1})/2 * dx_i )
        A = np.dot((d_clip[:-1] + d_clip[1:]) * 0.5, self._dx)

        # Top width: find leftmost and rightmost intersection with hw
        below = d > 0.0
        if not np.any(below):
            T = 0.0
        else:
            i_left = np.argmax(below)            # first True
            i_right = self._n - np.argmax(below[::-1]) - 1  # last True

            # left intersection
            if i_left == 0:
                xl = self.x[0]
            else:
                z0 = self.z[i_left - 1]
                z1 = self.z[i_left]
                x0 = self.x[i_left - 1]
                x1 = self.x[i_left]
                if z1 == z0:
                    xl = x0
                else:
                    t = (hw - z0) / (z1 - z0)
                    t = np.clip(t, 0.0, 1.0)
                    xl = x0 + t * (x1 - x0)

            # right intersection
            if i_right == self._n - 1:
                xr = self.x[-1]
            else:
                z0 = self.z[i_right]
                z1 = self.z[i_right + 1]
                x0 = self.x[i_right]
                x1 = self.x[i_right + 1]
                if z1 == z0:
                    xr = x1
                else:
                    t = (hw - z0) / (z1 - z0)
                    t = np.clip(t, 0.0, 1.0)
                    xr = x0 + t * (x1 - x0)

            T = max(0.0, float(xr - xl))

        # Wetted perimeter
        # Cases per segment:
        #  - both below hw -> full seg_len
        #  - both above -> 0
        #  - partial -> fraction of seg_len up to intersection
        z0 = self._z0
        z1 = self._z1
        seg = self._seg_len

        both_sub = (z0 < hw) & (z1 < hw)
        both_dry = (z0 >= hw) & (z1 >= hw)
        partial_mask = ~(both_sub | both_dry)

        P = float(np.sum(seg[both_sub]))

        if np.any(partial_mask):
            z0_p = z0[partial_mask]
            z1_p = z1[partial_mask]
            seg_p = seg[partial_mask]
            # compute intersection parameter t where z(t)=hw, t in [0,1]
            t = (hw - z0_p) / (z1_p - z0_p)
            t = np.clip(t, 0.0, 1.0)
            # if z0 < hw and z1 >= hw -> submerged fraction = t
            # if z0 >= hw and z1 < hw -> submerged fraction = 1 - t
            mask_z0_below = z0_p < hw
            frac = np.empty_like(t)
            frac[mask_z0_below] = t[mask_z0_below]
            frac[~mask_z0_below] = 1.0 - t[~mask_z0_below]
            # add absolute segment portions
            P += float(np.sum(seg_p * np.abs(frac)))

        R = A / P if P > 0.0 else 0.0

        res = (float(A), float(P), float(R), float(T))
        self._last_hw = hw
        self._last_res = res
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
        hw = h+self.bed
        
        n = self.get_equivalent_n(hw=hw)
        R = self.hydraulic_radius(hw=hw)
        A = self.area(hw=hw)
        
        return hydraulics.Sf(A=A, Q=Q, n=n, R=R)
    
    def curvature_slope(self, h: float, Q: float) -> float:
        if self.curvature == 0:
            return 0.0
        
        hw = h+self.bed
        
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
        
        def subsection_props(x_min, x_max, n_value):
            if self._is_rect:
                width = x_max - x_min
                sub_xs = CrossSection(width=width, bed=self.bed)
            else:
                mask = (self.x >= x_min) & (self.x <= x_max)
                if mask.sum() < 2:
                    return 0.0, 0.0, 0.0
                xs, zs = self.x[mask], self.z[mask]
                sub_xs = CrossSection(x=xs, z=zs)
                
            A = sub_xs.area(hw)
            P = sub_xs.wetted_perimeter(hw)
            if A <= 0 or P <= 0:
                return 0.0, 0.0, 0.0
            R = sub_xs.hydraulic_radius(hw)
            K = (1.0 / n_value) * A * (R ** (2.0 / 3.0))
            return A, R, K

        # Subsections
        left_A, left_R, left_K = subsection_props(0 if self._is_rect else self.x[0], self.left_fp_limit, self.n_left)
        main_A, main_R, main_K = subsection_props(self.left_fp_limit, self.right_fp_limit, self.n_main)
        right_A, right_R, right_K = subsection_props(self.right_fp_limit, self.width if self._is_rect else self.x[-1], self.n_right)

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

    def dSf_dA(self, h: float, Q: float) -> float:
        hw = h+self.bed
        
        n = self.get_equivalent_n(hw=hw)
        R = self.hydraulic_radius(hw=hw)
        A = self.area(hw=hw)
        dR_dA = self.dR_dA(hw=hw)
        
        return hydraulics.dSf_dA(A=A, Q=Q, n=n, R=R, dR_dA=dR_dA)
    
    def dSc_dA(self, h: float, Q: float) -> float:
        if self.curvature == 0:
            return 0.0
        
        hw = h+self.bed
        
        n = self.get_equivalent_n(hw=hw)
        R = self.hydraulic_radius(hw=hw)
        T = self.top_width(hw=hw)
        A = self.area(hw=hw)
        dR_dA = self.dR_dA(hw=hw)
        
        return hydraulics.dSc_dA(h=h, A=A, Q=Q, n=n, R=R, rc=(1.0 / self.curvature), dR_dA=dR_dA, T=T)
        
    def dSf_dQ(self, h: float, Q: float) -> float:
        hw = h+self.bed
        
        n = self.get_equivalent_n(hw=hw)
        R = self.hydraulic_radius(hw=hw)
        A = self.area(hw=hw)
        
        return hydraulics.dSf_dQ(A=A, Q=Q, n=n, R=R)
    
    def dSc_dQ(self, h: float, Q: float) -> float:
        if self.curvature == 0:
            return 0.0
        
        hw = h+self.bed
        
        n = self.get_equivalent_n(hw=hw)
        R = self.hydraulic_radius(hw=hw)
        T = self.top_width(hw=hw)
        A = self.area(hw=hw)
        
        return hydraulics.dSc_dQ(h=h, T=T, A=A, Q=Q, n=n, R=R, rc=(1.0 / self.curvature))

    def dR_dA(self, hw, dh=1e-6):
        A1 = self.area(hw - dh)
        A2 = self.area(hw + dh)
        R1 = self.hydraulic_radius(hw - dh)
        R2 = self.hydraulic_radius(hw + dh)
        return (R2 - R1) / (A2 - A1)
    
    def depth_at(self, A_target, h_min=0.0, h_max=None, tol=1e-8, max_iter=30):
        """
        Compute depth h for node i given target flow area A_target.
        Evaluates A(h) directly from the cross_section.area() function.
        """
        # Set default upper bound (covers all likely depths)
        if h_max is None:
            base = 0 if self._is_rect else (max(self.z) - min(self.z))
            h_max = base + 100.0  # large enough upper bound
            A_high = self.area(hw=self.bed+h_max)

        A_low = self.area(hw=self.bed+h_min)
        A_high = self.area(hw=self.bed+h_max)

        if not (A_low <= A_target <= A_high):
            raise ValueError("A_target outside valid area range for this cross-section")

        for _ in range(max_iter):
            h_mid = 0.5 * (h_min + h_max)
            A_mid = self.area(hw=self.bed+h_mid)

            if abs(A_mid - A_target) < tol:
                return h_mid

            if A_mid < A_target:
                h_min = h_mid
            else:
                h_max = h_mid

        # Return midpoint if not converged
        return 0.5 * (h_min + h_max)
    
    def dh_dA(self, hw, dh=1e-6):
        A1 = self.area(hw - dh)
        A2 = self.area(hw + dh)
                
        return 2 * dh / (A2 - A1)