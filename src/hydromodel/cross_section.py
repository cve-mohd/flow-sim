import numpy as np

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

            # cache
            self._last_hw = None
            self._last_res = None
            return

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
        self.n_left = None   # Manning's n for left floodplain
        self.n_main = None   # Manning's n for main channel
        self.n_right = None  # Manning's n for right floodplain
        self.left_fp_limit = None   # x-coordinate dividing left floodplain and main channel
        self.right_fp_limit = None  # x-coordinate dividing main channel and right floodplain
        
        if n is not None:
            self.n_left = n
            self.n_main = n
            self.n_right = n
            self.left_fp_limit = 0
            self.right_fp_limit = self.width

        # cache
        self._last_hw = None
        self._last_res = None

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
