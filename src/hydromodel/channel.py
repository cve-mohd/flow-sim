from .boundary import Boundary
from .utility import compute_radii_curv
from . import hydraulics
import numpy as np
from .cross_section import CrossSection

class Channel:
    """
    Represents a channel with hydraulic and geometric attributes.        
    """
    def __init__(self,
                 upstream_boundary: Boundary,
                 downstream_boundary: Boundary,
                 width: float,
                 initial_flow: float,
                 roughness: float,
                 interpolation_method: str = 'GVF_equation'):
        """Initializes a Channel object.

        Args:
            upstream_boundary (Boundary): Upstream boundary object.
            downstream_boundary (Boundary): Downstream boundary object.
            width (float): Cross-sectional width.
            initial_flow (float): Flow rate through the channel at t=0.
            roughness (float): Manning's roughness coefficient.
            dry_roughness (float, optional): Manning's roughness coefficient for initially-dry regions. Defaults to None.
            interpolation_method (str, optional): Method of computing initial conditions. Defaults to 'GVF_equation'.
            n_steepness (float, optional): Steepness of the n transition curve. Defaults to 0.15.
        """
        self.conditions_initialized = False
        self.initial_flow_rate = initial_flow
        self.roughness = roughness
        
        #
        us_xs = CrossSection(width=width, bed=  upstream_boundary.bed_level, n=roughness)
        ds_xs = CrossSection(width=width, bed=downstream_boundary.bed_level, n=roughness)
        #
        
        self.length = downstream_boundary.chainage - upstream_boundary.chainage
        self.upstream_boundary = upstream_boundary
        self.downstream_boundary = downstream_boundary       
                
        if interpolation_method in ['linear', 'GVF_equation', 'steady-state']:
            self.interpolation_method = interpolation_method
        else:
            raise ValueError("Invalid interpolation method.")
        
        self.initial_conditions = None
        self.coordinated = False
        self.radii_curv = None
        
        self.xs_chainages = [upstream_boundary.chainage, downstream_boundary.chainage]         # station coordinates [m]
        self.input_xs = [us_xs, ds_xs]          # list of cross_section objects
        self.ch_at_node = None    # spatial grid along the channel
        self.xs_at_node: list[CrossSection] = None          # interpolated cross_section at each node
        self.hw_grid = None           # elevation grid for lookup tables
        self.A_table = None
        self.P_table = None
        self.T_table = None
        self.bed_level = None
    
    def Se(self, A: float, Q: float, i: int) -> float:
        """Computes the energy slope at a location.

        Args:
            A (float): Cross-sectional flow area.
            Q (float): Flow rate.
            i (int): Index of spatial node where Se is computed.

        Returns:
            float: Energy slope (Se)
        """
        n = self.get_n(A=A, i=i)
        h = self.depth_at(i=i, A_target=A)
        R = self.hydraulic_radius(i=i, h=h)
        T = self.top_width(i=i, h=h)
        
        Sf = hydraulics.Sf(A=A, Q=Q, n=n, R=R)
        
        if self.coordinated:
            rc = self.radii_curv[i]
            Sc = hydraulics.Sc(h=h, T=T, A=A, Q=Q, n=n, R=R, rc=rc)
        else:
            Sc = 0
            
        return Sf + Sc
    
    def dSe_dA(self, A: float, Q: float, i: int) -> float:
        """Computes the derivative of the energy slope w.r.t. flow area.

        Args:
            A (float): Cross-sectional flow area.
            Q (float): Flow rate.
            i (int): Index of spatial node where Se is computed.

        Returns:
            float: dSe/dA
        """
        n = self.get_n(A=A, i=i)
        h = self.depth_at(i=i, A_target=A)
        R = self.hydraulic_radius(i=i, h=h)
        dR_dA = self.dR_dA(i=i, h=h)
        T = self.top_width(i=i, h=h)
        
        dSf_dA = hydraulics.dSf_dA(A=A, Q=Q, n=n, R=R, dR_dA=dR_dA)
        
        if self.coordinated:
            rc = self.radii_curv[i]
            dSc_dA = hydraulics.dSc_dA(h=h, A=A, Q=Q, n=n, R=R, rc=rc, dR_dA=dR_dA, T=T)
        else:
            dSc_dA = 0
        
        return dSf_dA + dSc_dA
    
    def dSe_dQ(self, A: float, Q: float, i: int) -> float:
        """Computes the derivative of the energy slope w.r.t. flow rate.

        Args:
            A (float): Cross-sectional flow area.
            Q (float): Flow rate.
            i (int): Index of spatial node where Se is computed.

        Returns:
            float: dSe/dQ
        """
        n = self.get_n(A=A, i=i)
        h = self.depth_at(i=i, A_target=A)
        R = self.hydraulic_radius(i=i, h=h)
        T = self.top_width(i=i, h=h)
        
        dSf_dQ = hydraulics.dSf_dQ(A=A, Q=Q, n=n, R=R)
        
        if self.coordinated:
            rc = self.radii_curv[i]
            dSc_dQ = hydraulics.dSc_dQ(h=h, T=T, A=A, Q=Q, n=n, R=R, rc=rc)
        else:
            dSc_dQ = 0
                
        return dSf_dQ + dSc_dQ

    def normal_flow(self, A: float, i: int) -> float:
        """Computes the normal flow rate for a given flow area at a given location.

        Args:
            A (float): Cross-sectional flow area.
            i (int): Spatial node index.

        Returns:
            float: Normal flow rate.
        """
        raise ValueError("channel.normal_flow is WIP")
        n = self.get_n(A=A, i=i)
        h = self.depth_at(i=i, A_target=A)
        R = self.hydraulic_radius(i=i, h=h)
        S_0 = self.bed_slopes[i]
        
        return hydraulics.normal_flow(A=A, S_0=S_0, n=n, R=R)
        
    def normal_area(self, Q: float, i: int) -> float:
        """Computes the normal flow area for a given flow rate at a given location.

        Args:
            Q (float): Flow rate.
            i (int): Spatial node index.

        Returns:
            float: Normal flow area.
        """
        raise ValueError("channel.normal_area is WIP")
        R = self.hydraulic_radius(i=i, h=h)
        B = self.width[i]
        h = self.downstream_boundary.initial_depth
        h = 1 if h is None else h
        A_guess = h * B
        n = self.get_n(A=A_guess, i=i)
        S_0 = self.bed_slopes[i]
        
        return hydraulics.normal_area(Q=Q, A_guess=A_guess, S_0=S_0, n=n, R=R)
            
    def initialize_conditions(self, n_nodes: int) -> None:
        """
        Computes the initial values of the flow variables (A, Q) at each node.
        
        Parameters
        ----------
        n_nodes : int
            Number of spatial nodes along the channel.
            
        Returns
        -------
        None.

        """
        self.initialize_geometry(n_nodes=n_nodes)
        self.initial_conditions = np.zeros(shape=(n_nodes, 2), dtype=np.float64)
        Q = self.initial_flow_rate
        
        if self.interpolation_method == 'linear':
            h0 = self.upstream_boundary.initial_depth
            hN = self.downstream_boundary.initial_depth
                    
            for i in range(n_nodes):
                distance = self.length * i / (n_nodes-1)
                
                h = h0 + (hN - h0) * distance / self.length
                A = self.area_at(i=i, h=h)
                
                self.initial_conditions[i, 0] = A
                self.initial_conditions[i, 1] = Q
                
        elif self.interpolation_method == 'GVF_equation':
            dx = self.length / (n_nodes - 1)
            h = self.downstream_boundary.initial_depth
            
            # Add last node
            A = self.area_at(i=-1, h=h)
            self.initial_conditions[n_nodes-1, 0] = A
            self.initial_conditions[n_nodes-1, 1] = Q

            for i in reversed(range(n_nodes-1)):
                Fr = hydraulics.froude_num(T=self.top_width(i=i, h=h), A=A, Q=Q)
                denominator = 1 - Fr**2
                
                if abs(denominator) < 1e-6:
                    dh_dx = 0.0
                else:
                    dz = self.bed_level_at(i+1) - self.bed_level_at(i)
                    S0 = -dz/dx
                    Sf = self.Se(A=A, Q=Q, i=i)
                    dh_dx = (S0 - Sf) / denominator

                h -= dh_dx * dx

                if h < 0:
                    raise ValueError("GVF failed.")

                A = self.area_at(i=i, h=h)
                
                self.initial_conditions[i, 0] = A
                self.initial_conditions[i, 1] = Q
        
        elif self.interpolation_method == 'steady-state':
            for i in range(n_nodes):
                A = self.normal_area(Q, i)
                
                self.initial_conditions[i, 0] = A
                self.initial_conditions[i, 1] = Q
            
        else:
            raise ValueError("Invalid flow type.")
        
        self.conditions_initialized = True
    
    def get_equivalent_n(self, i, h):
        """
        Compute equivalent Manning's n for node i at flow depth h.
        Uses separate roughness values for left floodplain, main channel, and right floodplain.
        """
        cs: CrossSection = self.xs_at_node[i]
        hw = cs.bed + h  # water surface elevation

        # Helper to compute subsection properties within x-range
        def subsection_props(x_min, x_max, n_value):
            mask = (cs.x >= x_min) & (cs.x <= x_max)
            if mask.sum() < 2:
                return 0.0, 0.0, 0.0
            xs, zs = cs.x[mask], cs.z[mask]
            sub_cs = CrossSection(x=xs, z=zs)
            A = sub_cs.area(hw)
            P = sub_cs.wetted_perimeter(hw)
            if A <= 0 or P <= 0:
                return 0.0, 0.0, 0.0
            R = sub_cs.hydraulic_radius(hw)
            K = (1.0 / n_value) * A * (R ** (2.0 / 3.0))
            return A, R, K

        # Subsections
        left_A, left_R, left_K = subsection_props(cs.x[0], cs.left_fp_limit, cs.n_left)
        main_A, main_R, main_K = subsection_props(cs.left_fp_limit, cs.right_fp_limit, cs.n_main)
        right_A, right_R, right_K = subsection_props(cs.right_fp_limit, cs.x[-1], cs.n_right)

        # Total area and hydraulic radius
        A_total = left_A + main_A + right_A
        P_total = cs.wetted_perimeter(hw)
        if A_total <= 0 or P_total <= 0:
            return cs.n_main  # fallback

        R_total = A_total / P_total

        # Combine conveyances
        K_total = (left_K**1.5 + main_K**1.5 + right_K**1.5) ** (2.0 / 3.0)

        # Equivalent Manning's n
        n_eq = (A_total * (R_total ** (2.0 / 3.0))) / K_total
        return n_eq

    def get_n(self, A: float = None, i: int = None) -> float:
        """Retrieves Manning's roughness coefficient.

        Args:
            A (float, optional): Cross-sectional flow area. Defaults to None.
            i (int, optional): Index of spatial node. Defaults to None.

        Returns:
            float: n
        """
        h = self.depth_at(i=i, A_target=A)
        return self.get_equivalent_n(i=i, h=h)
        if self.dry_roughness is None or not self.conditions_initialized:
            return self.roughness
        
        if A is None or i is None:
            raise ValueError("Insufficient parameters.")
        
        wet_h = self.wet_depth_at(i)
        h = self.depth_at(i=i, A_target=A)
        
        return hydraulics.effective_roughness(depth=h, wet_roughness=self.roughness, dry_roughness=self.dry_roughness, wet_depth=wet_h, steepness=self.n_steepness)
              
    def dn_dA(self, A: float, i: int) -> float:
        """Computes the derivative of Manning's roughness coefficient w.r.t. flow area.

        Args:
            A (float): Cross-sectional flow area.
            i (int): Index of spatial node.

        Returns:
            float: dn/dA
        """
        return 0
        if self.dry_roughness is None:
            return 0
        
        wet_h = self.wet_depth_at(i)
        h = self.depth_at(i=i, A_target=A)
        
        dn_dh = hydraulics.dn_dh(depth=h,
                                 steepness=self.n_steepness,
                                 roughness=self.roughness,
                                 dry_roughness=self.dry_roughness,
                                 wet_depth=wet_h)
        # dn/dA = dn/dh * dh/dA, dh/dA = 1/B
        return dn_dh * self.dh_dA(i=i, h=h)
    
    def set_coords(self, coords: list | np.ndarray, chainages: list | np.ndarray) -> None:
        """Sets horizontal coordinates along the channel.

        Args:
            coords (list | np.ndarray): Coordinates (list of x,y pairs).
            chainages (list | np.ndarray): Respective chainages along the channel.
        """
        self.coords_chainages = np.asarray(chainages, dtype=np.float64)
        self.coords = np.asarray(coords, dtype=np.float64)
        
        self.upstream_boundary.chainage = self.coords_chainages[0]
        self.downstream_boundary.chainage = self.coords_chainages[-1]
        self.length = self.downstream_boundary.chainage - self.upstream_boundary.chainage
        self.coordinated = True
    
    def set_cross_sections(self, chainages, sections: list[CrossSection]):
        """
        Register base cross-sections and their station coordinates.

        Parameters
        ----------
        chainages : 1D array-like
            Distances along the channel centerline [m], increasing downstream.
        sections : list of cross_section
            Cross-section objects at those chainages.
        """
        chainages = np.asarray(chainages, dtype=float)
        if len(chainages) != len(sections):
            raise ValueError("chainages and sections must have same length")
        if not np.all(np.diff(chainages) > 0):
            raise ValueError("chainages must be strictly increasing")

        self.xs_chainages = chainages
        self.input_xs = sections
        
    def initialize_geometry(self, n_nodes: int, n_hw: int = 201, dx_interp: float = 0.5):
        """
        Precompute interpolated cross-sections and stage tables
        for each computational node.

        Parameters
        ----------
        n_nodes : int
            Number of spatial nodes along the channel.
        n_hw : int, optional
            Number of water-surface elevations for lookup tables.
        dx_interp : float, optional
            Lateral spacing (m) used when interpolating between sections.
        """
        if self.xs_chainages is None or self.input_xs is None:
            raise RuntimeError("set_cross_sections must be called first")

        n_nodes = n_nodes
        ch_us, ch_ds = self.upstream_boundary.chainage, self.downstream_boundary.chainage
        self.ch_at_node = np.linspace(ch_us, ch_ds, n_nodes)
        
        if self.coordinated:
            x = np.interp(self.ch_at_node, self.coords_chainages, self.coords[:, 0])
            y = np.interp(self.ch_at_node, self.coords_chainages, self.coords[:, 1])
            self.curv, self.radii_curv = compute_radii_curv(x_coords=x, y_coords=y)

        # determine elevation range across all sections
        # z_min = min(cs.bed for cs in self.input_xs)
        # z_max = max(cs.bed for cs in self.input_xs)
        # self.hw_grid = np.linspace(z_min, z_max + 5.0, n_hw)

        # precompute interpolated cross-sections at each node
        self.xs_at_node = []
        for s in self.ch_at_node:
            # find bounding sections
            if s <= self.xs_chainages[0]:
                self.xs_at_node.append(self.input_xs[0])
                continue
            if s >= self.xs_chainages[-1]:
                self.xs_at_node.append(self.input_xs[-1])
                continue

            j = np.searchsorted(self.xs_chainages, s) - 1
            ch_left = self.xs_chainages[j]
            ch_right = self.xs_chainages[j + 1]
            alpha = (s - ch_left) / (ch_right - ch_left)

            xs_left: CrossSection = self.input_xs[j]
            xs_right: CrossSection = self.input_xs[j + 1]

            # interpolate geometry
            if xs_left._is_rect and xs_right._is_rect:
                width = (1 - alpha) * xs_left.width + alpha * xs_right.width
                bed = (1 - alpha) * xs_left.bed + alpha * xs_right.bed
                cs_interp = CrossSection(width=width, bed=bed)
            else:
                x_min = min(0 if xs_left._is_rect else xs_left.x[0],
                            0 if xs_right._is_rect else xs_right.x[0])
                x_max = max(xs_left.width if xs_left._is_rect else xs_left.x[-1],
                            xs_right if xs_right._is_rect else xs_right.x[-1])
                
                dx1 = x_max - x_min if xs_left._is_rect else np.min(xs_left.x[1:] - xs_left.x[:-1])
                dx2 = x_max - x_min if xs_right._is_rect else np.min(xs_right.x[1:] - xs_right.x[:-1])
                
                dx = min(dx1, dx2)
                X = np.arange(x_min, x_max + dx, dx)

                z1 = xs_left.bed * np.ones_like(X) if xs_left._is_rect else np.interp(X, xs_left.x, xs_left.z, left=xs_left.z[0], right=xs_left.z[-1])
                z2 = xs_right.bed * np.ones_like(X) if xs_right._is_rect else np.interp(X, xs_right.x, xs_right.z, left=xs_right.z[0], right=xs_right.z[-1])
                Z = (1 - alpha) * z1 + alpha * z2
                
                n_left  = (1 - alpha) * xs_left.n_left  + alpha * xs_right.n_left
                n_main  = (1 - alpha) * xs_left.n_main  + alpha * xs_right.n_main
                n_right = (1 - alpha) * xs_left.n_right + alpha * xs_right.n_right

                # interpolate floodplain limits
                left_fp_limit  = (1 - alpha) * xs_left.left_fp_limit  + alpha * xs_right.left_fp_limit
                right_fp_limit = (1 - alpha) * xs_left.right_fp_limit + alpha * xs_right.right_fp_limit

                # construct interpolated cross-section
                cs_interp = CrossSection(x=X, z=Z)
                cs_interp.n_left = n_left
                cs_interp.n_main = n_main
                cs_interp.n_right = n_right
                cs_interp.left_fp_limit = left_fp_limit
                cs_interp.right_fp_limit = right_fp_limit
                
            self.xs_at_node.append(cs_interp)

        # build lookup tables for A, P, T
        """self.A_table = np.empty((n_nodes, n_hw), dtype=float)
        self.P_table = np.empty_like(self.A_table)
        self.T_table = np.empty_like(self.A_table)
        
        for i, cs in enumerate(self.xs_at_node):
            for j, hw in enumerate(self.hw_grid):
                A, P, R, T = cs.properties(hw)
                self.A_table[i, j] = A
                self.P_table[i, j] = P
                self.T_table[i, j] = T"""
                
        self.bed_level = np.array(object=[xs.bed for xs in self.xs_at_node], dtype=np.float64)

    def _lookup(self, table, i, h):
        """
        Generic table lookup by node index and flow depth h (m).
        Depth is converted to water-surface elevation.
        """
        cs: CrossSection = self.xs_at_node[i]
        hw = cs.bed + h
        idx = np.searchsorted(self.hw_grid, hw, side="right")
        idx0 = max(0, idx - 1)
        idx1 = min(len(self.hw_grid) - 1, idx)
        h0 = self.hw_grid[idx0]
        h1 = self.hw_grid[idx1]
        if h1 == h0:
            return table[i, idx0]
        t = (hw - h0) / (h1 - h0)
        return (1 - t) * table[i, idx0] + t * table[i, idx1]

    def area_at(self, i, h):
        cs: CrossSection = self.xs_at_node[i]
        hw = h + cs.bed
        return cs.area(hw)
        return self._lookup(self.A_table, i, h)

    def hydraulic_radius(self, i, h):
        cs: CrossSection = self.xs_at_node[i]
        hw = h + cs.bed
        return cs.hydraulic_radius(hw)
        A = self._lookup(self.A_table, i, h)
        P = self._lookup(self.P_table, i, h)
        return A / P if P > 0 else 0.0

    def top_width(self, i, h):
        cs: CrossSection = self.xs_at_node[i]
        hw = h + cs.bed
        return cs.top_width(hw)
        return self._lookup(self.T_table, i, h)
        
    def depth_at(self, i, A_target, h_min=0.0, h_max=None, tol=1e-8, max_iter=30):
        """
        Compute depth h for node i given target flow area A_target.
        Evaluates A(h) directly from the cross_section.area() function.
        """
        cs: CrossSection = self.xs_at_node[i]

        # Set default upper bound (covers all likely depths)
        if h_max is None:
            base = 0 if cs._is_rect else (max(cs.z) - min(cs.z))
            h_max = base + 100.0  # large enough upper bound
            A_high = self.area_at(i, h_max)

        A_low = self.area_at(i, h_min)
        A_high = self.area_at(i, h_max)

        if not (A_low <= A_target <= A_high):
            raise ValueError("A_target outside valid area range for this cross-section")

        for _ in range(max_iter):
            h_mid = 0.5 * (h_min + h_max)
            A_mid = self.area_at(i, h_mid)

            if abs(A_mid - A_target) < tol:
                return h_mid

            if A_mid < A_target:
                h_min = h_mid
            else:
                h_max = h_mid

        # Return midpoint if not converged
        return 0.5 * (h_min + h_max)

        """
        Compute flow depth h at node i given target flow area A_target.
        Uses bisection search on precomputed cross-section geometry.
        """
        cs = self.xs_at_node[i]
        
        # Default search bounds
        if h_min is None:
            h_min = 0.0
        if h_max is None:
            base = 0 if cs._is_rect else (max(cs.z) - min(cs.z))
            h_max = base + 50.0  # large enough upper bound
            A_high = cs.area(h_max)
            
            while A_high <= A_target:
                h_max = h_max * A_target/A_high + 0.1
                A_high = cs.area(h_max)

        A_low = cs.area(h_min)

        # Check monotonicity
        if not (A_low <= A_target <= A_high):
            raise ValueError("A_target outside valid area range for this cross-section")

        # Bisection iteration
        for _ in range(max_iter):
            h_mid = 0.5 * (h_min + h_max)
            A_mid = cs.area(h_mid)

            if abs(A_mid - A_target) < tol:
                return h_mid
            if A_mid < A_target:
                h_min = h_mid
            else:
                h_max = h_mid

        # Fallback if not converged
        return 0.5 * (h_min + h_max)
    
    def dR_dA(self, i, h, dh=1e-6):
        cs: CrossSection = self.xs_at_node[i]
        hw = h + cs.bed
        A1 = cs.area(hw - dh)
        A2 = cs.area(hw + dh)
        R1 = cs.hydraulic_radius(hw - dh)
        R2 = cs.hydraulic_radius(hw + dh)
        return (R2 - R1) / (A2 - A1)

    def bed_level_at(self, i):
        return self.bed_level[i]
    
    def dh_dA(self, i, h, dh=1e-6):
        cs: CrossSection = self.xs_at_node[i]
        hw = h + cs.bed
        A1 = cs.area(hw - dh)
        A2 = cs.area(hw + dh)
                
        return 2 * dh / (A2 - A1)
