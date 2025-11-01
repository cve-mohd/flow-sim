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
                 initial_flow: float,
                 roughness: float = None,
                 width: float = None,
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
        self.initial_conditions = None
        self.conditions_initialized = False
        
        self.initial_flow_rate = initial_flow
        self.roughness = roughness
        self.width = width
        
        self.length = downstream_boundary.chainage - upstream_boundary.chainage
        self.upstream_boundary = upstream_boundary
        self.downstream_boundary = downstream_boundary       
                
        if interpolation_method in ['linear', 'GVF_equation', 'steady-state']:
            self.interpolation_method = interpolation_method
        else:
            raise ValueError("Invalid interpolation method.")
        
        self.xs_chainages = None
        self.input_xs = None
        self.ch_at_node = None
        self.xs_at_node: list[CrossSection] = None
        self.coords_chainages = None
        self.coords = None
    
    def Se(self, A: float, Q: float, i: int) -> float:
        """Computes the energy slope at a location.

        Args:
            A (float): Cross-sectional flow area.
            Q (float): Flow rate.
            i (int): Index of spatial node where Se is computed.

        Returns:
            float: Energy slope (Se)
        """
        h = self.depth_at(i=i, area=A)
        xs: CrossSection = self.xs_at_node[i]
        
        Sf = xs.friction_slope(h=h, Q=Q)
        Sc = xs.curvature_slope(h=h, Q=Q)
            
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
        h = self.depth_at(i=i, area=A)
        xs: CrossSection = self.xs_at_node[i]
        
        dSf_dA = xs.dSf_dA(h=h, Q=Q)
        dSc_dA = xs.dSc_dA(h=h, Q=Q)
            
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
        h = self.depth_at(i=i, area=A)
        xs: CrossSection = self.xs_at_node[i]
        
        dSf_dQ = xs.dSf_dQ(h=h, Q=Q)
        dSc_dQ = xs.dSc_dQ(h=h, Q=Q)
            
        return dSf_dQ + dSc_dQ
            
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
        self._initialize_geometry(n_nodes=n_nodes)
        
        self.initial_conditions = np.zeros(shape=(n_nodes, 2), dtype=np.float64)
        Q = self.initial_flow_rate
        
        if self.interpolation_method == 'linear':
            self._linear_conditions(n_nodes, Q)
                
        elif self.interpolation_method == 'GVF_equation':
            self._gvh_conditions(n_nodes, Q)
        
        elif self.interpolation_method == 'steady-state':
            self._steady_conditions(n_nodes, Q)
            
        else:
            raise ValueError("Invalid flow type.")
        
        self.conditions_initialized = True
    
    def set_coords(self, coords: list | np.ndarray, chainages: list | np.ndarray) -> None:
        """Sets horizontal coordinates along the channel.

        Args:
            coords (list | np.ndarray): Coordinates (list or 1D array of x,y pairs).
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
    
    def area_at(self, i, h):
        xs: CrossSection = self.xs_at_node[i]
        hw = h + xs.bed
        return xs.area(hw)

    def hydraulic_radius(self, i, h):
        xs: CrossSection = self.xs_at_node[i]
        hw = h + xs.bed
        return xs.hydraulic_radius(hw)

    def top_width(self, i, h):
        xs: CrossSection = self.xs_at_node[i]
        hw = h + xs.bed
        return xs.top_width(hw)
        
    def depth_at(self, i, area):
        """
        Compute depth h for node i given target flow area A_target.
        Evaluates A(h) directly from the cross_section.area() function.
        """
        xs: CrossSection = self.xs_at_node[i]
        return xs.depth_at(A_target=area)

    def bed_level_at(self, i):
        xs: CrossSection = self.xs_at_node[i]
        return xs.bed
    
    def dh_dA(self, i, h):
        xs: CrossSection = self.xs_at_node[i]
        return xs.dh_dA(hw=xs.bed+h)
            
    def _initialize_geometry(self, n_nodes: int):
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
            self._create_provisional_cross_sections()

        self._interpolate_chainages(n_nodes)
        
        self._interpolate_cross_sections()

    def _interpolate_cross_sections(self):
        if self.coords_chainages is not None and self.coords is not None:
            curvatures = self._calc_curvature()
        else:
            curvatures = np.zeros_like(self.ch_at_node)
            
        self.xs_at_node = []
        for i, s in enumerate(self.ch_at_node):
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
                dx = max(dx, (x_max-x_min)*1e-4, 0.01)
                X = np.arange(x_min, x_max + dx, dx)

                z1 = xs_left.bed * np.ones_like(X) if xs_left._is_rect else np.interp(X, xs_left.x, xs_left.z, left=xs_left.z[0], right=xs_left.z[-1])
                z2 = xs_right.bed * np.ones_like(X) if xs_right._is_rect else np.interp(X, xs_right.x, xs_right.z, left=xs_right.z[0], right=xs_right.z[-1])
                Z = (1 - alpha) * z1 + alpha * z2
                
                # construct interpolated cross-section
                cs_interp = CrossSection(x=X, z=Z)
                
            n_left  = (1 - alpha) * xs_left.n_left  + alpha * xs_right.n_left
            n_main  = (1 - alpha) * xs_left.n_main  + alpha * xs_right.n_main
            n_right = (1 - alpha) * xs_left.n_right + alpha * xs_right.n_right

            # interpolate floodplain limits
            left_fp_limit  = (1 - alpha) * xs_left.left_fp_limit  + alpha * xs_right.left_fp_limit
            right_fp_limit = (1 - alpha) * xs_left.right_fp_limit + alpha * xs_right.right_fp_limit

            cs_interp.n_left = n_left
            cs_interp.n_main = n_main
            cs_interp.n_right = n_right
            cs_interp.left_fp_limit = left_fp_limit
            cs_interp.right_fp_limit = right_fp_limit
            cs_interp.curvature = curvatures[i]
            cs_interp.bed_slope = (xs_right.bed - xs_left.bed) / (ch_left - ch_right)
                
            self.xs_at_node.append(cs_interp)

        self.upstream_boundary.cross_section = self.xs_at_node[0]
        self.downstream_boundary.cross_section = self.xs_at_node[-1]
        
    def _calc_curvature(self):
        x = np.interp(self.ch_at_node, self.coords_chainages, self.coords[:, 0])
        y = np.interp(self.ch_at_node, self.coords_chainages, self.coords[:, 1])
        curvatures = compute_radii_curv(x_coords=x, y_coords=y)
        return curvatures

    def _interpolate_chainages(self, n_nodes):
        self.ch_at_node = np.linspace(self.upstream_boundary.chainage, self.downstream_boundary.chainage, n_nodes)

    def _create_provisional_cross_sections(self):
        us_xs = CrossSection(width=self.width, bed = self.upstream_boundary.bed_level, n=self.roughness)
        ds_xs = CrossSection(width=self.width, bed=self.downstream_boundary.bed_level, n=self.roughness)
        
        bed_slope = (self.upstream_boundary.bed_level - self.downstream_boundary.bed_level) / self.length
        us_xs.bed_slope = bed_slope
        ds_xs.bed_slope = bed_slope
        
        self.upstream_boundary.cross_section = us_xs
        self.downstream_boundary.cross_section = ds_xs
            
        self.xs_chainages = [self.upstream_boundary.chainage, self.downstream_boundary.chainage]
        self.input_xs = [us_xs, ds_xs]

    def _steady_conditions(self, n_nodes, Q):
        for i in range(n_nodes):
            xs: CrossSection = self.xs_at_node[i]
            A = xs.normal_area(Q_target=Q)
                
            self.initial_conditions[i, 0] = A
            self.initial_conditions[i, 1] = Q

    def _gvh_conditions(self, n_nodes, Q):
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

    def _linear_conditions(self, n_nodes, Q):
        h0 = self.upstream_boundary.initial_depth
        hN = self.downstream_boundary.initial_depth
                    
        for i in range(n_nodes):
            distance = self.length * i / (n_nodes-1)
                
            h = h0 + (hN - h0) * distance / self.length
            A = self.area_at(i=i, h=h)
                
            self.initial_conditions[i, 0] = A
            self.initial_conditions[i, 1] = Q