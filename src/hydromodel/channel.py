from .boundary import Boundary
from .utility import compute_curv
from . import hydraulics
import numpy as np
from .cross_section import CrossSection, interpolate_cross_section, TrapezoidalSection

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
    
    def Se(self, h: float, Q: float, i: int) -> float:
        """Computes the energy slope at a location.

        Args:
            A (float): Cross-sectional flow area.
            Q (float): Flow rate.
            i (int): Index of spatial node where Se is computed.

        Returns:
            float: Energy slope (Se)
        """
        xs: CrossSection = self.xs_at_node[i]
        
        Sf = xs.friction_slope(h=h, Q=Q)
        Sc = xs.curvature_slope(h=h, Q=Q)
            
        return Sf + Sc
    
    def dSe_dA(self, h: float, Q: float, i: int) -> float:
        """Computes the derivative of the energy slope w.r.t. flow area.

        Args:
            A (float): Cross-sectional flow area.
            Q (float): Flow rate.
            i (int): Index of spatial node where Se is computed.

        Returns:
            float: dSe/dA
        """
        xs: CrossSection = self.xs_at_node[i]
        
        dSf_dA = xs.dSf_dA(h=h, Q=Q)
        dSc_dA = xs.dSc_dA(h=h, Q=Q)
            
        return dSf_dA + dSc_dA
    
    def dSe_dQ(self, h: float, Q: float, i: int) -> float:
        """Computes the derivative of the energy slope w.r.t. flow rate.

        Args:
            A (float): Cross-sectional flow area.
            Q (float): Flow rate.
            i (int): Index of spatial node where Se is computed.

        Returns:
            float: dSe/dQ
        """
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
    
    def area_at(self, i, hw):
        xs: CrossSection = self.xs_at_node[i]
        return xs.area(hw)

    def hydraulic_radius(self, i, hw):
        xs: CrossSection = self.xs_at_node[i]
        return xs.hydraulic_radius(hw)

    def top_width(self, i, hw):
        xs: CrossSection = self.xs_at_node[i]
        return xs.top_width(hw)
        
    def bed_level_at(self, i):
        xs: CrossSection = self.xs_at_node[i]
        return xs.z_min
    
    def dA_dh(self, i, hw):
        xs: CrossSection = self.xs_at_node[i]
        return xs.dA_dh(hw=hw)
            
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
            self._calc_curvature()
            
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

            xs_left: CrossSection = self.input_xs[j]
            xs_right: CrossSection = self.input_xs[j + 1]
            
            cs_interp = interpolate_cross_section(xs1=xs_left,
                                                  xs2=xs_right,
                                                  dist1=s-ch_left,
                                                  dist2=ch_right-s)
            self.xs_at_node.append(cs_interp)

        self.upstream_boundary.cross_section = self.xs_at_node[0]
        self.downstream_boundary.cross_section = self.xs_at_node[-1]
        
    def _calc_curvature(self):
        for i in range(1, len(self.input_xs) - 1):
            ch_left, ch, ch_right = (
                self.xs_chainages[i - 1],
                self.xs_chainages[i],
                self.xs_chainages[i + 1],
            )

            # interpolate to get coordinates along centerline
            chs = np.array([ch_left, ch, ch_right])
            xys = np.column_stack([
                np.interp(chs, self.coords_chainages, self.coords[:, 0]),
                np.interp(chs, self.coords_chainages, self.coords[:, 1])
            ])
            xy_left, xy, xy_right = xys

            # direction vectors
            v1 = xy - xy_left
            v2 = xy_right - xy

            # avoid zero-length vectors
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                curvature = 0.0
            else:
                # turning angle
                dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                theta = np.arccos(np.clip(dot, -1.0, 1.0))

                # average segment length
                L = 0.5 * (np.linalg.norm(v1) + np.linalg.norm(v2))

                # curvature
                curvature = 2 * np.sin(theta / 2) / L * np.sign(np.cross(v1, v2))

            self.input_xs[i].curvature = curvature
        
    def _interpolate_chainages(self, n_nodes):
        self.ch_at_node = np.linspace(self.upstream_boundary.chainage, self.downstream_boundary.chainage, n_nodes)

    def _create_provisional_cross_sections(self):
        us_xs = TrapezoidalSection(b_main=self.width, m_main=0, z_bed = self.upstream_boundary.bed_level, n_main=self.roughness)
        ds_xs = TrapezoidalSection(b_main=self.width, m_main=0, z_bed=self.downstream_boundary.bed_level, n_main=self.roughness)
        
        bed_slope = (us_xs.z_min - ds_xs.z_min) / self.length
        us_xs.bed_slope = bed_slope
        ds_xs.bed_slope = bed_slope
        
        self.upstream_boundary.cross_section = us_xs
        self.downstream_boundary.cross_section = ds_xs
            
        self.xs_chainages = [self.upstream_boundary.chainage, self.downstream_boundary.chainage]
        self.input_xs = [us_xs, ds_xs]

    def _steady_conditions(self, n_nodes, Q):
        for i in range(n_nodes):
            xs: CrossSection = self.xs_at_node[i]
            if xs.bed_slope is None:
                raise ValueError("Bed slope must be defined.")
        
            h = xs.normal_depth(Q_target=Q)
                
            self.initial_conditions[i, 0] = h
            self.initial_conditions[i, 1] = Q

    def _gvh_conditions(self, n_nodes, Q):
        dx = self.length / (n_nodes - 1)
        
        # Start at the downstream boundary
        h = self.downstream_boundary.initial_depth
        self.initial_conditions[n_nodes-1, 0] = h
        self.initial_conditions[n_nodes-1, 1] = Q

        # This helper function calculates dh/dx at a given node
        def get_dh_dx(h_in, node_idx):
            """Calculates dh/dx at a specific node given depth h_in."""
            hw = h_in + self.bed_level_at(i=node_idx)
            A = self.area_at(i=node_idx, hw=hw)
            T = self.top_width(i=node_idx, hw=hw)
            
            # Check for dry or invalid section
            if T < 1e-6 or A < 1e-6:
                return 0.0

            # 1. Check Flow Regime (Critical Check 1)
            Fr = hydraulics.froude_num(T=T, A=A, Q=Q)
            if Fr > 1.0:
                raise RuntimeError(
                    f"GVF Error: Flow became supercritical (Fr={Fr:.2f}) at node {node_idx}. "
                    "Downstream boundary control is not valid for this Q."
                )
            
            Fr_sq = Fr**2
            denominator = 1 - Fr_sq
            
            # 2. Prevent Numerical Explosion (Critical Check 2)
            if denominator < 0.01: # Don't allow denominator to be near-zero
                print(f"Warning: GVF approaching critical depth at node {node_idx} (Fr={Fr:.2f}). Clamping slope.")
                denominator = 0.01 
                
            # Slopes
            # S0 is the bed slope from the target node (i) to the known node (i+1)
            S0 = (self.bed_level_at(i) - self.bed_level_at(i+1)) / dx # (z_up - z_down) / dx
            Sf = self.Se(h=h_in, Q=Q, i=node_idx)
            
            return (S0 - Sf) / denominator

        # Iterate from downstream (n-1) to upstream (0)
        for i in reversed(range(n_nodes-1)):
            
            h_down = h # This is the known depth h_i+1
            
            # --- 1. PREDICTOR Step ---
            # Calculate slope at the known downstream node (i+1)
            dh_dx_down = get_dh_dx(h_down, i+1)       
            # Predict the depth at the upstream node (i)
            h_pred = h_down - dh_dx_down * dx        
            
            if h_pred <= 0:
                h_pred = 0.01 # Prevent negative depth

            # --- 2. CORRECTOR Step ---
            # Calculate slope at the upstream node (i) using the predicted depth
            dh_dx_pred = get_dh_dx(h_pred, i)        
            
            # --- 3. Average Slopes and Final Step ---
            dh_dx_avg = 0.5 * (dh_dx_down + dh_dx_pred)
            h_up = h_down - dh_dx_avg * dx           # This is the final, corrected h_i
            
            if h_up <= 0:
                print(f"Warning: GVF calculation resulted in h <= 0 at node {i}. Setting to 0.01.")
                h_up = 0.01
                
            # --- 4. Save and set up for next iteration ---
            h = h_up # The new upstream depth becomes the known downstream depth for the next loop
            self.initial_conditions[i, 0] = h
            self.initial_conditions[i, 1] = Q
            
    def _linear_conditions(self, n_nodes, Q):
        h0 = self.upstream_boundary.initial_depth
        hN = self.downstream_boundary.initial_depth
                    
        for i in range(n_nodes):
            distance = self.length * i / (n_nodes-1)
                
            h = h0 + (hN - h0) * distance / self.length
                
            self.initial_conditions[i, 0] = h
            self.initial_conditions[i, 1] = Q