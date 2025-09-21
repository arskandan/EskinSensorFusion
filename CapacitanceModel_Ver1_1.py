import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Tuple, Dict, Optional, Union, Any
import warnings
import json
import os
import time
warnings.filterwarnings('ignore')
def calculate_euclidean_distances(point: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distances from a single point to multiple points using numpy only.
    
    This replaces scipy.spatial.distance.cdist to avoid import hanging issues.
    
    Args:
        point: Single point as 1D numpy array
        points: Multiple points as 2D numpy array (n_points, n_dimensions)
        
    Returns:
        1D numpy array of distances
    """
    return np.sqrt(np.sum((points - point)**2, axis=1))
class CapacitiveModel:
    """
    Physics-based model for capacitive sensing using parallel plate capacitor theory.
    
    This class models a 4-electrode capacitive touch sensor with the following specifications:
    - Sensor area: 16x16 cm
    - 4 electrodes positioned at corners with 2cm margin
    - Stylus: 9mm diameter (contact area = 0.636 cm²)
    
    The model calculates capacitance changes when a conductive stylus approaches
    the sensor surface, accounting for proximity effects and sensor coupling.
    """
    
    def __init__(self, 
                 skin_size: Tuple[float, float] = (16.0, 16.0),
                 electrode_size: Tuple[float, float] = (5.0, 5.0),
                 margin: float = 2.0,
                 epsilon_0: float = 8.854e-12,  # Vacuum permittivity
                 epsilon_r: float = 1.2,        # Relative permittivity
                 baseline_distance: float = 2.089,  # Stylus height (cm)
                 stylus_area: float = 0.65,   # Stylus contact area (cm²)
                 decay_constant: float = 2.0,   # Proximity decay factor
                 use_tuned_parameters: bool = False,
                 tuned_params_file: str = "tuned_model_parameters.json",
                 channel_poly_coeffs: Optional[dict] = None,
                 poly2d_features: Optional[Any] = None,
                 poly2d_coef: Optional[np.ndarray] = None,
                 poly2d_intercept: Optional[float] = None,
                 channel_poly2d_fits: Optional[dict] = None):
        """
        Initialize the capacitive model with physical and electrical parameters.
        
        Args:
            skin_size: Sensor dimensions (width, height) in cm
            electrode_size: Individual electrode size in cm
            margin: Distance from sensor edge to electrode in cm
            epsilon_0: Vacuum permittivity (F/m)
            epsilon_r: Relative permittivity of medium
            baseline_distance: Default stylus distance from surface (cm)
            stylus_area: Stylus contact area (cm²)
            decay_constant: Controls proximity effect decay rate
            use_tuned_parameters: Whether to load optimized parameters from file
            tuned_params_file: Path to JSON file with tuned parameters
            channel_poly_coeffs: Dictionary of per-channel cubic polynomial coefficients for proximity factor
        """
        # Store geometric parameters
        self.skin_size = skin_size
        self.electrode_size = electrode_size
        self.margin = margin
        self.epsilon_0 = epsilon_0
        # Per-channel polynomial coefficients for proximity factor (dict: channel -> [a, b, c, d])
        self.channel_poly_coeffs = channel_poly_coeffs
        # 2D polynomial fit objects (if provided, single/global)
        self.poly2d_features = poly2d_features
        self.poly2d_coef = poly2d_coef
        self.poly2d_intercept = poly2d_intercept
        # Per-channel 2D polynomial fit objects (dict: channel -> {'features', 'coef', 'intercept'})
        self.channel_poly2d_fits = channel_poly2d_fits
        # Load parameters (tuned or default)
        if use_tuned_parameters and os.path.exists(tuned_params_file):
            self.load_tuned_parameters(tuned_params_file)
        else:
            self.epsilon_r = epsilon_r
            self.baseline_distance = baseline_distance
            self.stylus_area = stylus_area
            self.decay_constant = decay_constant
        # Calculate electrode positions based on geometry
        self._calculate_electrode_positions()

        # Load fitted coupling coefficients (alpha_ij) from CSV
        self.coupling_alpha = None
        alpha_path = 'coupling_alpha_matrix.csv'
        if os.path.exists(alpha_path):
            try:
                alpha_df = pd.read_csv(alpha_path, index_col=0)
                self.coupling_alpha = alpha_df.values
                self.coupling_channels = list(alpha_df.columns)
                print(f"Loaded coupling coefficients from {alpha_path}")
            except Exception as e:
                print(f"Warning: Could not load coupling coefficients: {e}")
                self.coupling_alpha = None
        else:
            print(f"Warning: Coupling coefficient file '{alpha_path}' not found. No coupling will be applied.")
    
    def load_tuned_parameters(self, filename: str):
        """
        Load optimized parameters from a JSON file containing tuning results.
        
        Args:
            filename: Path to JSON file with optimized parameters
        """
        try:
            with open(filename, 'r') as f:
                params = json.load(f)
            
            # Load core parameters
            self.epsilon_r = params['epsilon_r']
            self.baseline_distance = params['baseline_distance']
            self.stylus_area = params.get('stylus_area', 0.6362)
            self.decay_constant = params.get('decay_constant', 0.6640)
            
            # Load geometry parameters if available
            if 'skin_size' in params:
                self.skin_size = tuple(params['skin_size'])
            if 'electrode_size' in params:
                self.electrode_size = tuple(params['electrode_size'])
            if 'margin' in params:
                self.margin = params['margin']
            if 'epsilon_0' in params:
                self.epsilon_0 = params['epsilon_0']
                
            print(f"Loaded tuned parameters from {filename}")
            
        except Exception as e:
            print(f"Warning: Could not load tuned parameters from {filename}: {e}")
            print("Using default parameters instead.")
            self.stylus_area = 0.6362
            self.decay_constant = 0.6640
        
    def _calculate_electrode_positions(self):
        """
        Calculate the center positions of the 4 electrodes based on sensor geometry.
        
        Electrodes are positioned at the four corners with specified margins:
        - CH0: Bottom-right electrode
        - CH1: Bottom-left electrode  
        - CH2: Top-left electrode
        - CH3: Top-right electrode
        """
        electrode_x_left = self.margin + self.electrode_size[0] / 2
        electrode_x_right = self.skin_size[0] - self.margin - self.electrode_size[0] / 2
        electrode_y_bottom = self.margin + self.electrode_size[1] / 2
        electrode_y_top = self.skin_size[1] - self.margin - self.electrode_size[1] / 2
        
        self.electrode_positions = {
            'CH0': (electrode_x_right, electrode_y_bottom),   # Bottom-right
            'CH1': (electrode_x_left, electrode_y_bottom),    # Bottom-left
            'CH2': (electrode_x_left, electrode_y_top),       # Top-left
            'CH3': (electrode_x_right, electrode_y_top)       # Top-right
        }
    
    def calculate_capacitance(self,
                            stylus_pos: Tuple[float, float],
                            stylus_distance: Optional[float] = None,
                            stylus_area: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate capacitance changes for all sensors when stylus is at given position.
        
        Physics Implementation:
        1. Calculate individual electrode responses using parallel plate capacitor theory
        2. Apply coupling matrix: C_measured = A @ C_ideal
           - This models that each electrode measures combined effects from all coupling paths
           - The stylus creates simultaneous capacitive coupling to all electrodes
           - Matrix A represents learned inter-electrode coupling coefficients
        
        Always applies learned coupling coefficients from real sensor data if available.
        """

        # Ensure stylus_distance and stylus_area are set
        if stylus_distance is None:
            stylus_distance = self.baseline_distance
        if stylus_area is None or stylus_area is False:
            stylus_area = self.stylus_area

        x_stylus, y_stylus = stylus_pos

        # Calculate ideal (no coupling) responses
        ideal_responses = {}
        for channel, (x_electrode, y_electrode) in self.electrode_positions.items():
            # Ensure stylus_distance and stylus_area are set for each calculation
            sd = stylus_distance if stylus_distance is not None else self.baseline_distance
            sa = stylus_area if stylus_area is not None else self.stylus_area
            dx = x_stylus - x_electrode
            dy = y_stylus - y_electrode
            lateral_distance = np.sqrt(dx**2 + dy**2)
            total_distance = np.sqrt(lateral_distance**2 + sd**2)
            
            # Prevent division by zero with minimum distance threshold
            min_distance = 0.01  # 0.01 cm = 0.1 mm minimum distance
            total_distance = max(total_distance, min_distance)
            
            distance_m = total_distance / 100  # cm to m
            stylus_area_m2 = sa / 10000  # cm² to m²
            proximity_factor = self._calculate_proximity_factor(dx, dy, channel=channel, x_abs=x_stylus, y_abs=y_stylus)
            base_capacitance = (self.epsilon_0 * self.epsilon_r * stylus_area_m2 * proximity_factor) / distance_m
            ideal_responses[channel] = base_capacitance * 1e12  # Convert to pF

        # Apply learned coupling correction if available
        if self.coupling_alpha is not None:
            # Map channel order to match the coupling matrix
            ch_order = ['CH0', 'CH1', 'CH2', 'CH3']
            ideal_vec = np.array([ideal_responses[ch] for ch in ch_order])
            
            # Apply coupling matrix: C_measured = A @ C_ideal
            # This correctly models that each electrode measures the combined effect
            # of its direct capacitance plus cross-coupling from other electrodes
            measured_vec = self.coupling_alpha @ ideal_vec
            
            # Return as dict with minimum threshold to avoid negative values
            return {ch: float(max(measured_vec[idx], 1e-8)) for idx, ch in enumerate(ch_order)}
        else:
            # Fallback: return ideal responses
            return {ch: float(max(val, 1e-8)) for ch, val in ideal_responses.items()}
    
    def _calculate_proximity_factor(self, dx: float, dy: float, channel: Optional[str] = None, x_abs: Optional[float] = None, y_abs: Optional[float] = None) -> float:
        """
        Proximity factor using corrected 2D polynomial fit if available, else per-channel 1D fit, else fallback to default.
        
        The corrected 2D polynomial fits model normalized proximity factors (0-1) instead of raw capacitance values.
        This fixes the physics issues where the original fits were saturated at ~69pF.
        
        Args:
            dx: x distance between stylus and electrode center (cm)
            dy: y distance between stylus and electrode center (cm)
            channel: channel name (e.g., 'CH0')
            x_abs, y_abs: absolute X, Y position (cm) if available
        Returns:
            Proximity factor (0-1) representing coupling strength
        """
        # Use per-channel 2D polynomial fit if available
        if (
            self.channel_poly2d_fits is not None and channel is not None and x_abs is not None and y_abs is not None
            and channel in self.channel_poly2d_fits
        ):
            fit = self.channel_poly2d_fits[channel]
            features = fit.get('features', None)
            coef = fit.get('coef', None)
            intercept = fit.get('intercept', None)
            if features is not None and coef is not None and intercept is not None:
                XY = np.array([[x_abs, y_abs]])
                XY_poly = features.transform(XY)
                pf_raw = float(np.dot(XY_poly.flatten(), coef.flatten()) + intercept)
                pf_clipped = float(np.clip(pf_raw, 0.0, 1.0))
                return pf_clipped
        # Use global 2D polynomial fit if provided
        if self.poly2d_features is not None and self.poly2d_coef is not None and self.poly2d_intercept is not None and x_abs is not None and y_abs is not None:
            XY = np.array([[x_abs, y_abs]])
            XY_poly = self.poly2d_features.transform(XY)
            pf = float(np.dot(XY_poly, self.poly2d_coef) + self.poly2d_intercept)
            return float(np.clip(pf, 0.0, 1.0))
        # Use per-channel 1D polynomial fit if available
        if self.channel_poly_coeffs is not None and channel in self.channel_poly_coeffs:
            r = np.sqrt(dx**2 + dy**2)
            a, b, c, d = self.channel_poly_coeffs[channel]
            pf = a * r**3 + b * r**2 + c * r + d
            return float(np.clip(pf, 0.0, 1.0))
        # Fallback: original box+exponential model
        datasheet_area = 20.9 * 13.9  # mm^2
        my_area = 50.0 * 50.0         # mm^2
        area_scale = my_area / datasheet_area

        dx_abs = abs(dx)
        dy_abs = abs(dy)
        # Reduce plateau to ~3x3 cm (so half-width is 1.5 cm)
        plateau_half_width_x = self.electrode_size[0] / 2
        plateau_half_width_y = self.electrode_size[1] / 2

        if dx_abs <= plateau_half_width_x and dy_abs <= plateau_half_width_y:
            # Uniform sensitivity inside the reduced plateau area
            proximity_factor = 1.0
        else:
            # Rapid exponential decay just outside the edge
            dx_out = max(dx_abs - plateau_half_width_x, 0)
            dy_out = max(dy_abs - plateau_half_width_y, 0)
            dist_out = np.sqrt(dx_out**2 + dy_out**2)
            scaled_decay = self.decay_constant * area_scale
            proximity_factor = np.exp(-dist_out * 5 / scaled_decay)  # convert cm to mm
            proximity_factor = min(max(proximity_factor, 0.0), 1.0)
        return proximity_factor
    
    def _apply_sensor_coupling(self, ideal_responses: Dict[str, float]) -> Dict[str, float]:
        """
        Apply learned coupling matrix to ideal capacitance responses.
        
        Physics: C_measured = A @ C_ideal
        
        Where A is the coupling matrix that represents how each electrode's 
        measurement is affected by the capacitive fields from all electrodes.
        This models the fact that when a stylus approaches, it creates 
        simultaneous coupling to all electrodes.
        """
        if self.coupling_alpha is not None:
            ch_order = ['CH0', 'CH1', 'CH2', 'CH3']
            ideal_vec = np.array([ideal_responses[ch] for ch in ch_order])
            measured_vec = self.coupling_alpha @ ideal_vec
            return {ch: float(max(measured_vec[idx], 1e-8)) for idx, ch in enumerate(ch_order)}
        else:
            # Fallback: return ideal responses
            return {ch: float(max(val, 1e-8)) for ch, val in ideal_responses.items()}
    
    
    def set_material_properties(self, epsilon_r: float, baseline_distance: Optional[float] = None):
        """
        Update material properties for different environmental conditions.
        
        Args:
            epsilon_r: New relative permittivity value
            baseline_distance: New baseline distance in cm (optional)
        """
        self.epsilon_r = epsilon_r
        if baseline_distance is not None:
            self.baseline_distance = baseline_distance
class LookupTable:
    """
    Generates and manages lookup table for capacitance-to-position mapping.
    
    This class creates a comprehensive lookup table by calculating capacitance
    values for a grid of positions across the sensor surface. This table is
    used for reverse mapping from measured capacitances to estimated positions.
    """
    
    def __init__(self, model: CapacitiveModel, resolution: float = 0.1):
        """
        Initialize lookup table generator.
        
        Args:
            model: CapacitiveModel instance for calculations
            resolution: Grid spacing in cm for lookup table generation
        """
        self.model = model
        self.resolution = resolution
        self.lookup_data = None  # Will store pandas DataFrame
        self.positions = None    # Will store numpy array of positions
        
    def generate_lookup_table(self, 
                            x_range: Optional[Tuple[float, float]] = None,
                            y_range: Optional[Tuple[float, float]] = None,
                            stylus_distance: Optional[float] = None) -> pd.DataFrame:
        """
        Generate lookup table for all positions within the sensor area.
        
        Creates a grid of positions and calculates corresponding capacitance
        values for each electrode. This forms the basis for position estimation.
        
        IMPORTANT: For touch position estimation, use stylus_distance=0 to model
        capacitance when the stylus is in contact with the surface, not the 
        baseline distance (which represents stylus hovering above surface).
        
        Args:
            x_range: (min, max) x coordinates in cm. Defaults to full sensor width
            y_range: (min, max) y coordinates in cm. Defaults to full sensor height  
            stylus_distance: Distance from sensor surface in cm. Use 0 for touch contact.
            
        Returns:
            DataFrame with columns: x, y, CH0, CH1, CH2, CH3
        """
        # Use full sensor area if ranges not specified
        if x_range is None:
            x_range = (0, self.model.skin_size[0])
        if y_range is None:
            y_range = (0, self.model.skin_size[1])
            
        # Generate position grids
        x_positions = np.arange(x_range[0], x_range[1] + self.resolution, self.resolution)
        y_positions = np.arange(y_range[0], y_range[1] + self.resolution, self.resolution)
        
        total_points = len(x_positions) * len(y_positions)
        print(f"Generating lookup table: {len(x_positions)}x{len(y_positions)} = {total_points} points")
        
        data = []
        positions = []
        
        # Calculate capacitances for each position
        for i, x in enumerate(x_positions):
            if i % 2 == 0:  # Progress updates
                progress = (i + 1) / len(x_positions) * 100
                print(f"Processing x={x:.1f} ({i+1}/{len(x_positions)}) - {progress:.1f}% complete")
            for y in y_positions:
                pos = (float(x), float(y))
                capacitances = self.model.calculate_capacitance(pos, stylus_distance)
                # Create row with position and capacitance data
                row = {'x': x, 'y': y}
                for ch, cap_val in capacitances.items():
                    row[ch] = np.float64(cap_val)
                data.append(row)
                positions.append(pos)
        
        # Store results
        self.lookup_data = pd.DataFrame(data)
        self.positions = np.array(positions)
        
        print(f"Generated lookup table with {len(self.lookup_data)} entries")
        return self.lookup_data
    
    def save_lookup_table(self, filename: str):
        """
        Save lookup table to CSV file.
        
        Args:
            filename: Path for output CSV file
        """
        if self.lookup_data is not None:
            self.lookup_data.to_csv(filename, index=False)
            print(f"Lookup table saved to {filename}")
        else:
            print("No lookup table data to save. Generate table first.")
        
    def load_lookup_table(self, filename: str):
        """
        Load lookup table from CSV file.
        
        Args:
            filename: Path to CSV file containing lookup table
        """
        try:
            self.lookup_data = pd.read_csv(filename)
            self.positions = self.lookup_data[['x', 'y']].values
            print(f"Lookup table loaded from {filename}")
        except Exception as e:
            print(f"Error loading lookup table: {e}")
class PositionEstimator:
    """Estimates stylus position from capacitance measurements using multiple methods with baseline correction."""
    
    def __init__(self, lookup_table: LookupTable):
        self.lookup_table = lookup_table
        self.baseline = None  # Store baseline for consistent corrections
        
    def set_baseline(self, baseline: Dict[str, float]):
        """Set the baseline values for future measurements."""
        self.baseline = baseline
        print("Baseline set for position estimation:")
        for ch, val in baseline.items():
            print(f"  {ch}: {val:.6f}")
    
    def apply_baseline_correction_to_measurement(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """Apply baseline correction to a single measurement."""
        if self.baseline is None:
            print("Warning: No baseline set. Using raw measurements.")
            return measurements
        
        corrected = {}
        for ch in ['CH0', 'CH1', 'CH2', 'CH3']:
            if ch in measurements and ch in self.baseline:
                corrected[ch] = measurements[ch] - self.baseline[ch]
            else:
                corrected[ch] = measurements.get(ch, 0.0)
        
        return corrected
        
    def estimate_position_nearest_neighbor(self, measurements: Dict[str, float], 
                                          apply_baseline_correction: bool = True) -> Tuple[float, float, float]:
        """Estimate position using nearest neighbor in capacitance space with baseline correction."""
        if self.lookup_table.lookup_data is None or self.lookup_table.positions is None:
            raise ValueError("No lookup table available. Generate one first.")
        
        # Apply baseline correction if requested and baseline is available
        if apply_baseline_correction:
            measurements = self.apply_baseline_correction_to_measurement(measurements)
        
        channels = ['CH0', 'CH1', 'CH2', 'CH3']
        measurement_vector = np.array([measurements[ch] for ch in channels])
        lookup_vectors = self.lookup_table.lookup_data[channels].values
        
        distances = calculate_euclidean_distances(measurement_vector, lookup_vectors)
        best_idx = np.argmin(distances)
        best_position = self.lookup_table.positions[best_idx]
        
        return float(best_position[0]), float(best_position[1]), float(distances[best_idx])
    
    def estimate_position_weighted(self, measurements: Dict[str, float], k: int = 5, 
                                 apply_baseline_correction: bool = True) -> Tuple[float, float, float]:
        """Estimate position using weighted average of k nearest neighbors with baseline correction."""
        if self.lookup_table.lookup_data is None or self.lookup_table.positions is None:
            raise ValueError("No lookup table available. Generate one first.")
        
        # Apply baseline correction if requested and baseline is available
        if apply_baseline_correction:
            measurements = self.apply_baseline_correction_to_measurement(measurements)
        
        channels = ['CH0', 'CH1', 'CH2', 'CH3']
        measurement_vector = np.array([measurements[ch] for ch in channels])
        lookup_vectors = self.lookup_table.lookup_data[channels].values
        
        distances = calculate_euclidean_distances(measurement_vector, lookup_vectors)
        nearest_indices = np.argpartition(distances, k)[:k]
        nearest_distances = distances[nearest_indices]
        nearest_positions = self.lookup_table.positions[nearest_indices]
        
        weights = 1.0 / (nearest_distances + 1e-10)
        weights = weights / np.sum(weights)
        
        estimated_x = np.sum(weights * nearest_positions[:, 0])
        estimated_y = np.sum(weights * nearest_positions[:, 1])
        
        return float(estimated_x), float(estimated_y), float(np.mean(nearest_distances))
    
    def estimate_position_gradient_based(self, measurements: Dict[str, float], 
                                       apply_baseline_correction: bool = True) -> Tuple[float, float, float]:
        """Estimate position using gradient-based approach for center positions."""
        if apply_baseline_correction:
            measurements = self.apply_baseline_correction_to_measurement(measurements)
        
        # Get sensor positions
        sensor_positions = np.array([
            [11.5, 4.5],   # CH0
            [4.5, 4.5],    # CH1
            [4.5, 11.5],   # CH2
            [11.5, 11.5]   # CH3
        ])
        
        # Get measurement values
        channels = ['CH0', 'CH1', 'CH2', 'CH3']
        values = np.array([abs(measurements[ch]) for ch in channels])
        
        # Normalize values to avoid division by zero
        values = values + 1e-10
        
        # Calculate weighted centroid
        weights = values / np.sum(values)
        estimated_x = np.sum(weights * sensor_positions[:, 0])
        estimated_y = np.sum(weights * sensor_positions[:, 1])
        
        # Calculate confidence based on signal distribution
        signal_std = np.std(values)
        signal_mean = np.mean(values)
        confidence = float(signal_std / (signal_mean + 1e-10))
        
        return float(estimated_x), float(estimated_y), confidence
    
    def detect_center_position(self, measurements: Dict[str, float]) -> bool:
        """Detect if the touch is near the center where all sensors are equidistant."""
        channels = ['CH0', 'CH1', 'CH2', 'CH3']
        values = [abs(measurements[ch]) for ch in channels]
        
        # Check if all values are similar (indicating center position)
        values_array = np.array(values)
        relative_std = np.std(values_array) / (np.mean(values_array) + 1e-10)
        
        # If relative standard deviation is low, it's likely a center position
        return bool(relative_std < 0.3)
    def compare_estimation_methods(self, measurements: Dict[str, float], k: int = 5, 
                                 apply_baseline_correction: bool = True) -> Dict[str, Tuple[float, float, float]]:
        """Enhanced estimation with center-aware methods.
        
        Args:
            measurements: Dictionary with capacitance measurements
            k: Number of neighbors for weighted method
            apply_baseline_correction: Whether to apply baseline correction to measurements
            
        Returns:
            Dictionary with results from each method
        """
        results = {}
        
        # Nearest neighbor method
        try:
            results['nearest_neighbor'] = self.estimate_position_nearest_neighbor(measurements, apply_baseline_correction)
        except Exception as e:
            results['nearest_neighbor'] = (None, None, f"Error: {e}")
        
        # Weighted k-NN method
        try:
            results['weighted_knn'] = self.estimate_position_weighted(measurements, k, apply_baseline_correction)
        except Exception as e:
            results['weighted_knn'] = (None, None, f"Error: {e}")
        
        # Enhanced gradient-based method for center positions
        try:
            results['gradient_based'] = self.estimate_position_gradient_based(measurements, apply_baseline_correction)
        except Exception as e:
            results['gradient_based'] = (None, None, f"Error: {e}")
        
        # Select best method based on position characteristics
        is_center = self.detect_center_position(measurements if not apply_baseline_correction 
                                              else self.apply_baseline_correction_to_measurement(measurements))
        
        if is_center:
            # For center positions, prefer gradient-based method
            if 'gradient_based' in results and results['gradient_based'][0] is not None:
                results['best_estimate'] = results['gradient_based']
            elif 'weighted_knn' in results and results['weighted_knn'][0] is not None:
                results['best_estimate'] = results['weighted_knn']
            else:
                results['best_estimate'] = results.get('nearest_neighbor', (None, None, "No valid estimate"))
        else:
            # For non-center positions, prefer standard methods
            if 'weighted_knn' in results and results['weighted_knn'][0] is not None:
                results['best_estimate'] = results['weighted_knn']
            else:
                results['best_estimate'] = results.get('nearest_neighbor', (None, None, "No valid estimate"))
        
        return results
    
    def estimate_positions_from_csv(self, csv_filename: str, baseline_samples: int = 5, 
                                  skip_baseline_rows: bool = True) -> pd.DataFrame:
        """
        Estimate positions for all data points in a CSV file using baseline correction.
        
        Args:
            csv_filename: Path to CSV file with sensor data
            baseline_samples: Number of initial samples to use for baseline
            skip_baseline_rows: Whether to skip the baseline rows in estimation (default: True)
        
        Returns:
            DataFrame with original data plus estimated positions
        """
        # Load and correct data
        corrected_data, baseline = DataHandler.load_and_correct_csv_data(csv_filename, baseline_samples)
        if corrected_data is None or baseline is None:
            raise ValueError("Failed to load or correct sensor data")
        
        # Set baseline for estimation
        self.set_baseline(baseline)
        
        # Prepare results
        results = []
        channels = ['CH0', 'CH1', 'CH2', 'CH3']
        
        # Determine which rows to process
        start_row = baseline_samples if skip_baseline_rows else 0
        
        print(f"Estimating positions for {len(corrected_data) - start_row} data points...")
        
        for idx in range(start_row, len(corrected_data)):
            row_data = corrected_data.iloc[idx]
            
            # Extract measurements (already baseline-corrected)
            measurements = {}
            for ch in channels:
                if ch in row_data:
                    measurements[ch] = float(row_data[ch])
                else:
                    measurements[ch] = 0.0
            
            # Estimate position (don't apply baseline correction again since data is already corrected)
            try:
                estimates = self.compare_estimation_methods(measurements, apply_baseline_correction=False)
            except Exception as e:
                print(f"Error estimating position for row {idx}: {e}")
                estimates = {'nearest_neighbor': (None, None, "Error"), 
                           'weighted_knn': (None, None, "Error")}
            
            # Store results
            result: Dict[str, Any] = {'row_index': idx}
            
            # Add original measurements
            for ch in channels:
                result[f'{ch}_corrected'] = measurements[ch]
                result[f'{ch}_original'] = row_data[ch] + baseline[ch]  # Add back baseline for reference
            
            # Add estimates
            for method, (x, y, score) in estimates.items():
                result[f'{method}_x'] = float(x) if x is not None else None
                result[f'{method}_y'] = float(y) if y is not None else None
                result[f'{method}_score'] = score  # Can be float or string (error message)
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        print(f"Position estimation completed for {len(results_df)} data points")
        
        return results_df
        
class Visualizer:
    def plot_individual_parallel_plate_heatmaps(self, lookup_table: LookupTable, figsize: Tuple[int, int] = (16, 12)):
        """
        Plot 4 separate heatmaps for each channel using only the parallel plate formula (no mutual coupling).
        """
        if lookup_table.lookup_data is None:
            raise ValueError("No lookup table data available.")

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        layout_mapping = [
            (0, 0, 'CH2', 'Top Left'),
            (0, 1, 'CH3', 'Top Right'),
            (1, 0, 'CH1', 'Bottom Left'),
            (1, 1, 'CH0', 'Bottom Right')
        ]

        x_unique = sorted(lookup_table.lookup_data['x'].unique())
        y_unique = sorted(lookup_table.lookup_data['y'].unique())

        for row, col, channel, title in layout_mapping:
            ax = axes[row, col]
            Z = np.zeros((len(y_unique), len(x_unique)))
            for j, y in enumerate(y_unique):
                for k, x in enumerate(x_unique):
                    # Calculate parallel plate capacitance for this channel at (x, y)
                    x_electrode, y_electrode = self.model.electrode_positions[channel]
                    dx = x - x_electrode
                    dy = y - y_electrode
                    stylus_distance = self.model.baseline_distance
                    stylus_area = self.model.stylus_area
                    # Proximity factor (box + decay)
                    pf = self.model._calculate_proximity_factor(dx, dy)
                    d = stylus_distance / 100  # cm to m
                    a = stylus_area / 10000    # cm^2 to m^2
                    c = (self.model.epsilon_0 * self.model.epsilon_r * a * pf) / d * 1e12  # pF
                    Z[j, k] = float(c)
            im = ax.imshow(Z, extent=[0, 16, 0, 16], origin='lower', cmap='viridis', aspect='equal')
            ax.set_title(f'{channel} (Parallel Plate Only)', fontsize=14, fontweight='bold')
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Y (cm)')
            ax.set_xlim(0, 16)
            ax.set_ylim(0, 16)
            plt.colorbar(im, ax=ax)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, axes
    """Visualization tools for the capacitive sensing system."""
    
    def __init__(self, model: CapacitiveModel):
        self.model = model
        
    def plot_capacitance_heatmaps_physical_layout(self, lookup_table: LookupTable, figsize: Tuple[int, int] = (16, 12)):
        """Plot capacitance heatmaps arranged to match the physical sensor layout."""
        if lookup_table.lookup_data is None:
            raise ValueError("No lookup table data available.")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        layout_mapping = [
            (0, 0, 'CH2', 'Top Left'),      # CH2: Top Left electrode
            (0, 1, 'CH3', 'Top Right'),     # CH3: Top Right electrode  
            (1, 0, 'CH1', 'Bottom Left'),   # CH1: Bottom Left electrode
            (1, 1, 'CH0', 'Bottom Right')   # CH0: Bottom Right electrode
        ]
        
        x_unique = sorted(lookup_table.lookup_data['x'].unique())
        y_unique = sorted(lookup_table.lookup_data['y'].unique())
        
        for row, col, channel, title in layout_mapping:
            ax = axes[row, col]
            Z = np.zeros((len(y_unique), len(x_unique)))
            
            for j, y in enumerate(y_unique):
                for k, x in enumerate(x_unique):
                    mask = (lookup_table.lookup_data['x'] == x) & (lookup_table.lookup_data['y'] == y)
                    if mask.any():
                        value = lookup_table.lookup_data.loc[mask, channel].values[0]  # type: ignore
                        Z[j, k] = float(value)
            
            im = ax.imshow(Z, extent=[0, 16, 0, 16], origin='lower', cmap='viridis', aspect='equal')
            ax.set_title(f'{channel}', fontsize=14, fontweight='bold')
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Y (cm)')
            ax.set_xlim(0, 16)
            ax.set_ylim(0, 16)
            
            plt.colorbar(im, ax=ax)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    
    def plot_combined_electrode_heatmap(self, lookup_table: LookupTable, figsize: Tuple[int, int] = (8, 6)):
        """
        Plot a heatmap showing the combined response when all sensors are working simultaneously.
        The combined response is the sum of all four channels at each position.
        """
        if lookup_table.lookup_data is None:
            raise ValueError("No lookup table data available.")

        x_unique = sorted(lookup_table.lookup_data['x'].unique())
        y_unique = sorted(lookup_table.lookup_data['y'].unique())
        Z = np.zeros((len(y_unique), len(x_unique)))

        for j, y in enumerate(y_unique):
            for k, x in enumerate(x_unique):
                mask = (lookup_table.lookup_data['x'] == x) & (lookup_table.lookup_data['y'] == y)
                if mask.any():
                    row = lookup_table.lookup_data.loc[mask]
                    value = row[['CH0', 'CH1', 'CH2', 'CH3']].sum(axis=1).values[0]
                    Z[j, k] = float(value)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(Z, extent=(0, 16, 0, 16), origin='lower', cmap='viridis', aspect='equal')
        ax.set_title('Combined Electrode Response', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 16)
        plt.colorbar(im, ax=ax)
        ax.grid(True, alpha=0.3)

    
        overlap_w, overlap_h = 5.0, 5.0
        for (ex, ey) in self.model.electrode_positions.values():
            rect = mpatches.Rectangle((ex - overlap_w/2, ey - overlap_h/2), overlap_w, overlap_h,
                                     linewidth=1.5, edgecolor='w', linestyle=':', facecolor='none', alpha=0.8)
            ax.add_patch(rect)

        plt.tight_layout()
        return fig, ax

    @staticmethod
    def generate_fine_mesh_heatmap(resolution: float = 0.1, save_data: bool = True):
        """
        Generate high-resolution heatmap with fine mesh for detailed visualization.
        
        Args:
            resolution: Grid spacing in cm (smaller = finer mesh, more detail)
            save_data: Whether to save the lookup table data to CSV
        
        Returns:
            Tuple of (model, lookup_table) for further analysis
        """
        print(f"=== Generating Fine Mesh Heatmap (Resolution: {resolution} cm) ===")
        start_time = time.time()
        
        # Create model and fine-resolution lookup table using tuned parameters
        print("1. Initializing capacitive model with tuned parameters...")
        model = CapacitiveModel(use_tuned_parameters=True, tuned_params_file="tuned_model_parameters.json")
        
        print(f"2. Creating fine mesh lookup table (resolution: {resolution} cm)...")
        lookup = LookupTable(model, resolution=resolution)
        
        # Calculate expected grid size
        grid_x = int((model.skin_size[0] / resolution) + 1)
        grid_y = int((model.skin_size[1] / resolution) + 1)
        total_points = grid_x * grid_y
        print(f"   Expected grid: {grid_x}x{grid_y} = {total_points} points")
        
        # Generate fine mesh data
        step_start = time.time()
        lookup_data = lookup.generate_lookup_table()
        generation_time = time.time() - step_start
        print(f"   Fine mesh generated in {generation_time:.2f} seconds")
        
        # Save fine mesh data if requested
        if save_data:
            filename = f"fine_mesh_capacitance_lookup_{resolution:.1f}cm.csv"
            lookup.save_lookup_table(filename)
        
        # Generate high-resolution visualization
        print("3. Creating high-resolution heatmap visualization...")
        step_start = time.time()
        visualizer = Visualizer(model)
        
        # Create figure with higher DPI for better quality
        fig, axes = visualizer.plot_capacitance_heatmaps_physical_layout(lookup, figsize=(20, 15))
        
        # Save high-resolution heatmap
        output_file = f'fine_mesh_heatmaps_{resolution:.1f}cm_resolution.png'
        plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"   High-resolution heatmap saved as '{output_file}'")
        
        # Also save as PDF for vector graphics
        pdf_file = f'fine_mesh_heatmaps_{resolution:.1f}cm_resolution.pdf'
        plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
        print(f"   Vector heatmap saved as '{pdf_file}'")
        
        plt.close(fig)
        
        total_time = time.time() - start_time
        print(f"Fine mesh heatmap generation completed in {total_time:.2f} seconds")
        print(f"Grid resolution: {grid_x}x{grid_y} points")
        print(f"Total data points: {len(lookup_data)} entries")
        
        return model, lookup
class DataHandler:
    """Handles loading and processing of real sensor data with baseline correction."""
    
    @staticmethod
    def load_csv_data(filename: str) -> Union[pd.DataFrame, None]:
        """Load FDC2214 data from CSV file."""
        try:
            data = pd.read_csv(filename)
            
            # Map CSV column names to expected channel names
            column_mapping = {
                'DATA0_pF': 'CH0',
                'DATA1_pF': 'CH1', 
                'DATA2_pF': 'CH2',
                'DATA3_pF': 'CH3'
            }
            
            # Rename columns if they exist in the data
            columns_to_rename = {old_name: new_name for old_name, new_name in column_mapping.items() 
                               if old_name in data.columns}
            if columns_to_rename:
                data = data.rename(columns=columns_to_rename)
                print(f"Mapped columns: {columns_to_rename}")
            
            print(f"Loaded data with shape: {data.shape}")
            print(f"Available columns: {list(data.columns)}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    @staticmethod
    def compute_baseline(data: pd.DataFrame, baseline_samples: int = 20) -> Dict[str, float]:
        """
        Compute baseline values from the first N samples of each channel.
        
        Args:
            data: DataFrame with sensor data containing CH0, CH1, CH2, CH3 columns
            baseline_samples: Number of initial samples to use for baseline (default: 20)
        
        Returns:
            Dictionary with baseline values for each channel
        """
        channels = ['CH0', 'CH1', 'CH2', 'CH3']
        baseline = {}
        
        for channel in channels:
            if channel in data.columns:
                baseline_values = data[channel].head(baseline_samples)
                baseline[channel] = float(baseline_values.mean())
            else:
                print(f"Warning: Channel {channel} not found in data")
                baseline[channel] = 0.0
        
        print(f"Computed baseline from first {baseline_samples} samples:")
        for ch, val in baseline.items():
            print(f"  {ch}: {val:.6f}")
        
        return baseline
    
    @staticmethod
    def apply_baseline_correction(data: pd.DataFrame, baseline: Optional[Dict[str, float]] = None, 
                                baseline_samples: int = 20) -> pd.DataFrame:
        """
        Apply baseline correction to sensor data using deviations from baseline.
        
        Args:
            data: DataFrame with sensor data
            baseline: Pre-computed baseline values (if None, will compute from first samples)
            baseline_samples: Number of samples to use for baseline if baseline is None
        
        Returns:
            DataFrame with baseline-corrected data (deviations from baseline)
        """
        channels = ['CH0', 'CH1', 'CH2', 'CH3']
        corrected_data = data.copy()
        
        # Compute baseline if not provided
        if baseline is None:
            baseline = DataHandler.compute_baseline(data, baseline_samples)
        
        # Apply baseline correction (compute deviations)
        for channel in channels:
            if channel in corrected_data.columns:
                corrected_data[channel] = corrected_data[channel] - baseline[channel]
                print(f"Applied baseline correction to {channel}")
        
        print(f"Baseline correction applied. Data now represents deviations from baseline.")
        return corrected_data
    
    @staticmethod
    def load_and_correct_csv_data(filename: str, baseline_samples: int = 20) -> Tuple[Union[pd.DataFrame, None], Union[Dict[str, float], None]]:
        """
        Load CSV data and automatically apply baseline correction.
        
        Args:
            filename: Path to CSV file
            baseline_samples: Number of initial samples to use for baseline
        
        Returns:
            Tuple of (corrected_data, baseline_values)
        """
        try:
            data = DataHandler.load_csv_data(filename)
            if data is None:
                return None, None
            
            print(f"\nLoaded data from {filename}")
            print(f"Data shape: {data.shape}")
            
            # Compute and apply baseline correction
            baseline = DataHandler.compute_baseline(data, baseline_samples)
            corrected_data = DataHandler.apply_baseline_correction(data, baseline, baseline_samples)
            
            return corrected_data, baseline
            
        except Exception as e:
            print(f"Error loading and correcting data: {e}")
            return None, None
def main():
    """
    Main function demonstrating the complete capacitive localization system.
    
    This function:
    1. Creates a physics-based capacitive model
    2. Generates a lookup table for position mapping
    3. Demonstrates position estimation with real and theoretical data
    4. Creates visualization heatmaps
    """
    start_time = time.time()
    print("=== Capacitive Touch Localization System ===")
    
    # Step 1: Initialize the physics model with 2D proximity factor fit if available
    print("\n1. Initializing capacitive model...")
    step_start = time.time()

    # --- Load per-channel 2D polynomial fit objects ---
    import joblib
    channel_poly2d_fits = {}
    channels = ['CH0', 'CH1', 'CH2', 'CH3']
    
    print("Loading corrected 2D polynomial proximity factor fits...")
    for ch in channels:
        try:
            features = joblib.load(f"poly2d_features_{ch}.joblib")
            coef = joblib.load(f"poly2d_coef_{ch}.joblib")
            intercept = joblib.load(f"poly2d_intercept_{ch}.joblib")
            channel_poly2d_fits[ch] = {'features': features, 'coef': coef, 'intercept': intercept}
            print(f" Loaded corrected 2D polynomial fit for {ch} (intercept: {intercept:.4f})")
        except Exception as e:
            print(f" Could not load 2D fit for {ch}: {e}")
            channel_poly2d_fits[ch] = {'features': None, 'coef': None, 'intercept': None}

    # Print status of per-channel 2D fit objects after loading
    print("\n--- Corrected Per-Channel 2D Polynomial Fit Status ---")
    for ch in channels:
        fit = channel_poly2d_fits[ch]
        has_fit = all(fit[key] is not None for key in ['features', 'coef', 'intercept'])
        status = " LOADED" if has_fit else "❌ NOT LOADED"
        intercept_info = f" (intercept: {fit['intercept']:.4f})" if fit['intercept'] is not None else ""
        print(f"{ch}: {status}{intercept_info}")
    print("Note: Using corrected 2D polynomial fits that model proximity factors (0-1)")
    print("      instead of raw capacitance values (~69pF). This fixes physics issues.")
    print("--------------------------------------------------------\n")

    model = CapacitiveModel(
        use_tuned_parameters=True,
        tuned_params_file="tuned_model_parameters.json",
        channel_poly2d_fits=channel_poly2d_fits
    )
    print(f"   Model initialized in {time.time() - step_start:.2f} seconds")
    
    # Step 2: Create lookup table for position estimation
    print("\n2. Creating lookup table...")
    step_start = time.time()
    lookup = LookupTable(model, resolution=0.1)  
    # Generate lookup table with stylus_distance=0 for touch contact prediction
    lookup_data = lookup.generate_lookup_table(stylus_distance=2.0)
    print(f"   Lookup table created in {time.time() - step_start:.2f} seconds")
    
    # Step 3: Save lookup table for future use
    print("\n3. Saving lookup table...")
    step_start = time.time()
    lookup.save_lookup_table("capacitance_lookup_table.csv")
    print(f"   Lookup table saved in {time.time() - step_start:.2f} seconds")
    
    # Step 4: Create position estimator
    print("\n4. Creating position estimator...")
    step_start = time.time()
    estimator = PositionEstimator(lookup)
    print(f"   Estimator created in {time.time() - step_start:.2f} seconds")
    # Step 5: Generate visualizations
    generate_system_visualizations(model, lookup)
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n=== System Analysis Complete ===")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("System ready for use!")
    print_usage_instructions()


def generate_system_visualizations(model: CapacitiveModel, lookup: LookupTable):
    """
    Generate and save visualization heatmaps showing capacitance distributions.
    """
    print("\n=== Generating Visualizations ===")
    
    try:
        visualizer = Visualizer(model)
        print("Creating capacitance heatmaps...")
        fig, axes = visualizer.plot_capacitance_heatmaps_physical_layout(lookup)
        output_file = 'capacitance_heatmaps_physical_layout.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Heatmaps saved as '{output_file}'")
        plt.close(fig)

        print("Creating combined electrode heatmap...")
        fig_combined, ax_combined = visualizer.plot_combined_electrode_heatmap(lookup)
        output_file_combined = 'combined_electrode_heatmap.png'
        plt.savefig(output_file_combined, dpi=300, bbox_inches='tight')
        print(f"Combined electrode heatmap saved as '{output_file_combined}'")
        plt.close(fig_combined)

        print("Creating parallel plate only heatmaps (no coupling)...")
        fig_pp, axes_pp = visualizer.plot_individual_parallel_plate_heatmaps(lookup)
        output_file_pp = 'parallel_plate_only_heatmaps.png'
        plt.savefig(output_file_pp, dpi=300, bbox_inches='tight')
        print(f"Parallel plate only heatmaps saved as '{output_file_pp}'")
        plt.close(fig_pp)

        print("Visualization completed successfully!")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
def print_usage_instructions():
    """
    Print instructions for using the system components.
    """
    print("\n=== Available System Components ===")
    print("1. Position Estimation Methods:")
    print("   - estimator.estimate_position_nearest_neighbor()")
    print("   - estimator.estimate_position_weighted()")
    print("   - estimator.compare_estimation_methods()")
    print("   - estimator.estimate_positions_from_csv()")
    
    print("\n2. Data Processing:")
    print("   - DataHandler.load_and_correct_csv_data()")
    print("   - DataHandler.compute_baseline()")
    print("   - DataHandler.apply_baseline_correction()")
    
    print("\n3. Model Configuration:")
    print("   - model.set_material_properties()")
    print("   - model.load_tuned_parameters()")
    
    print("\n4. Features:")
    print("   - Automatic baseline correction using first 5 samples")
    print("   - Multiple position estimation algorithms")
    print("   - Physics-based capacitance modeling")
    print("   - Sensor coupling simulation")
    print("   - Visualization tools")
if __name__ == "__main__":
    main()






