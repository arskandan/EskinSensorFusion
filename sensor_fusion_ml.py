"""
Sensor Fusion Script for Contact Localization and Force Estimation
Integrates Capacitance Model (CapacitanceModel_Ver1_1) and Pressure Model (Pressure_to_Force)
Simplified physics-based approach for robotic tactile sensing

DUAL FILE INPUT VERSION:
========================
This script expects TWO separate CSV files:
1. Capacitive proximity sensor data (e.g., "capacitance_data.csv")
2. Pressure sensor data (e.g., "pressure_data.csv")

Force Model Approach:
-------------------
- Isothermal model for all touch events
- Physics-based material property calculations with position-dependent elastic modulus

Expected Data Formats:
---------------------
Capacitance CSV should contain columns like:
- DATA0_pF, DATA1_pF, DATA2_pF, DATA3_pF (capacitance values for CH0-CH3)

Pressure CSV should contain columns like:
- Pressure_Pa (pressure values in Pascals)
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Tuple, Dict, Optional, Union, Any
import joblib
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Import the capacitive model
from CapacitanceModel_Ver1_1 import CapacitiveModel, LookupTable, PositionEstimator, DataHandler

# --- Enhanced utility functions ---
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

# --- Hampel filter implementation ---
def hampel_filter(series, window_size=5, n_sigmas=3):
    """Apply Hampel filter to a pandas Series to remove outliers/noise."""
    new_series = series.copy()
    k = 1.4826  # scale factor for Gaussian distribution
    rolling_median = series.rolling(window=2*window_size+1, center=True).median()
    diff = np.abs(series - rolling_median)
    median_abs_deviation = k * diff.rolling(window=2*window_size+1, center=True).median()
    threshold = n_sigmas * median_abs_deviation
    outlier_idx = diff > threshold
    new_series[outlier_idx] = rolling_median[outlier_idx]
    return new_series

# Import pressure-to-force models (add these functions from your Pressure_to_Force.py)
from scipy.signal import butter, filtfilt
import os

def get_position_dependent_E_eq(x, y, center_x=8.0, center_y=8.0):
    """
    Calculate position-dependent elastic modulus E_eq based on distance from center.
    
    Args:
        x, y: Position coordinates in cm
        center_x, center_y: Center coordinates (default: 8cm, 8cm for 16x16cm sensor)
    
    Returns:
        E_eq: Elastic modulus in Pa
    """
    # Calculate distance from center
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    if distance <= 5.0:
        # Inner circle (0-5cm): constant softest material
        return 1.43e5  # Pa
    elif distance <= 7.0:
        # Middle ring (5-7cm): linear increase from 1.43e5 to 1.63e5
        ratio = (distance - 5.0) / (7.0 - 5.0)  # 0 to 1
        return 1.43e5 + ratio * (1.60e5 - 1.43e5)  # Linear interpolation
    elif distance <= 10.0:
        # Outer ring (7-10cm): linear increase from 1.63e5 to 6.8e5
        ratio = (distance - 7.0) / (10.0 - 7.0)  # 0 to 1
        return 1.63e5 + ratio * (6.8e5 - 1.60e5)  # Linear interpolation
    else:
        # Beyond 10cm: use maximum value
        return 6.8e5   # Pa

def isothermal_pressure_to_force(pressure_Pa, baseline_pressure_Pa=None, position_x=8.0, position_y=8.0):
    """Convert pressure to force using isothermal model with position-dependent E_eq (for sustained push events)"""
    # Physics-based constants from Pressure_to_Force.py
    V0 = 420 * 1e-6       # m^3 - Volume
    D_t = 0.02                    # m - Thickness
    E_eq = get_position_dependent_E_eq(position_x, position_y)  # Position-dependent elastic modulus
    
    # Use provided baseline or calculate from first elements
    if baseline_pressure_Pa is None:
        baseline_pressure_Pa = pressure_Pa if np.isscalar(pressure_Pa) else np.mean(pressure_Pa[:min(50, len(pressure_Pa))])
    
    P0 = baseline_pressure_Pa
    P = pressure_Pa
    
    # Isothermal model: delta_V_iso = V0 * (1 - P0/P)
    # Handle division by zero and negative pressures
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(P > 0, P0 / P, 1.0)
        delta_V_iso = V0 * (1 - ratio)
        force_N = E_eq * (delta_V_iso / D_t)
        
    # Clamp negative force values to zero
    force_N = np.maximum(force_N, 0)
    return force_N


class SensorFusionML:
    """
    Machine Learning-based sensor fusion for capacitive touch and pressure sensors
    Handles separate data files for each sensor modality
    """
    
    def __init__(self):
        self.capacitance_model = None
        self.pressure_model = None
        self.position_estimator = None
        self.lookup_table = None
        
        # ML models for fusion
        
        # Model performance metrics
        self.metrics = {}
        
        # Data file paths (can be set externally)
        self.capacitance_file = None
        self.pressure_file = None
    
    def set_data_files(self, capacitance_file, pressure_file):
        """Set the paths to the capacitance and pressure data files"""
        self.capacitance_file = capacitance_file
        self.pressure_file = pressure_file
        print(f"Capacitance data file set to: {capacitance_file}")
        print(f"Pressure data file set to: {pressure_file}")
    
    def run_complete_fusion_pipeline(self, capacitance_file=None, pressure_file=None):
        """Run the complete sensor fusion pipeline with separate data files and peak detection"""
        # Use provided files or fall back to stored file paths
        cap_file = capacitance_file or self.capacitance_file
        press_file = pressure_file or self.pressure_file
        
        if not cap_file or not press_file:
            raise ValueError("Both capacitance and pressure data files must be specified!")
        
        print("Starting Complete Sensor Fusion Pipeline with Peak Detection")
        print("=" * 60)
        
        # Load raw data first for peak detection
        print(f"Loading data files...")
        
        # Check if files exist
        for file_path, sensor_type in [(cap_file, 'capacitance'), (press_file, 'pressure')]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{sensor_type.title()} file '{file_path}' not found")
        
        capacitance_data = pd.read_csv(cap_file)
        pressure_data = pd.read_csv(press_file)
        
        print(f"Loaded {len(capacitance_data)} capacitance samples")
        print(f"Loaded {len(pressure_data)} pressure samples")
        
        # Step 1: Peak Detection and Touch Classification
        classification, peak_info = self.detect_peaks_and_classify_touch(capacitance_data, pressure_data)
        
        # Step 2: Process based on classification
        if classification == "NO_OBJECT_NO_TOUCH":
            print("\n" + "=" * 60)
            print("RESULT: No object detected and no touch")
            print("=" * 60)
            print("Analysis: No significant changes detected in either sensor")
            print("Recommendation: System is idle, no action required")
            return None, peak_info
            
        elif classification == "OBJECT_DETECTED_NO_TOUCH":
            print("\n" + "=" * 60)
            print("RESULT: Object detected but no touch")
            print("=" * 60)
            print("Analysis: Proximity sensor shows object presence")
            print("Analysis: No pressure detected - object is nearby but not touching")
            print("Recommendation: Object hovering above sensor surface")
            
            # Generate proximity-only visualization
            self.visualize_proximity_only_results(capacitance_data, peak_info)
            return capacitance_data, peak_info
            
        elif classification == "TOUCH_DETECTED":
            print("\n" + "=" * 60)
            print("RESULT: Touch detected!")
            print("=" * 60)
            print("Analysis: Both proximity and pressure sensors activated")
            print("Analysis: Proceeding with full sensor fusion analysis...")
            
            # Initialize models for full analysis
            self.initialize_capacitance_model()
            self.initialize_pressure_model()
            
            # Prepare fusion dataset
            features, raw_data = self.prepare_fusion_dataset(cap_file, press_file)
            
            # Make predictions using physics-based models only
            results_df, _ = self.predict_contact_localization_and_force(cap_file, press_file)
            
            # Extract touch location and force from peak
            touch_location, touch_force = self.extract_touch_details(results_df, peak_info)
            
            print(f"\nTOUCH DETAILS:")
            print(f"Location: ({touch_location['x']:.1f}, {touch_location['y']:.1f}) cm")
            print(f"Force: {touch_force:.4f} N")
            print(f"SNR: {touch_location['snr']:.2f} dB")
        
            # Create comprehensive visualizations
            self.visualize_touch_results(results_df, touch_location, touch_force, peak_info)
            
            # Create detailed force comparison plot
            self.plot_force_comparison(results_df)
        
            
            # Save detailed force analysis
            force_analysis = self.save_force_analysis(results_df)
            
            # Save SNR analysis results
            snr_analysis = self.save_snr_analysis(peak_info)
            
            # Save results
            results_df.to_csv('sensor_fusion_predictions.csv', index=False)
            print("Results saved as 'sensor_fusion_predictions.csv'")
            
            return results_df, {'force_analysis': force_analysis, 'snr_analysis': snr_analysis}
            
        else:  # PRESSURE_ONLY
            print("\n" + "=" * 60)
            print("RESULT: Pressure detected without proximity change")
            print("=" * 60)
            print("Analysis: Unusual condition - pressure without proximity")
            print("Recommendation: Check sensor calibration or external interference")
            return pressure_data, peak_info
        
    def initialize_capacitance_model(self):
        """Initialize the capacitance-based touch localization model"""
        print("Initializing capacitance model...")
        
        # Load 2D polynomial fits
        channel_poly2d_fits = {}
        channels = ['CH0', 'CH1', 'CH2', 'CH3']
        
        for ch in channels:
            try:
                features = joblib.load(f"poly2d_features_{ch}.joblib")
                coef = joblib.load(f"poly2d_coef_{ch}.joblib")
                intercept = joblib.load(f"poly2d_intercept_{ch}.joblib")
                channel_poly2d_fits[ch] = {
                    'features': features, 
                    'coef': coef, 
                    'intercept': intercept
                }
                print(f"Loaded 2D polynomial fit for {ch}")
            except Exception as e:
                print(f"Warning: Could not load 2D fit for {ch}: {e}")
                channel_poly2d_fits[ch] = {'features': None, 'coef': None, 'intercept': None}
        
        # Initialize models
        self.capacitance_model = CapacitiveModel(
            use_tuned_parameters=True, 
            channel_poly2d_fits=channel_poly2d_fits
        )
        
        self.lookup_table = LookupTable(self.capacitance_model, resolution=0.5)  # Higher resolution
        self.lookup_table.generate_lookup_table()
        
        self.position_estimator = PositionEstimator(self.lookup_table)
        
        print("Capacitance model initialized successfully")
    
    def initialize_pressure_model(self):
        """Initialize the physics-based pressure-to-force models using Pressure_to_Force.py approach"""
        print("Initializing physics-based pressure models with position-dependent E_eq...")
        
        # Pressure model parameters from Pressure_to_Force.py
        self.pressure_model = {
            # Physics-based constants (from Pressure_to_Force.py)
            'volume_m3': 16 * 16 * 2 * 1e-6,      # V0 = 16x16x2 mm^3 in m^3
            'thickness_m': 0.02,                   # D_t = 20 mm in m
            'elastic_modulus_Pa': 'position_dependent',  # E_eq varies by position
            
            # Event detection parameters
            'event_detection': {
                'tap_thresh_multiplier': 2.0,    # Threshold for tap detection
                'push_duration_min': 10,         # Minimum samples for push detection
                'low_pass_cutoff': 0.1           # Low-pass filter cutoff (normalized)
            }
        }
        print("Physics-based pressure models initialized with position-dependent E_eq:")
        print("  - Inner circle (0-5cm): E_eq = 1.43e5 Pa")
        print("  - Middle ring (5-8cm): E_eq = 1.60e5 Pa") 
        print("  - Outer ring (8-10cm): E_eq = 6.8e5 Pa")
    
    def detect_peaks_and_classify_touch(self, capacitance_data, pressure_data):
        """
        Detect touch using SNR-based analysis and classify touch events
        Returns: touch_classification, peak_info
        """
        print("\n=== SNR-Based Touch Detection and Classification ===")
        
        # Apply Hampel filter to remove outliers from capacitance data
        print("Applying Hampel filter to remove outliers...")
        filtered_cap_data = capacitance_data.copy()
        for ch in ['CH0', 'CH1', 'CH2', 'CH3']:
            if f'DATA{ch[-1]}_pF' in filtered_cap_data.columns:
                filtered_cap_data[f'DATA{ch[-1]}_pF'] = hampel_filter(
                    filtered_cap_data[f'DATA{ch[-1]}_pF'], window_size=5, n_sigmas=3
                )
        
        # Extract signals
        cap_signals = {}
        for ch in ['CH0', 'CH1', 'CH2', 'CH3']:
            if f'DATA{ch[-1]}_pF' in filtered_cap_data.columns:
                cap_signals[ch] = filtered_cap_data[f'DATA{ch[-1]}_pF'].values
        
        # Extract pressure signal
        if 'Pressure_Pa' in pressure_data.columns:
            pressure_signal = pressure_data['Pressure_Pa'].values
        else:
            print("Warning: No Pressure_Pa column found in pressure data")
            pressure_signal = np.zeros(len(pressure_data))
        
        # Calculate baselines (first 30 samples)
        baseline_samples = min(30, len(filtered_cap_data))
        
        cap_baselines = {}
        for ch, signal in cap_signals.items():
            cap_baselines[ch] = np.mean(signal[:baseline_samples])
        
        pressure_baseline = np.mean(pressure_signal[:baseline_samples])
        
        # Baseline-corrected signals
        cap_corrected = {}
        for ch, signal in cap_signals.items():
            cap_corrected[ch] = signal - cap_baselines[ch]
        
        pressure_corrected = pressure_signal - pressure_baseline
        
        # --- SNR-BASED TOUCH DETECTION ---
        print("Detecting sure touch using Signal-to-Noise Ratio (SNR)...")

        # Calculate SNR for each sample
        def calculate_snr(measurements, baseline_noise_std=None):
            """Calculate SNR for a set of channel measurements."""
            signals = np.array([abs(measurements[ch]) for ch in ['CH0', 'CH1', 'CH2', 'CH3']])
            
            # Signal power (RMS of absolute values)
            signal_power = np.sqrt(np.mean(signals**2))
            
            # Noise estimation: use either provided baseline std or estimate from signal variation
            if baseline_noise_std is not None:
                noise_power = baseline_noise_std
            else:
                # Estimate noise as standard deviation of the signals (assumes some channels have low signal)
                noise_power = np.std(signals) + 1e-10  # Add small epsilon to avoid division by zero
            
            # SNR in dB
            snr_db = 20 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
            return max(snr_db, 0)  # Ensure non-negative SNR

        # Estimate baseline noise from early samples (no touch)
        baseline_noise_levels = []
        for idx in range(baseline_samples):
            measurements = {ch: float(cap_corrected[ch][idx]) for ch in cap_corrected.keys()}
            signals = np.array([abs(measurements[ch]) for ch in ['CH0', 'CH1', 'CH2', 'CH3']])
            baseline_noise_levels.extend(signals)

        baseline_noise_std = np.std(baseline_noise_levels)
        print(f"Estimated baseline noise level: {baseline_noise_std:.6f}")

        # Calculate SNR for each sample
        snr_values = []
        signal_powers = []
        for idx in range(len(filtered_cap_data)):
            measurements = {ch: float(cap_corrected[ch][idx]) for ch in cap_corrected.keys()}
            
            # Calculate both SNR and signal power for comparison
            snr = calculate_snr(measurements, baseline_noise_std)
            signal_power = np.sqrt(np.mean([measurements[ch]**2 for ch in ['CH0', 'CH1', 'CH2', 'CH3']]))
            
            snr_values.append(snr)
            signal_powers.append(signal_power)

        snr_values = np.array(snr_values)
        signal_powers = np.array(signal_powers)

        # Find sure touch using SNR (highest SNR indicates clearest/most reliable signal)
        sure_idx = int(np.argmax(snr_values))
        sure_snr = snr_values[sure_idx]
        sure_signal_power = signal_powers[sure_idx]

        print(f"Sure touch detected at sample {sure_idx}:")
        print(f"  SNR: {sure_snr:.2f} dB")
        print(f"  Signal Power: {sure_signal_power:.6f}")

        # Compare with magnitude-based approach
        magnitude_idx = int(np.argmax(signal_powers))
        magnitude_snr = snr_values[magnitude_idx]
        print(f"  Comparison - Highest magnitude at sample {magnitude_idx} (SNR: {magnitude_snr:.2f} dB)")

        # --- TOUCH CLASSIFICATION BASED ON SNR AND PRESSURE ---
        # Set thresholds for classification
        snr_threshold = 20.0  # dB - minimum SNR for reliable touch detection
        pressure_threshold = 2.0 * np.std(np.abs(pressure_corrected))  # Pressure threshold
        
        # Check pressure at sure touch location
        pressure_at_touch = abs(pressure_corrected[sure_idx]) if sure_idx < len(pressure_corrected) else 0
        
        # Classification logic based on SNR and pressure
        if sure_snr < snr_threshold and pressure_at_touch < pressure_threshold:
            classification = "NO_OBJECT_NO_TOUCH"
            message = "No object detected and no touch (low SNR and pressure)"
            
        elif sure_snr >= snr_threshold and pressure_at_touch < pressure_threshold:
            classification = "OBJECT_DETECTED_NO_TOUCH"
            message = "Object detected but no touch (high SNR, low pressure)"
            
        elif sure_snr >= snr_threshold and pressure_at_touch >= pressure_threshold:
            classification = "TOUCH_DETECTED"
            message = "Touch detected (high SNR and pressure)"
            
        else:  # Low SNR but high pressure (unusual case)
            classification = "PRESSURE_ONLY"
            message = "Pressure detected without clear proximity signal (unusual)"
        
        print(f"Classification: {message}")
        
        # Create peak info structure compatible with existing code
        peak_info = {
            'capacitance': {
                'detected': sure_snr >= snr_threshold,
                'info': {
                    'peak_index': sure_idx,
                    'peak_value': sure_signal_power,
                    'peak_height': sure_snr,
                    'all_peaks': [sure_idx],  # Single peak approach
                    'threshold': snr_threshold
                },
                'threshold': snr_threshold,
                'signal_std': baseline_noise_std
            },
            'pressure': {
                'detected': pressure_at_touch >= pressure_threshold,
                'info': {
                    'peak_index': sure_idx,  # Use same index for consistency
                    'peak_value': pressure_at_touch,
                    'peak_height': pressure_at_touch,
                    'all_peaks': [sure_idx],
                    'threshold': pressure_threshold
                },
                'threshold': pressure_threshold,
                'signal_std': np.std(np.abs(pressure_corrected))
            },
            'classification': classification,
            'message': message,
            'snr_analysis': {
                'sure_touch_idx': sure_idx,
                'sure_snr': sure_snr,
                'sure_signal_power': sure_signal_power,
                'avg_snr': np.mean(snr_values),
                'max_snr': np.max(snr_values),
                'snr_std': np.std(snr_values),
                'baseline_noise_std': baseline_noise_std,
                'snr_values': snr_values,
                'signal_powers': signal_powers
            }
        }
        
        return classification, peak_info

    def detect_touch_events(self, pressure_corrected):
        """Detect tap vs push events from pressure signal (legacy method)"""
        # Check if pressure model is initialized
        if self.pressure_model is None:
            print("Warning: Pressure model not initialized. Using default parameters.")
            # Use default parameters
            event_params = {
                'tap_thresh_multiplier': 2.0,
                'push_duration_min': 10,
                'low_pass_cutoff': 0.1
            }
        else:
            event_params = self.pressure_model['event_detection']
        
        # Low-pass filter for push detection
        
        # Design low-pass filter
        order = 4
        cutoff = event_params['low_pass_cutoff']
        low_pass = pressure_corrected.copy()  # Initialize with original signal
        
        try:
            from scipy.signal import butter, filtfilt
            filter_result = butter(order, cutoff, btype='low', analog=False)
            if isinstance(filter_result, tuple) and len(filter_result) >= 2:
                b, a = filter_result[0], filter_result[1]
                # Apply filter
                low_pass = filtfilt(b, a, pressure_corrected)
            else:
                raise ValueError("Unexpected filter result format")
        except Exception as e:
            print(f"Warning: Filter design/application failed: {e}. Using simple moving average.")
            # Fallback to simple moving average
            for i in range(1, len(pressure_corrected)):
                low_pass[i] = 0.9 * low_pass[i-1] + 0.1 * pressure_corrected[i]
        
        # Calculate tap intensity (high-frequency component)
        tap_intensity = np.abs(pressure_corrected - low_pass)
        
        # Dynamic threshold based on signal statistics
        pressure_std = np.std(pressure_corrected)
        tap_thresh = event_params['tap_thresh_multiplier'] * pressure_std
        push_thresh = 0.5 * pressure_std  # Lower threshold for push detection
        
        # Initialize event classification
        events = ["None"] * len(pressure_corrected)
        
        # Detect push regions (sustained pressure)
        push_samples = low_pass > push_thresh
        
        # Convert to numpy array to ensure proper iteration
        push_samples = np.array(push_samples)
        
        # Apply minimum duration filter for pushes
        min_duration = event_params['push_duration_min']
        push_regions = []
        in_push = False
        push_start = 0
        
        for i, is_push in enumerate(push_samples):
            if is_push and not in_push:
                in_push = True
                push_start = i
            elif not is_push and in_push:
                in_push = False
                if i - push_start >= min_duration:
                    push_regions.append((push_start, i))
        
        # Mark push events
        for start, end in push_regions:
            for i in range(start, end):
                events[i] = "Push"
        
        # Detect tap events (only in non-push regions)
        for i, intensity in enumerate(tap_intensity):
            if intensity > tap_thresh and events[i] == "None":
                events[i] = "Tap"
        
        return events, tap_intensity, low_pass
    
    def extract_capacitance_features(self, capacitance_data):
        """Extract features from capacitance sensor data"""
        features = {}
        
        # Raw capacitance values
        for ch in ['CH0', 'CH1', 'CH2', 'CH3']:
            if f'DATA{ch[-1]}_pF' in capacitance_data.columns:
                features[f'{ch}_raw'] = capacitance_data[f'DATA{ch[-1]}_pF'].values
        
        # Baseline-corrected values
        baseline = {ch: capacitance_data[f'DATA{ch[-1]}_pF'].head(30).mean() 
                   for ch in ['CH0', 'CH1', 'CH2', 'CH3'] 
                   if f'DATA{ch[-1]}_pF' in capacitance_data.columns}
        
        for ch in ['CH0', 'CH1', 'CH2', 'CH3']:
            if f'DATA{ch[-1]}_pF' in capacitance_data.columns:
                features[f'{ch}_corrected'] = (capacitance_data[f'DATA{ch[-1]}_pF'] - baseline[ch]).values
        
        # Position estimates from capacitance model
        if self.position_estimator:
            # Set baseline once before the loop
            self.position_estimator.set_baseline(baseline)
            positions = []
            for idx, row in capacitance_data.iterrows():
                measurements = {}
                for ch in ['CH0', 'CH1', 'CH2', 'CH3']:
                    if f'DATA{ch[-1]}_pF' in row:
                        measurements[ch] = row[f'DATA{ch[-1]}_pF'] - baseline[ch]
                
                if measurements:
                    results = self.position_estimator.compare_estimation_methods(
                        measurements, apply_baseline_correction=False
                    )
                    x, y, confidence = results['best_estimate']
                    positions.append([x if x is not None else 0, 
                                    y if y is not None else 0, 
                                    confidence if confidence is not None else 0])
                else:
                    positions.append([0, 0, 0])
            
            positions = np.array(positions)
            features['cap_pos_x'] = positions[:, 0]
            features['cap_pos_y'] = positions[:, 1]
            features['cap_confidence'] = positions[:, 2]
        
        # Derived features
        if all(f'{ch}_corrected' in features for ch in ['CH0', 'CH1', 'CH2', 'CH3']):
            # Total signal magnitude
            features['total_magnitude'] = np.sum([np.abs(features[f'{ch}_corrected']) 
                                                for ch in ['CH0', 'CH1', 'CH2', 'CH3']], axis=0)
            
            # Signal centroid (weighted position)
            total_signal = features['total_magnitude'] + 1e-10
            features['signal_centroid_x'] = np.sum([features[f'{ch}_corrected'] * ch_x 
                                                  for ch, ch_x in zip(['CH0', 'CH1', 'CH2', 'CH3'], 
                                                                     [0, 16, 0, 16])], axis=0) / total_signal
            features['signal_centroid_y'] = np.sum([features[f'{ch}_corrected'] * ch_y 
                                                  for ch, ch_y in zip(['CH0', 'CH1', 'CH2', 'CH3'], 
                                                                     [0, 0, 16, 16])], axis=0) / total_signal
        
        return pd.DataFrame(features)
    
    def extract_pressure_features(self, pressure_data):
        """Extract features from pressure sensor data using physics-based models"""
        features = {}
        
        # Raw pressure
        if 'Pressure_Pa' in pressure_data.columns:
            features['pressure_raw'] = pressure_data['Pressure_Pa'].values
            
            # Baseline-corrected pressure
            baseline_pressure = pressure_data['Pressure_Pa'].head(30).mean()
            pressure_corrected = (pressure_data['Pressure_Pa'] - baseline_pressure).values
            features['pressure_corrected'] = pressure_corrected
            
            # Event detection (tap vs push)
            events, tap_intensity, low_pass = self.detect_touch_events(pressure_corrected)
            
            # Convert string events to numeric (for ML compatibility)
            event_mapping = {"None": 0, "Tap": 1, "Push": 2}
            events_numeric = [event_mapping.get(event, 0) for event in events]
            
            features['events'] = events_numeric  # Numeric values for ML training
            features['events_string'] = events  # Keep original strings for analysis
            features['tap_intensity'] = tap_intensity
            features['pressure_low_pass'] = low_pass
            
            # Physics-based force estimation using simplified models with position-dependent E_eq
            baseline_pressure = baseline_pressure  # Use calculated baseline
            
            # For now, use center position as default (8cm, 8cm)
            # TODO: This should be updated to use actual estimated touch positions
            default_x, default_y = 8.0, 8.0
            
            # Isothermal force model for all events - using full pressure values, not corrected  
            force_isothermal = isothermal_pressure_to_force(
                pressure_data['Pressure_Pa'].values, 
                baseline_pressure_Pa=baseline_pressure,
                position_x=default_x, position_y=default_y
            )
            
            # Use isothermal model for all force calculations
            force_selected = force_isothermal
            
            features['force_isothermal'] = force_isothermal
            features['force_selected'] = force_selected  # Isothermal force for all events
            
            # Additional pressure-based features
            features['pressure_derivative'] = np.gradient(pressure_corrected)
            features['pressure_abs'] = np.abs(pressure_corrected)
            
            # Rolling statistics
            window_size = 5
            pressure_series = pd.Series(pressure_corrected)
            features['pressure_rolling_mean'] = pressure_series.rolling(window_size, center=True).mean().fillna(0).values
            features['pressure_rolling_std'] = pressure_series.rolling(window_size, center=True).std().fillna(0).values
            features['pressure_rolling_max'] = pressure_series.rolling(window_size, center=True).max().fillna(0).values
            
            # Event-based statistics
            tap_count = sum(1 for e in events if e == "Tap")
            push_count = sum(1 for e in events if e == "Push") 
            features['tap_ratio'] = np.full(len(events), tap_count / len(events))
            features['push_ratio'] = np.full(len(events), push_count / len(events))
        
        return pd.DataFrame(features)
    
    def calculate_position_aware_force(self, pressure_data, position_estimates, baseline_pressure_Pa=None):
        """
        Calculate force using position-dependent E_eq for each sample.
        
        Args:
            pressure_data: Pressure values
            position_estimates: Array of (x, y) position estimates for each sample
            baseline_pressure_Pa: Baseline pressure
            
        Returns:
            Dictionary with position-aware force estimates
        """
        if baseline_pressure_Pa is None:
            baseline_pressure_Pa = np.mean(pressure_data[:min(50, len(pressure_data))])
        
        n_samples = len(pressure_data)
        force_isothermal_pos = np.zeros(n_samples)
        
        # Calculate force for each sample using its estimated position
        for i in range(n_samples):
            if i < len(position_estimates):
                pos_x, pos_y = position_estimates[i]
                # Ensure position is valid, otherwise use center
                if np.isnan(pos_x) or np.isnan(pos_y):
                    pos_x, pos_y = 8.0, 8.0
            else:
                pos_x, pos_y = 8.0, 8.0  # Default to center
            
            # Calculate isothermal force for this sample with position-dependent E_eq
            force_isothermal_pos[i] = isothermal_pressure_to_force(
                pressure_data[i], baseline_pressure_Pa, pos_x, pos_y
            )
        
        return {
            'force_isothermal_pos': force_isothermal_pos
        }
    
    def prepare_fusion_dataset(self, capacitance_file, pressure_file):
        """Prepare the combined dataset for ML training from separate sensor files"""
        print(f"Loading capacitance data from {capacitance_file}...")
        print(f"Loading pressure data from {pressure_file}...")
        
        # Check if files exist
        import os
        for file_path, sensor_type in [(capacitance_file, 'capacitance'), (pressure_file, 'pressure')]:
            if not os.path.exists(file_path):
                print(f"Error: {sensor_type.title()} file '{file_path}' not found!")
                print("Available CSV files in workspace:")
                # List available CSV files
                for root, dirs, files in os.walk('.'):
                    for file in files:
                        if file.endswith('.csv'):
                            file_path_found = os.path.join(root, file).replace('\\', '/')
                            print(f"  {file_path_found}")
                raise FileNotFoundError(f"{sensor_type.title()} file '{file_path}' not found")
        
        # Load raw data from both sensors
        capacitance_data = pd.read_csv(capacitance_file)
        pressure_data = pd.read_csv(pressure_file)
        
        print(f"Loaded {len(capacitance_data)} capacitance samples with columns: {list(capacitance_data.columns)}")
        print(f"Loaded {len(pressure_data)} pressure samples with columns: {list(pressure_data.columns)}")
        
        # Synchronize the datasets (align by timestamp or sample index)
        min_samples = min(len(capacitance_data), len(pressure_data))
        if len(capacitance_data) != len(pressure_data):
            print(f"Warning: Sample count mismatch! Capacitance: {len(capacitance_data)}, Pressure: {len(pressure_data)}")
            print(f"Using first {min_samples} samples from both datasets for synchronization")
            capacitance_data = capacitance_data.head(min_samples)
            pressure_data = pressure_data.head(min_samples)
        
        # Extract features from both sensor modalities
        cap_features = self.extract_capacitance_features(capacitance_data)
        pressure_features = self.extract_pressure_features(pressure_data)
        
        # Calculate position-aware force estimates using capacitance-based position estimates
        if 'cap_pos_x' in cap_features.columns and 'cap_pos_y' in cap_features.columns:
            print("Calculating position-aware force estimates using capacitance-based positions...")
            position_estimates = list(zip(cap_features['cap_pos_x'].values, cap_features['cap_pos_y'].values))
            
            # Calculate force with position-dependent E_eq
            baseline_pressure = pressure_data['Pressure_Pa'].head(30).mean() if 'Pressure_Pa' in pressure_data.columns else None
            position_aware_forces = self.calculate_position_aware_force(
                pressure_data['Pressure_Pa'].values if 'Pressure_Pa' in pressure_data.columns else np.zeros(len(pressure_data)),
                position_estimates,
                baseline_pressure
            )
            
            # Add position-aware force features
            for force_name, force_values in position_aware_forces.items():
                pressure_features[force_name] = force_values
            
            print(f"Added position-aware force features: {list(position_aware_forces.keys())}")
        
        # Combine features
        fusion_features = pd.concat([cap_features, pressure_features], axis=1)
        
        # Add interaction features (capacitance × pressure)
        if 'total_magnitude' in cap_features.columns and 'pressure_corrected' in pressure_features.columns:
            fusion_features['cap_pressure_interaction'] = (
                cap_features['total_magnitude'] * pressure_features['pressure_corrected']
            )
        
        # Combine raw data for reference
        combined_raw_data = pd.concat([capacitance_data.reset_index(drop=True), 
                                     pressure_data.reset_index(drop=True)], axis=1)
        
        print(f"Generated {len(fusion_features.columns)} fusion features from synchronized datasets")
        return fusion_features, combined_raw_data
    
    
    def predict_contact_localization_and_force(self, capacitance_file, pressure_file):
        """Make predictions using physics-based models only"""
        print(f"\n=== Making Predictions on {capacitance_file} + {pressure_file} ===")
        
        # Prepare data
        features, raw_data = self.prepare_fusion_dataset(capacitance_file, pressure_file)
        
        results = {}
        
        # Use capacitive model position predictions
        if 'cap_pos_x' in features.columns and 'cap_pos_y' in features.columns:
            results['cap_pos_x'] = features['cap_pos_x'].values
            results['cap_pos_y'] = features['cap_pos_y'].values
        
        # Use physics-based force estimates
        if 'force_selected' in features.columns:
            results['physics_force'] = features['force_selected'].values
            results['force_isothermal'] = features['force_isothermal'].values
            
            # Include event classification
            if 'events_string' in features.columns:
                results['event_type'] = features['events_string'].values
                results['tap_intensity'] = features['tap_intensity'].values
        
        # Add raw sensor data
        for col in raw_data.columns:
            if col not in results:
                results[col] = raw_data[col].values
        
        results_df = pd.DataFrame(results)
        return results_df, features
    
    def extract_touch_details(self, results_df, peak_info):
        """Extract touch location and force details from SNR analysis"""
        touch_location = {'x': 0, 'y': 0, 'snr': 0}
        touch_force = 0
        
        # Get peak indices from SNR analysis
        if 'snr_analysis' in peak_info and 'sure_touch_idx' in peak_info['snr_analysis']:
            peak_sample_idx = peak_info['snr_analysis']['sure_touch_idx']
            touch_location['snr'] = peak_info['snr_analysis']['sure_snr']
        else:
            # Fallback to old peak detection method
            cap_peak_idx = peak_info['capacitance']['info'].get('peak_index', 0) if peak_info['capacitance']['detected'] else 0
            pressure_peak_idx = peak_info['pressure']['info'].get('peak_index', 0) if peak_info['pressure']['detected'] else 0
            peak_sample_idx = cap_peak_idx if peak_info['capacitance']['detected'] else pressure_peak_idx
        
        if peak_sample_idx < len(results_df):
            # Extract position
            if 'fused_pos_x' in results_df.columns and 'fused_pos_y' in results_df.columns:
                touch_location['x'] = results_df['fused_pos_x'].iloc[peak_sample_idx]
                touch_location['y'] = results_df['fused_pos_y'].iloc[peak_sample_idx]
            elif 'cap_pos_x' in results_df.columns and 'cap_pos_y' in results_df.columns:
                touch_location['x'] = results_df['cap_pos_x'].iloc[peak_sample_idx]
                touch_location['y'] = results_df['cap_pos_y'].iloc[peak_sample_idx]
            
            # Extract force
            if 'fused_force' in results_df.columns:
                touch_force = results_df['fused_force'].iloc[peak_sample_idx]
            elif 'force_selected' in results_df.columns:
                touch_force = results_df['force_selected'].iloc[peak_sample_idx]
            elif 'force_isothermal' in results_df.columns:
                touch_force = results_df['force_isothermal'].iloc[peak_sample_idx]
            
            # Use SNR from analysis if not already set
            if touch_location['snr'] == 0 and all(col in results_df.columns for col in ['DATA0_pF', 'DATA1_pF', 'DATA2_pF', 'DATA3_pF']):
                measurements = {
                    'CH0': results_df['DATA0_pF'].iloc[peak_sample_idx],
                    'CH1': results_df['DATA1_pF'].iloc[peak_sample_idx],
                    'CH2': results_df['DATA2_pF'].iloc[peak_sample_idx],
                    'CH3': results_df['DATA3_pF'].iloc[peak_sample_idx]
                }
                touch_location['snr'] = self.calculate_snr(measurements)
        
        return touch_location, abs(touch_force)
    
    def visualize_proximity_only_results(self, capacitance_data, peak_info):
        """Visualize results when only proximity sensor detects an object"""
        print("\n=== Generating Proximity-Only Visualization ===")
        
        # Create proximity detection plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Capacitance signals over time
        time_axis = range(len(capacitance_data))
        
        for i, ch in enumerate(['DATA0_pF', 'DATA1_pF', 'DATA2_pF', 'DATA3_pF']):
            if ch in capacitance_data.columns:
                ax1.plot(time_axis, capacitance_data[ch], label=f'CH{i}', alpha=0.8)
        
        # Highlight peak if detected
        if peak_info['capacitance']['detected']:
            peak_idx = peak_info['capacitance']['info']['peak_index']
            threshold = peak_info['capacitance']['threshold']
            ax1.axvline(x=peak_idx, color='red', linestyle='--', alpha=0.8, label='Object Detection Peak')
            ax1.axhline(y=threshold, color='orange', linestyle=':', alpha=0.6, label='Detection Threshold')
        
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Capacitance (pF)')
        ax1.set_title('Object Detected (Proximity Only)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Object detection indicator
        ax2.text(0.5, 0.7, 'OBJECT DETECTED', fontsize=20, ha='center', va='center', 
                transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        ax2.text(0.5, 0.5, 'Proximity sensor activated', fontsize=14, ha='center', va='center', transform=ax2.transAxes)
        ax2.text(0.5, 0.3, 'No physical contact detected', fontsize=14, ha='center', va='center', transform=ax2.transAxes)
        ax2.text(0.5, 0.1, 'Object hovering above surface', fontsize=12, ha='center', va='center', 
                transform=ax2.transAxes, style='italic')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('proximity_only_detection.png', dpi=300, bbox_inches='tight')
        print("Proximity-only visualization saved as 'proximity_only_detection.png'")
        plt.show()
    
    def visualize_touch_results(self, results_df, touch_location, touch_force, peak_info):
        """Visualize results when touch is detected with location and force"""
        print(f"\n=== Generating Touch Detection Visualization ===")
        print(f"Touch at ({touch_location['x']:.1f}, {touch_location['y']:.1f}) cm with {touch_force:.4f} N force")
        
        # Create main figure with subplots - only 3 plots now
        fig = plt.figure(figsize=(18, 6))
        
        # 1. Touch Location Heatmap (SNR-based)
        ax1 = plt.subplot(1, 3, 1)
        
        # Create Gaussian heatmap centered at touch location
        grid_size = 0.05
        x_grid = np.arange(0, 16 + grid_size, grid_size)
        y_grid = np.arange(0, 16 + grid_size, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        sigma = 0.5
        Z = np.exp(-((X - touch_location['x'])**2 + (Y - touch_location['y'])**2) / (2 * sigma**2))
        
        im = ax1.imshow(Z, extent=(0, 16, 0, 16), origin='lower', cmap='hot', alpha=0.95)
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='Touch Likelihood')
        ax1.set_xlabel('X Position (cm)')
        ax1.set_ylabel('Y Position (cm)')
        ax1.set_title(f'TOUCH DETECTED\nLocation: ({touch_location["x"]:.1f}, {touch_location["y"]:.1f}) cm\nSNR: {touch_location["snr"]:.2f} dB')
        ax1.set_xlim(0, 16)
        ax1.set_ylim(0, 16)
        ax1.grid(True, alpha=0.3)
        
        # Add crosshair at touch location
        ax1.axhline(y=touch_location['y'], color='white', linestyle='--', alpha=0.8, linewidth=2)
        ax1.axvline(x=touch_location['x'], color='white', linestyle='--', alpha=0.8, linewidth=2)
        
        # 2. Force Plot (Peak Force Only)
        ax2 = plt.subplot(1, 3, 2)
        time_axis = range(len(results_df))
        
        # Plot force data without labels (background)
        if 'force_isothermal' in results_df.columns:
            ax2.plot(time_axis, results_df['force_isothermal'], 
                    color="red", linewidth=2, alpha=0.3)
        
        # Highlight peak force
        if peak_info['pressure']['detected']:
            force_peak_idx = peak_info['pressure']['info']['peak_index']
            ax2.axvline(x=force_peak_idx, color='orange', linestyle=':', alpha=0.8, 
                       label=f'Peak Force: {touch_force:.4f} N')
            ax2.scatter([force_peak_idx], [touch_force], color='red', s=100, zorder=5)
        
        ax2.set_title(f"Force: {touch_force:.4f} N")
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Force (N)")
        ax2.grid(True)
        ax2.legend()
        
        # 3. Raw sensor signals with peaks
        ax3 = plt.subplot(1, 3, 3)
        
        # Plot capacitance channels
        for i, ch in enumerate(['DATA0_pF', 'DATA1_pF', 'DATA2_pF', 'DATA3_pF']):
            if ch in results_df.columns:
                ax3.plot(time_axis, results_df[ch], label=f'CH{i}', alpha=0.7)
        
        # Highlight capacitance peak
        if peak_info['capacitance']['detected']:
            cap_peak_idx = peak_info['capacitance']['info']['peak_index']
            ax3.axvline(x=cap_peak_idx, color='blue', linestyle='--', alpha=0.8, label='Proximity Peak')
        
        # Plot pressure on secondary axis
        if 'Pressure_Pa' in results_df.columns:
            ax3_twin = ax3.twinx()
            ax3_twin.plot(time_axis, results_df['Pressure_Pa'], 'r-', label='Pressure', alpha=0.8)
            ax3_twin.set_ylabel('Pressure (Pa)', color='r')
            
            # Highlight pressure peak
            if peak_info['pressure']['detected']:
                pressure_peak_idx = peak_info['pressure']['info']['peak_index']
                ax3_twin.axvline(x=pressure_peak_idx, color='red', linestyle=':', alpha=0.8, label='Pressure Peak')
        
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Capacitance (pF)')
        ax3.set_title('Raw Sensor Signals with Peak Detection')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('touch_detection_comprehensive.png', dpi=300, bbox_inches='tight')
        print("Touch detection visualization saved as 'touch_detection_comprehensive.png'")
        plt.show()
        
        # Create separate large heatmap for touch location
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 8))
        im = ax_heatmap.imshow(Z, extent=(0, 16, 0, 16), origin='lower', cmap='hot', alpha=0.95)
        cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04, label='Touch Likelihood')
        
        # Add crosshair and annotation
        ax_heatmap.axhline(y=touch_location['y'], color='white', linestyle='--', alpha=0.8, linewidth=2)
        ax_heatmap.axvline(x=touch_location['x'], color='white', linestyle='--', alpha=0.8, linewidth=2)
        ax_heatmap.scatter([touch_location['x']], [touch_location['y']], 
                         color='yellow', s=300, marker='*', edgecolor='black', linewidth=3, zorder=5)
        
        ax_heatmap.set_xlabel('X Position (cm)', fontsize=14)
        ax_heatmap.set_ylabel('Y Position (cm)', fontsize=14)
        ax_heatmap.set_title(f'TOUCH DETECTED\nLocation: ({touch_location["x"]:.1f}, {touch_location["y"]:.1f}) cm | Force: {touch_force:.4f} N', 
                           fontsize=16)
        ax_heatmap.set_xlim(0, 16)
        ax_heatmap.set_ylim(0, 16)
        ax_heatmap.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('touch_heatmap_large.png', dpi=300, bbox_inches='tight')
        print("Large touch heatmap saved as 'touch_heatmap_large.png'")
        plt.show()

    def calculate_snr(self, measurements, baseline_noise_std=None):
        """Calculate SNR for a set of channel measurements."""
        signals = np.array([abs(measurements[ch]) for ch in ['CH0', 'CH1', 'CH2', 'CH3'] if ch in measurements])
        
        # Signal power (RMS of absolute values)
        signal_power = np.sqrt(np.mean(signals**2))
        
        # Noise estimation: use either provided baseline std or estimate from signal variation
        if baseline_noise_std is not None:
            noise_power = baseline_noise_std
        else:
            # Estimate noise as standard deviation of the signals (assumes some channels have low signal)
            noise_power = np.std(signals) + 1e-10  # Add small epsilon to avoid division by zero
        
        # SNR in dB
        snr_db = 20 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        return max(snr_db, 0)  # Ensure non-negative SNR

    def visualize_results(self, results_df, save_plots=True):
        """Visualize sensor fusion results with SNR-based heatmap and force analysis"""
        print("\n=== Visualizing Results ===")
        
        # Create main figure with subplots
        fig = plt.figure(figsize=(18, 14))
        
        # 1. Sure Touch Heatmap (SNR-based) - Main focus like your example
        ax1 = plt.subplot(2, 3, 1)
        
        # Calculate SNR for each sample to find sure touch
        if all(col in results_df.columns for col in ['DATA0_pF', 'DATA1_pF', 'DATA2_pF', 'DATA3_pF']):
            # Estimate baseline noise from early samples
            baseline_samples = 30
            baseline_noise_levels = []
            for i in range(min(baseline_samples, len(results_df))):
                measurements = {
                    'CH0': results_df['DATA0_pF'].iloc[i],
                    'CH1': results_df['DATA1_pF'].iloc[i], 
                    'CH2': results_df['DATA2_pF'].iloc[i],
                    'CH3': results_df['DATA3_pF'].iloc[i]
                }
                signals = np.array([abs(measurements[ch]) for ch in ['CH0', 'CH1', 'CH2', 'CH3']])
                baseline_noise_levels.extend(signals)
            
            baseline_noise_std = np.std(baseline_noise_levels)
            
            # Calculate SNR for each sample
            snr_values = []
            for i in range(len(results_df)):
                measurements = {
                    'CH0': results_df['DATA0_pF'].iloc[i],
                    'CH1': results_df['DATA1_pF'].iloc[i],
                    'CH2': results_df['DATA2_pF'].iloc[i], 
                    'CH3': results_df['DATA3_pF'].iloc[i]
                }
                snr = self.calculate_snr(measurements, baseline_noise_std)
                snr_values.append(snr)
            
            # Find sure touch (highest SNR)
            sure_idx = int(np.argmax(snr_values))
            if 'fused_pos_x' in results_df.columns and 'fused_pos_y' in results_df.columns:
                sure_x = results_df['fused_pos_x'].iloc[sure_idx]
                sure_y = results_df['fused_pos_y'].iloc[sure_idx]
                sure_snr = snr_values[sure_idx]
                
                # Create Gaussian heatmap centered at sure touch
                grid_size = 0.05
                x_grid = np.arange(0, 16 + grid_size, grid_size)
                y_grid = np.arange(0, 16 + grid_size, grid_size)
                X, Y = np.meshgrid(x_grid, y_grid)
                sigma = 0.5
                Z = np.exp(-((X - sure_x)**2 + (Y - sure_y)**2) / (2 * sigma**2))
                
                im = ax1.imshow(Z, extent=(0, 16, 0, 16), origin='lower', cmap='hot', alpha=0.95)
                cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='Touch Likelihood')
                ax1.set_xlabel('X Position (cm)')
                ax1.set_ylabel('Y Position (cm)')
                ax1.set_title(f'Sure Touch (SNR-Based) at ({sure_x:.1f}, {sure_y:.1f}) cm\nSNR: {sure_snr:.2f} dB')
                ax1.set_xlim(0, 16)
                ax1.set_ylim(0, 16)
                ax1.grid(True, alpha=0.3)
        
        # 2. Force Model - Focus on peak force detection
        ax2 = plt.subplot(2, 3, 2)
        time_axis = range(len(results_df))
        
        # Plot force data without labels (background)
        if 'force_isothermal' in results_df.columns:
            ax2.plot(time_axis, results_df['force_isothermal'], 
                    color="red", linewidth=2, alpha=0.3)
        
        # Find and highlight peak force
        if 'force_isothermal' in results_df.columns:
            force_values = results_df['force_isothermal'].values
            peak_idx = int(np.argmax(force_values))
            peak_force = force_values[peak_idx]
            ax2.axvline(x=peak_idx, color='orange', linestyle=':', alpha=0.8, 
                       label=f'Peak Force: {peak_force:.4f} N')
            ax2.scatter([peak_idx], [peak_force], color='red', s=100, zorder=5)
        
        ax2.set_title("Peak Force Detection")
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Force (N)")
        ax2.grid(True)
        ax2.legend()
        
        # 3. Position comparison scatter plot
        ax3 = plt.subplot(2, 3, 3)
        if all(col in results_df.columns for col in ['fused_pos_x', 'fused_pos_y', 'cap_pos_x', 'cap_pos_y']):
            ax3.scatter(results_df['cap_pos_x'], results_df['cap_pos_y'], 
                       alpha=0.6, label='Capacitance Only', s=20, c='blue')
            ax3.scatter(results_df['fused_pos_x'], results_df['fused_pos_y'], 
                       alpha=0.8, label='Sensor Fusion', s=20, c='red')
            ax3.set_xlabel('X Position (cm)')
            ax3.set_ylabel('Y Position (cm)')
            ax3.set_title('Contact Localization Comparison')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, 16)
            ax3.set_ylim(0, 16)
        
        # 4. Raw sensor signals with event overlay
        ax4 = plt.subplot(2, 3, 4)
        
        # Plot capacitance channels
        for i, ch in enumerate(['DATA0_pF', 'DATA1_pF', 'DATA2_pF', 'DATA3_pF']):
            if ch in results_df.columns:
                ax4.plot(time_axis, results_df[ch], label=f'CH{i}', alpha=0.7)
        
        # Plot pressure on secondary axis
        if 'Pressure_Pa' in results_df.columns:
            ax4_twin = ax4.twinx()
            ax4_twin.plot(time_axis, results_df['Pressure_Pa'], 'r-', label='Pressure', alpha=0.8)
            ax4_twin.set_ylabel('Pressure (Pa)', color='r')
            
            # Highlight events if available
            if 'event_type' in results_df.columns:
                events = results_df['event_type'].values
                tap_indices = [i for i, e in enumerate(events) if e == "Tap"]
                push_indices = [i for i, e in enumerate(events) if e == "Push"]
                
                if tap_indices:
                    ax4_twin.scatter([time_axis[i] for i in tap_indices], 
                                   [results_df['Pressure_Pa'].iloc[i] for i in tap_indices], 
                                   c='blue', marker='^', s=50, alpha=0.8, label='Tap Events')
                
                if push_indices:
                    ax4_twin.scatter([time_axis[i] for i in push_indices], 
                                   [results_df['Pressure_Pa'].iloc[i] for i in push_indices], 
                                   c='green', marker='s', s=30, alpha=0.6, label='Push Events')
                
                ax4_twin.legend(loc='upper right')
        
        ax4.set_xlabel('Sample Index')
        ax4.set_ylabel('Capacitance (pF)')
        ax4.set_title('Raw Sensor Signals with Event Detection')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # 5. SNR Analysis
        ax5 = plt.subplot(2, 3, 5)
        if 'snr_values' in locals():
            ax5.plot(time_axis, snr_values, color='green', linewidth=2, label='SNR (dB)')
            ax5.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='20 dB threshold')
            ax5.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='30 dB threshold')
            ax5.set_xlabel('Sample Index')
            ax5.set_ylabel('SNR (dB)')
            ax5.set_title('Signal-to-Noise Ratio Analysis')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
        
        # 6. Force vs Position Heatmap
        ax6 = plt.subplot(2, 3, 6)
        if all(col in results_df.columns for col in ['fused_pos_x', 'fused_pos_y', 'fused_force']):
            # Find high-force samples for touch detection
            force_threshold = np.percentile(results_df['fused_force'], 75)
            touch_samples = results_df[results_df['fused_force'] > force_threshold]
            
            if len(touch_samples) > 0:
                scatter = ax6.scatter(touch_samples['fused_pos_x'], touch_samples['fused_pos_y'], 
                                    c=touch_samples['fused_force'], cmap='hot', 
                                    s=60, alpha=0.8)
                plt.colorbar(scatter, ax=ax6, label='Force (N)')
            
            ax6.set_xlabel('X Position (cm)')
            ax6.set_ylabel('Y Position (cm)')
            ax6.set_title('Force Distribution Heatmap')
            ax6.set_xlim(0, 16)
            ax6.set_ylim(0, 16)
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('sensor_fusion_comprehensive_results.png', dpi=300, bbox_inches='tight')
            print("Comprehensive plots saved as 'sensor_fusion_comprehensive_results.png'")
        
        plt.show()
        
        # Create separate SNR-based sure touch heatmap like your example
        if 'sure_x' in locals() and 'sure_y' in locals():
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(6, 6))
            im = ax_heatmap.imshow(Z, extent=(0, 16, 0, 16), origin='lower', cmap='hot', alpha=0.95)
            cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04, label='Touch Likelihood')
            ax_heatmap.set_xlabel('X Position (cm)')
            ax_heatmap.set_ylabel('Y Position (cm)')
            ax_heatmap.set_title(f'Sure Touch (SNR-Based) at ({sure_x:.1f}, {sure_y:.1f}) cm\nSNR: {sure_snr:.2f} dB')
            ax_heatmap.set_xlim(0, 16)
            ax_heatmap.set_ylim(0, 16)
            ax_heatmap.grid(True, alpha=0.3)
            fig_heatmap.tight_layout(rect=(0, 0, 1, 1))
            plt.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.10)
            
            if save_plots:
                plt.savefig('touch_location_sure_touch_SNR_heatmap.png', dpi=200, bbox_inches='tight')
                print("SNR-based sure touch heatmap saved as 'touch_location_sure_touch_SNR_heatmap.png'")
            plt.show()
    
    def save_force_analysis(self, results_df, filename='processed_force_estimation.csv'):
        """Save detailed force analysis similar to your example"""
        print(f"\n=== Saving Force Analysis to {filename} ===")
        
        # Create comprehensive force analysis dataframe
        force_analysis = pd.DataFrame({
            'sample_idx': range(len(results_df)),
            'timestamp': results_df.index if hasattr(results_df.index, 'name') else range(len(results_df))
        })
        
        # Add all force estimates
        if 'force_isothermal' in results_df.columns:
            force_analysis['Force_Isothermal_N'] = results_df['force_isothermal']
        
        if 'force_selected' in results_df.columns:
            force_analysis['Force_Selected_N'] = results_df['force_selected']
        
        if 'fused_force' in results_df.columns:
            force_analysis['Force_MLFusion_N'] = results_df['fused_force']
        
        # Add pressure and capacitance data
        if 'Pressure_Pa' in results_df.columns:
            force_analysis['Pressure_Pa'] = results_df['Pressure_Pa']
        
        for i, ch in enumerate(['DATA0_pF', 'DATA1_pF', 'DATA2_pF', 'DATA3_pF']):
            if ch in results_df.columns:
                force_analysis[f'Capacitance_CH{i}_pF'] = results_df[ch]
        
        # Add position estimates
        if 'fused_pos_x' in results_df.columns:
            force_analysis['Position_X_cm'] = results_df['fused_pos_x']
        if 'fused_pos_y' in results_df.columns:
            force_analysis['Position_Y_cm'] = results_df['fused_pos_y']
        
        # Add event classification
        if 'event_type' in results_df.columns:
            force_analysis['Event_Type'] = results_df['event_type']
        
        # Save to CSV
        force_analysis.to_csv(filename, index=False)
        print(f"Force analysis saved with {len(force_analysis.columns)} columns and {len(force_analysis)} samples")
        
        return force_analysis
    
    def save_snr_analysis(self, peak_info, filename='snr_analysis_results.csv'):
        """Save detailed SNR analysis results"""
        print(f"\n=== Saving SNR Analysis to {filename} ===")
        
        if 'snr_analysis' not in peak_info:
            print("No SNR analysis data available")
            return None
        
        snr_data = peak_info['snr_analysis']
        
        # Create SNR analysis dataframe
        snr_analysis = pd.DataFrame({
            'sample_idx': range(len(snr_data['snr_values'])),
            'snr_db': snr_data['snr_values'],
            'signal_power': snr_data['signal_powers']
        })
        
        # Save to CSV
        snr_analysis.to_csv(filename, index=False)
        
        # Print summary statistics
        print(f"=== SNR Analysis Summary ===")
        print(f"Total samples analyzed: {len(snr_data['snr_values'])}")
        print(f"Average SNR: {snr_data['avg_snr']:.2f} dB")
        print(f"Max SNR: {snr_data['max_snr']:.2f} dB (sample {snr_data['sure_touch_idx']})")
        print(f"SNR standard deviation: {snr_data['snr_std']:.2f} dB")
        print(f"Samples with SNR > 20 dB: {np.sum(snr_data['snr_values'] > 20)}")
        print(f"Samples with SNR > 30 dB: {np.sum(snr_data['snr_values'] > 30)}")
        print(f"Baseline noise level: {snr_data['baseline_noise_std']:.6f}")
        print(f"SNR analysis saved with {len(snr_analysis)} samples")
        
        return snr_analysis
    
    def plot_force_comparison(self, results_df, save_plot=True):
        """Create detailed force comparison plot focusing on peak force detection"""
        print("\n=== Creating Peak Force Detection Plot ===")
        
        plt.figure(figsize=(14, 8))
        
        # Plot force data without labels (background)
        time_axis = range(len(results_df))
        
        if 'force_isothermal' in results_df.columns:
            plt.plot(time_axis, results_df['force_isothermal'], 
                    color="red", linewidth=2, alpha=0.3)
        
        # Find and highlight peak force
        if 'force_isothermal' in results_df.columns:
            force_values = results_df['force_isothermal'].values
            peak_idx = int(np.argmax(force_values))
            peak_force = force_values[peak_idx]
            plt.axvline(x=peak_idx, color='orange', linestyle=':', alpha=0.8, 
                       label=f'Peak Force: {peak_force:.4f} N')
            plt.scatter([peak_idx], [peak_force], color='red', s=100, zorder=5)
        
        plt.title("Peak Force Detection")
        plt.xlabel("Sample Index")
        plt.ylabel("Force (N)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('force_peak_detection_model.png', dpi=300, bbox_inches='tight')
            print("Peak force detection plot saved as 'force_peak_detection_model.png'")
        
        plt.show()


def main():
    """Main execution function with SNR-based touch detection and classification"""
    print("Sensor Fusion ML Pipeline - SNR-Based Touch Detection & Classification")
    print("=" * 70)
    print("This script performs intelligent SNR-based touch detection:")
    print("Low SNR & low pressure → No object detected, no touch")
    print("High SNR & low pressure → Object detected, no touch") 
    print("High SNR & high pressure → Touch detected (full analysis)")
    print("Low SNR & high pressure → Unusual condition")
    print("=" * 70)
    print("Features:")
    print("  • Hampel filter for noise removal")
    print("  • SNR-based touch location detection")
    print("  • Physics-based force estimation with elastic modulus")
    print("  • CapacitanceModel_Ver1_1 integration")
    print("  • Machine learning sensor fusion")
    print("=" * 70)
    
    # Configuration - Update these paths to your actual data files
    CAPACITANCE_FILE = "DataProximity6.csv"    # UPDATE: Your capacitance file
    PRESSURE_FILE = "synced_pressure_6.csv"         # UPDATE: Your pressure file
    
    print(f"Capacitance Data File: {CAPACITANCE_FILE}")
    print(f"Pressure Data File: {PRESSURE_FILE}")
    print()
    
    # Initialize fusion system
    fusion = SensorFusionML()
    fusion.set_data_files(CAPACITANCE_FILE, PRESSURE_FILE)
    
    try:
        # Run complete pipeline with peak detection
        results, analysis = fusion.run_complete_fusion_pipeline()
        
        # Print final summary based on results
        print("\n" + "=" * 70)
        print("FINAL ANALYSIS SUMMARY")
        print("=" * 70)
        
        if results is None:
            print("System Status: IDLE")
            print("No significant activity detected")
            
        elif isinstance(analysis, dict) and 'classification' in analysis:
            classification = analysis['classification']
            
            if classification == "OBJECT_DETECTED_NO_TOUCH":
                print("System Status: OBJECT PROXIMITY")
                print("Object hovering near sensor surface")
                print("No physical contact established")
                
            elif classification == "TOUCH_DETECTED":
                print("System Status: TOUCH ACTIVE")
                if hasattr(fusion, 'metrics') and fusion.metrics:
                    print("ML Model Performance:")
                    for metric, value in fusion.metrics.items():
                        print(f"   {metric}: {value:.6f}")
                
                print("Complete analysis files generated")
                print("Touch location and force visualizations saved")
                
            else:
                print("System Status: ANOMALY DETECTED")
                print("Unusual sensor behavior - check calibration")
        
        print("\nPeak detection and classification completed!")
        
    except FileNotFoundError as e:
        print(f"\nFile Error: {e}")
        print("\nSETUP INSTRUCTIONS:")
        print("1. Update CAPACITANCE_FILE and PRESSURE_FILE paths above")
        print("2. Ensure both CSV files exist in specified locations")
        print("3. Available reference files:")
        print("   - Testing For Data Points/data6x6location.csv (capacitance)")
        print("   - Upload your pressure sensor CSV file")
        print("\nREQUIRED DATA FORMATS:")
        print("   Capacitance CSV: DATA0_pF, DATA1_pF, DATA2_pF, DATA3_pF columns")
        print("   Pressure CSV: Pressure_Pa column")
        
    except Exception as e:
        print(f"\nPipeline Error: {e}")
        print("Check data file formats and sensor calibration")


def run_with_custom_files(capacitance_file, pressure_file):
    """
    Convenience function to run sensor fusion with custom file paths
    
    Args:
        capacitance_file (str): Path to capacitive sensor CSV file
        pressure_file (str): Path to pressure sensor CSV file
    
    Example:
        run_with_custom_files("my_cap_data.csv", "my_pressure_data.csv")
    """
    print(f"Running Sensor Fusion with Custom Files")
    print(f"Capacitance: {capacitance_file}")
    print(f"Pressure: {pressure_file}")
    
    fusion = SensorFusionML()
    results_df, force_analysis = fusion.run_complete_fusion_pipeline(capacitance_file, pressure_file)
    return results_df, force_analysis
    
    # Create comprehensive visualizations (including SNR-based heatmap)
    fusion.visualize_results(results_df)
    
    # Create detailed force comparison plot
    fusion.plot_force_comparison(results_df)
    
    # Save detailed force analysis
    force_analysis = fusion.save_force_analysis(results_df)
    
    # Save models and hyperparameter results
    fusion.save_models()
    fusion.save_hyperparameter_results()
    results_df.to_csv('sensor_fusion_predictions.csv', index=False)
    print("Results saved as 'sensor_fusion_predictions.csv'")
    
    # Print performance summary
    print("\n" + "=" * 50)
    print("SENSOR FUSION PERFORMANCE SUMMARY")
    print("=" * 50)
    for metric, value in fusion.metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # Print SNR and force analysis summary
    if all(col in results_df.columns for col in ['DATA0_pF', 'DATA1_pF', 'DATA2_pF', 'DATA3_pF']):
        # Calculate and print SNR statistics
        baseline_noise_levels = []
        for i in range(min(30, len(results_df))):
            measurements = {
                'CH0': results_df['DATA0_pF'].iloc[i],
                'CH1': results_df['DATA1_pF'].iloc[i], 
                'CH2': results_df['DATA2_pF'].iloc[i],
                'CH3': results_df['DATA3_pF'].iloc[i]
            }
            signals = np.array([abs(measurements[ch]) for ch in ['CH0', 'CH1', 'CH2', 'CH3']])
            baseline_noise_levels.extend(signals)
        
        baseline_noise_std = np.std(baseline_noise_levels)
        
        # Calculate SNR for each sample
        snr_values = []
        for i in range(len(results_df)):
            measurements = {
                'CH0': results_df['DATA0_pF'].iloc[i],
                'CH1': results_df['DATA1_pF'].iloc[i],
                'CH2': results_df['DATA2_pF'].iloc[i], 
                'CH3': results_df['DATA3_pF'].iloc[i]
            }
            snr = fusion.calculate_snr(measurements, baseline_noise_std)
            snr_values.append(snr)
        
        print(f"\n=== SNR Analysis Summary ===")
        print(f"Total samples analyzed: {len(snr_values)}")
        print(f"Average SNR: {np.mean(snr_values):.2f} dB")
        print(f"Max SNR: {np.max(snr_values):.2f} dB (sample {np.argmax(snr_values)})")
        print(f"SNR standard deviation: {np.std(snr_values):.2f} dB")
        print(f"Samples with SNR > 20 dB: {np.sum(np.array(snr_values) > 20)}")
        print(f"Samples with SNR > 30 dB: {np.sum(np.array(snr_values) > 30)}")
    
    # Print force analysis summary
    if 'force_selected' in results_df.columns:
        force_values = np.array(results_df['force_selected'].values)
        meaningful_force = force_values[force_values > 0.001]
        print(f"\n=== Force Analysis Summary ===")
        print(f"Total force samples: {len(force_values)}")
        print(f"Samples with meaningful force (>0.001 N): {len(meaningful_force)}")
        if len(meaningful_force) > 0:
            print(f"Average force: {np.mean(meaningful_force):.4f} N")
            print(f"Max force: {np.max(meaningful_force):.4f} N")
            print(f"Force standard deviation: {np.std(meaningful_force):.4f} N")
    
    print("\nSensor fusion pipeline completed successfully!")


if __name__ == "__main__":
    main()



