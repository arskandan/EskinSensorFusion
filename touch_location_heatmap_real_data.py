"""
Touch location heatmap generation with real data
Fixed threading issues that cause hanging on Windows
"""
import os
# Fix threading issues that cause hanging on Windows
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from CapacitanceModel_Ver1_1 import CapacitiveModel, LookupTable, PositionEstimator, DataHandler
from scipy.signal import butter, filtfilt

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

# --- Low-pass filter implementation ---
def apply_lowpass_filter(data, cutoff_freq=0.2, order=4, fs=1.0):
    """Apply Butterworth low-pass filter to reduce high-frequency noise."""
    try:
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff_freq / nyquist
        
        # Ensure cutoff is valid
        if normalized_cutoff >= 1.0:
            print(f"Warning: Cutoff frequency too high. Using 0.8 * Nyquist frequency.")
            normalized_cutoff = 0.8
        
        filter_result = butter(order, normalized_cutoff, btype='low', analog=False)
        if isinstance(filter_result, tuple) and len(filter_result) >= 2:
            b, a = filter_result[0], filter_result[1]
            filtered_data = filtfilt(b, a, data)
            return filtered_data
        else:
            print("Warning: Unexpected filter result format. Using original data.")
            return data
    except Exception as e:
        print(f"Warning: Low-pass filter failed: {e}. Using original data.")
        return data

def calculate_noise_reduction(original, filtered):
    """Calculate noise reduction in dB."""
    original_power = np.var(original)
    filtered_power = np.var(filtered)
    if original_power > 0 and filtered_power > 0:
        return 10 * np.log10(original_power / filtered_power)
    return 0

# --- CONFIGURATION ---
# DATA_FILE = "Testing For Data Points/data4x4finger.csv"  
DATA_FILE = "Results data/data9x9.csv"  
LOOKUP_RESOLUTION = 0.3  
BASELINE_SAMPLES = 30 

# --- LOW-PASS FILTER CONFIGURATION ---
ENABLE_LOWPASS_FILTER = True
FILTER_CUTOFF_FREQ = 5       # Hz - adjust based on touch dynamics
FILTER_ORDER = 4               # Filter order (higher = steeper roll-off)
SAMPLING_RATE = 20.0            # Hz - adjust based on your data acquisition rate  


# --- LOAD AND PREPARE MODEL WITH PER-CHANNEL 2D POLYNOMIAL FITS ---
print("Initializing capacitive model with per-channel 2D polynomial fits...")
import joblib
channel_poly2d_fits = {}
channels = ['CH0', 'CH1', 'CH2', 'CH3']
for ch in channels:
    try:
        features = joblib.load(f"poly2d_features_{ch}.joblib")
        coef = joblib.load(f"poly2d_coef_{ch}.joblib")
        intercept = joblib.load(f"poly2d_intercept_{ch}.joblib")
        channel_poly2d_fits[ch] = {'features': features, 'coef': coef, 'intercept': intercept}
        print(f"Loaded 2D polynomial fit for {ch}.")
    except Exception as e:
        print(f"Warning: Could not load 2D fit for {ch}: {e}")
        channel_poly2d_fits[ch] = {'features': None, 'coef': None, 'intercept': None}

print("\n--- Per-Channel 2D Polynomial Fit Object Status ---")
for ch in channels:
    fit = channel_poly2d_fits[ch]
    print(f"{ch}: features={type(fit['features'])}, coef={'None' if fit['coef'] is None else f'shape {getattr(fit['coef'], 'shape', type(fit['coef']))}'}, intercept={fit['intercept']}")
print("--------------------------------------\n")

model = CapacitiveModel(use_tuned_parameters=True, channel_poly2d_fits=channel_poly2d_fits)
lookup = LookupTable(model, resolution=LOOKUP_RESOLUTION)
lookup.generate_lookup_table()


# --- LOAD, FILTER, AND CORRECT REAL DATA ---
print(f"\nLoading real data from {DATA_FILE}...")
raw_data = DataHandler.load_csv_data(DATA_FILE)
if raw_data is None:
    raise RuntimeError("Failed to load real data.")

print("Applying Hampel filter to remove outliers...")
# Apply Hampel filter to each channel
for ch in ['CH0', 'CH1', 'CH2', 'CH3']:
    if ch in raw_data.columns:
        raw_data[ch] = hampel_filter(raw_data[ch], window_size=5, n_sigmas=3)

print("Computing baseline and applying baseline correction...")
baseline = DataHandler.compute_baseline(raw_data, baseline_samples=BASELINE_SAMPLES)
corrected_data = DataHandler.apply_baseline_correction(raw_data, baseline, baseline_samples=BASELINE_SAMPLES)
if corrected_data is None or baseline is None:
    raise RuntimeError("Failed to load or correct real data.")

# --- APPLY LOW-PASS FILTERING ---
filtered_data = corrected_data.copy()
if ENABLE_LOWPASS_FILTER:
    print(f"\nApplying low-pass filter (cutoff: {FILTER_CUTOFF_FREQ} Hz, order: {FILTER_ORDER})...")
    
    # Store original data for comparison
    original_data = corrected_data.copy()
    
    # Apply low-pass filter to each channel
    noise_reduction_results = {}
    for ch in ['CH0', 'CH1', 'CH2', 'CH3']:
        if ch in corrected_data.columns:
            print(f"  Filtering {ch}...")
            original_channel = corrected_data[ch].values
            filtered_channel = apply_lowpass_filter(
                original_channel, 
                cutoff_freq=FILTER_CUTOFF_FREQ, 
                order=FILTER_ORDER, 
                fs=SAMPLING_RATE
            )
            filtered_data[ch] = filtered_channel
            
            # Calculate noise reduction
            noise_reduction_db = calculate_noise_reduction(original_channel, filtered_channel)
            noise_reduction_results[ch] = noise_reduction_db
            print(f"    Noise reduction: {noise_reduction_db:.2f} dB")
    
    print(f"\nLow-pass filtering completed!")
    print(f"Average noise reduction: {np.mean(list(noise_reduction_results.values())):.2f} dB")
    
    # Use filtered data for position estimation
    corrected_data = filtered_data
else:
    print("\nLow-pass filtering disabled. Using Hampel-filtered data only.")

# --- ESTIMATE TOUCH LOCATIONS ---
estimator = PositionEstimator(lookup)
estimator.set_baseline(baseline)

# --- SCALE COMPARISON: LOOKUP TABLE vs REAL DATA ---
print("\n=== SCALE COMPARISON: Lookup Table vs Real Sensor Data ===")

# Analyze lookup table scale
lookup_data = lookup.lookup_data
channels = ['CH0', 'CH1', 'CH2', 'CH3']

if lookup_data is not None:
    print("LOOKUP TABLE (Model-Generated) Scale Analysis:")
    for ch in channels:
        if ch in lookup_data.columns:
            values = lookup_data[ch]
            print(f"  {ch}: Min={values.min():.6f}, Max={values.max():.6f}, Mean={values.mean():.6f}, Std={values.std():.6f}")

    lookup_ranges = {ch: (lookup_data[ch].min(), lookup_data[ch].max()) for ch in channels if ch in lookup_data.columns}

    print(f"\nREAL SENSOR DATA (Baseline-Corrected) Scale Analysis:")
    for ch in channels:
        if ch in corrected_data.columns:
            values = corrected_data[ch]
            print(f"  {ch}: Min={values.min():.6f}, Max={values.max():.6f}, Mean={values.mean():.6f}, Std={values.std():.6f}")

    real_ranges = {ch: (corrected_data[ch].min(), corrected_data[ch].max()) for ch in channels if ch in corrected_data.columns}

    # Calculate scale ratios and differences
    print(f"\nSCALE COMPARISON (Real Data vs Lookup Table):")
    scale_issues = False
    for ch in channels:
        if ch in lookup_ranges and ch in real_ranges:
            lookup_range = lookup_ranges[ch][1] - lookup_ranges[ch][0]
            real_range = real_ranges[ch][1] - real_ranges[ch][0]
            
            ratio = real_range / lookup_range if lookup_range > 0 else float('inf')
            
            # Check if scales are similar (within factor of 10)
            if ratio < 0.1 or ratio > 10:
                scale_issues = True
                status = "MISMATCH"
            elif ratio < 0.5 or ratio > 2:
                status = "DIFFERENT"
            else:
                status = "SIMILAR"
            
            print(f"  {ch}: Range Ratio = {ratio:.2f}x {status}")
            print(f"       Lookup: [{lookup_ranges[ch][0]:.6f}, {lookup_ranges[ch][1]:.6f}]")
            print(f"       Real:   [{real_ranges[ch][0]:.6f}, {real_ranges[ch][1]:.6f}]")

    if scale_issues:
        print(f"\nSCALE MISMATCH DETECTED!")
        print(f"   The lookup table and real data are on very different scales.")
        print(f"   This will cause poor position estimation accuracy.")
        print(f"   Consider:")
        print(f"   1. Check if baseline correction is properly applied to both")
        print(f"   2. Verify the physics model parameters match real sensor")
        print(f"   3. Consider scaling/normalization of lookup table or real data")
    else:
        print(f"\nSCALES ARE REASONABLY CONSISTENT")
        print(f"   Lookup table and real data are on compatible scales.")

    # Signal spike analysis - find samples with largest changes from baseline
    print(f"\nSIGNAL SPIKE ANALYSIS (Largest signal changes from baseline):")
    
    # Calculate signal magnitude for each sample (sum of absolute changes across all channels)
    signal_magnitudes = []
    start_idx = BASELINE_SAMPLES  # Skip baseline samples
    
    for idx in range(start_idx, len(corrected_data)):
        row = corrected_data.iloc[idx]
        magnitude = 0
        for ch in channels:
            if ch in row:
                magnitude += abs(row[ch])  # Already baseline-corrected, so this is the change
        signal_magnitudes.append((idx, magnitude))
    
    # Sort by magnitude and get top 5 signal spikes
    signal_magnitudes.sort(key=lambda x: x[1], reverse=True)
    top_spikes = signal_magnitudes[:5]
    
    print(f"Top 5 signal spikes (sorted by total magnitude):")
    for rank, (idx, magnitude) in enumerate(top_spikes, 1):
        row = corrected_data.iloc[idx]
        print(f"  Rank {rank} - Sample {idx} (Total magnitude: {magnitude:.6f}):")
        for ch in channels:
            if ch in row and ch in lookup_data.columns:
                real_val = row[ch]
                # Find closest lookup table value for comparison
                lookup_vals = lookup_data[ch].values
                closest_lookup_idx = np.argmin(np.abs(lookup_vals - real_val))
                closest_lookup_val = lookup_vals[closest_lookup_idx]
                closest_pos = (lookup_data['x'].iloc[closest_lookup_idx], lookup_data['y'].iloc[closest_lookup_idx])
                print(f"    {ch}: Real_Change={real_val:.6f}, Closest_Lookup={closest_lookup_val:.6f} at pos{closest_pos}")
    
    # Additional analysis: show signal spike distribution
    all_magnitudes = [mag for _, mag in signal_magnitudes]
    print(f"\nSignal Spike Distribution Analysis:")
    print(f"  Mean magnitude: {np.mean(all_magnitudes):.6f}")
    print(f"  Max magnitude: {np.max(all_magnitudes):.6f}")
    print(f"  Std deviation: {np.std(all_magnitudes):.6f}")
    print(f"  Samples above mean: {np.sum(np.array(all_magnitudes) > np.mean(all_magnitudes))}")
    print(f"  Samples above 2*std: {np.sum(np.array(all_magnitudes) > (np.mean(all_magnitudes) + 2*np.std(all_magnitudes)))}")
                
else:
    print("ERROR: Lookup table data is not available!")

# --- FILTERING ANALYSIS AND VISUALIZATION ---
if ENABLE_LOWPASS_FILTER and 'original_data' in locals():
    print("\n" + "=" * 60)
    print("FILTERING ANALYSIS: Original vs Filtered Data")
    print("=" * 60)
    
    # Create visualization comparing original vs filtered signals
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Low-Pass Filter Effect: Original vs Filtered Signals', fontsize=16)
    
    channels = ['CH0', 'CH1', 'CH2', 'CH3']
    for i, ch in enumerate(channels):
        if ch in original_data.columns and ch in filtered_data.columns:
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Plot original and filtered signals
            sample_indices = range(len(original_data))
            ax.plot(sample_indices, original_data[ch], 'b-', alpha=0.7, linewidth=1, label='Original (Hampel + Baseline)')
            ax.plot(sample_indices, filtered_data[ch], 'r-', linewidth=2, label=f'Low-Pass Filtered ({FILTER_CUTOFF_FREQ} Hz)')
            
            ax.set_title(f'{ch} - Noise Reduction: {noise_reduction_results[ch]:.1f} dB')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Capacitance (pF)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            orig_std = np.std(original_data[ch])
            filt_std = np.std(filtered_data[ch])
            ax.text(0.02, 0.98, f'Original σ: {orig_std:.4f} pF\nFiltered σ: {filt_std:.4f} pF', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('lowpass_filter_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print(f"\nFILTERING PERFORMANCE SUMMARY:")
    print(f"Filter Configuration:")
    print(f"  Cutoff Frequency: {FILTER_CUTOFF_FREQ} Hz")
    print(f"  Filter Order: {FILTER_ORDER}")
    print(f"  Sampling Rate: {SAMPLING_RATE} Hz")
    
    print(f"\nNoise Reduction per Channel:")
    total_reduction = 0
    for ch in channels:
        if ch in noise_reduction_results:
            reduction = noise_reduction_results[ch]
            print(f"  {ch}: {reduction:.2f} dB")
            total_reduction += reduction
    
    avg_reduction = total_reduction / len(noise_reduction_results)
    print(f"\nAverage Noise Reduction: {avg_reduction:.2f} dB")
    
    # Signal integrity check
    print(f"\nSignal Integrity Analysis:")
    for ch in channels:
        if ch in original_data.columns and ch in filtered_data.columns:
            # Check signal correlation
            correlation = np.corrcoef(original_data[ch], filtered_data[ch])[0, 1]
            # Check peak preservation
            orig_peak = np.max(np.abs(original_data[ch]))
            filt_peak = np.max(np.abs(filtered_data[ch]))
            peak_preservation = (filt_peak / orig_peak) * 100 if orig_peak > 0 else 0
            
            print(f"  {ch}: Correlation = {correlation:.4f}, Peak Preservation = {peak_preservation:.1f}%")
    
    print("=" * 60)
    
else:
    print("\nNo filtering comparison available (filtering disabled or no original data stored).")
    
print("=" * 60)



# --- ESTIMATE TOUCH LOCATIONS FOR ALL SAMPLES ---
print("\n" + "=" * 60)
print("POSITION ESTIMATION WITH FILTERED DATA")
print("=" * 60)

positions = []
for idx, row in corrected_data.iterrows():
    measurements = {ch: float(row[ch]) for ch in ['CH0', 'CH1', 'CH2', 'CH3'] if ch in row}
    # Estimate position (using best method)
    results = estimator.compare_estimation_methods(measurements, apply_baseline_correction=False)
    x, y, _ = results['best_estimate']
    positions.append((x, y))

positions = np.array([p for p in positions if p[0] is not None and p[1] is not None])

# --- POSITION ESTIMATION COMPARISON (if filtering was applied) ---
if ENABLE_LOWPASS_FILTER and 'original_data' in locals():
    print("\nCOMPARING POSITION ESTIMATION: Original vs Filtered Data")
    print("-" * 50)
    
    # Estimate positions using original (unfiltered) data
    positions_original = []
    for idx, row in original_data.iterrows():
        measurements = {ch: float(row[ch]) for ch in ['CH0', 'CH1', 'CH2', 'CH3'] if ch in row}
        results = estimator.compare_estimation_methods(measurements, apply_baseline_correction=False)
        x, y, _ = results['best_estimate']
        positions_original.append((x, y))
    
    positions_original = np.array([p for p in positions_original if p[0] is not None and p[1] is not None])
    
    # Calculate position stability metrics
    if len(positions) > 0 and len(positions_original) > 0:
        # Position variability (standard deviation)
        orig_x_std = np.std(positions_original[:, 0])
        orig_y_std = np.std(positions_original[:, 1])
        filt_x_std = np.std(positions[:, 0])
        filt_y_std = np.std(positions[:, 1])
        
        # Position drift (range)
        orig_x_range = np.max(positions_original[:, 0]) - np.min(positions_original[:, 0])
        orig_y_range = np.max(positions_original[:, 1]) - np.min(positions_original[:, 1])
        filt_x_range = np.max(positions[:, 0]) - np.min(positions[:, 0])
        filt_y_range = np.max(positions[:, 1]) - np.min(positions[:, 1])
        
        print(f"Position Variability Comparison:")
        print(f"  X-coordinate std: Original = {orig_x_std:.3f} cm, Filtered = {filt_x_std:.3f} cm")
        print(f"  Y-coordinate std: Original = {orig_y_std:.3f} cm, Filtered = {filt_y_std:.3f} cm")
        print(f"  X-coordinate range: Original = {orig_x_range:.3f} cm, Filtered = {filt_x_range:.3f} cm")
        print(f"  Y-coordinate range: Original = {orig_y_range:.3f} cm, Filtered = {filt_y_range:.3f} cm")
        
        # Calculate improvement percentages
        x_std_improvement = ((orig_x_std - filt_x_std) / orig_x_std) * 100 if orig_x_std > 0 else 0
        y_std_improvement = ((orig_y_std - filt_y_std) / orig_y_std) * 100 if orig_y_std > 0 else 0
        x_range_improvement = ((orig_x_range - filt_x_range) / orig_x_range) * 100 if orig_x_range > 0 else 0
        y_range_improvement = ((orig_y_range - filt_y_range) / orig_y_range) * 100 if orig_y_range > 0 else 0
        
        print(f"\nPosition Stability Improvement:")
        print(f"  X-coordinate std: {x_std_improvement:+.1f}%")
        print(f"  Y-coordinate std: {y_std_improvement:+.1f}%")
        print(f"  X-coordinate range: {x_range_improvement:+.1f}%")
        print(f"  Y-coordinate range: {y_range_improvement:+.1f}%")
        
        # Create position comparison plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(positions_original[:, 0], positions_original[:, 1], 
                   c=range(len(positions_original)), cmap='Blues', alpha=0.7, s=30, label='Original')
        plt.xlabel('X Position (cm)')
        plt.ylabel('Y Position (cm)')
        plt.title(f'Position Estimates - Original Data\n(X std: {orig_x_std:.3f}, Y std: {orig_y_std:.3f})')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.subplot(1, 2, 2)
        plt.scatter(positions[:, 0], positions[:, 1], 
                   c=range(len(positions)), cmap='Reds', alpha=0.7, s=30, label='Filtered')
        plt.xlabel('X Position (cm)')
        plt.ylabel('Y Position (cm)')
        plt.title(f'Position Estimates - Filtered Data\n(X std: {filt_x_std:.3f}, Y std: {filt_y_std:.3f})')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig('position_estimation_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("-" * 50)


# --- SURE TOUCH WITH SNR-BASED DETECTION ---
print("\nDetecting sure touch using Signal-to-Noise Ratio (SNR)...")

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
baseline_samples = corrected_data.head(BASELINE_SAMPLES)
baseline_noise_levels = []
for idx, row in baseline_samples.iterrows():
    measurements = {ch: float(row[ch]) for ch in ['CH0', 'CH1', 'CH2', 'CH3'] if ch in row}
    signals = np.array([abs(measurements[ch]) for ch in ['CH0', 'CH1', 'CH2', 'CH3']])
    baseline_noise_levels.extend(signals)

baseline_noise_std = np.std(baseline_noise_levels)
print(f"Estimated baseline noise level: {baseline_noise_std:.6f}")

# Calculate SNR for each sample
snr_values = []
signal_powers = []
for idx, row in corrected_data.iterrows():
    measurements = {ch: float(row[ch]) for ch in ['CH0', 'CH1', 'CH2', 'CH3'] if ch in row}
    
    # Calculate both SNR and signal power for comparison
    snr = calculate_snr(measurements, baseline_noise_std)
    signal_power = np.sqrt(np.mean([measurements[ch]**2 for ch in ['CH0', 'CH1', 'CH2', 'CH3']]))
    
    snr_values.append(snr)
    signal_powers.append(signal_power)

snr_values = np.array(snr_values)
signal_powers = np.array(signal_powers)

# Find sure touch using SNR (highest SNR indicates clearest/most reliable signal)
sure_idx = int(np.argmax(snr_values))
sure_x, sure_y = positions[sure_idx]
sure_snr = snr_values[sure_idx]
sure_signal_power = signal_powers[sure_idx]

print(f"Sure touch detected at sample {sure_idx}:")
print(f"  Position: ({sure_x:.2f}, {sure_y:.2f}) cm")
print(f"  SNR: {sure_snr:.2f} dB")
print(f"  Signal Power: {sure_signal_power:.6f}")

# Compare with magnitude-based approach
magnitude_idx = int(np.argmax(signal_powers))
magnitude_snr = snr_values[magnitude_idx]
print(f"  Comparison - Highest magnitude at sample {magnitude_idx} (SNR: {magnitude_snr:.2f} dB)")

print("\nVisualizing sure touch with SNR-based local heatmap...")

# Create a local Gaussian heatmap centered at the sure touch
grid_size = 0.05
x_grid = np.arange(0, 16 + grid_size, grid_size)
y_grid = np.arange(0, 16 + grid_size, grid_size)
X, Y = np.meshgrid(x_grid, y_grid)
sigma = 0.5  
Z = np.exp(-((X - sure_x)**2 + (Y - sure_y)**2) / (2 * sigma**2))

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(Z, extent=(0, 16, 0, 16), origin='lower', cmap='hot', alpha=0.95)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Touch Likelihood')
ax.set_xlabel('X Position (cm)')
ax.set_ylabel('Y Position (cm)')
ax.set_title(f'Sure Touch (SNR-Based) at ({sure_x:.1f}, {sure_y:.1f}) cm\nSNR: {sure_snr:.2f} dB')
ax.set_xlim(0, 16)
ax.set_ylim(0, 16)
ax.grid(True, alpha=0.3)
fig.tight_layout(rect=(0, 0, 1, 1))
plt.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.10)
plt.savefig('touch_location_sure_touch_SNR_heatmap.png', dpi=200, bbox_inches='tight')
plt.show()
print("SNR-based sure touch visualization saved as 'touch_location_sure_touch_SNR_heatmap.png'")

# --- ADDITIONAL SNR ANALYSIS ---
print(f"\n=== SNR Analysis Summary ===")
print(f"Total samples analyzed: {len(snr_values)}")
print(f"Average SNR: {np.mean(snr_values):.2f} dB")
print(f"Max SNR: {np.max(snr_values):.2f} dB (sample {np.argmax(snr_values)})")
print(f"SNR standard deviation: {np.std(snr_values):.2f} dB")
print(f"Samples with SNR > 20 dB: {np.sum(snr_values > 20)}")
print(f"Samples with SNR > 30 dB: {np.sum(snr_values > 30)}")

# Save SNR analysis to file
snr_analysis = pd.DataFrame({
    'sample_idx': range(len(snr_values)),
    'snr_db': snr_values,
    'signal_power': signal_powers,
    'position_x': [p[0] if p[0] is not None else np.nan for p in positions],
    'position_y': [p[1] if p[1] is not None else np.nan for p in positions]
})
snr_analysis.to_csv('snr_analysis_results.csv', index=False)
print("SNR analysis saved to 'snr_analysis_results.csv'")



