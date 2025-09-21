import pandas as pd
import numpy as np

# Load the pressure data file
file_path = "pressure_force_log_fourth1.csv"  
df = pd.read_csv(file_path)

# Extract pressure signal
P = df['Pressure_Pa'].to_numpy(dtype=float)

# === Constants for Force Estimation ===
V0 = 420 * 1e-6               # m^3
D_t = 0.02                    # m
E_eq = 6.8e5                 # Pa
gamma = 1.4                   # Adiabatic index for air
P0 = np.mean(np.array(P[:50]))          # Baseline pressure (first 100 samples)

# === Filtering Parameters for Push/Tap Classification ===
alpha1 = 0.6  # low-pass
alpha2 = 0.5  # high-pass
alpha3 = 0.8  # smoothing

# Initialize signals
low_pass = np.zeros_like(P)
high_pass = np.zeros_like(P)
tap_intensity = np.zeros_like(P)

# Apply filters
for n in range(1, len(P)):
    low_pass[n] = alpha1 * low_pass[n-1] + (1 - alpha1) * P[n]
    high_pass[n] = alpha2 * high_pass[n-1] + alpha2 * (P[n] - P[n-1])
    tap_intensity[n] = alpha3 * tap_intensity[n-1] + (1 - alpha3) * abs(high_pass[n])

# === Force Estimation ===
# Isothermal
delta_V_iso = V0 * (1 - P0 / P)
force_iso_raw = E_eq * (delta_V_iso / D_t)

# Adiabatic
delta_V_adia = V0 * (1 - np.power(P0 / P, 1/gamma))
force_adia_raw = E_eq * (delta_V_adia / D_t)

# Clamp negative force values to zero in raw signals
force_iso_raw = np.maximum(force_iso_raw, 0)
force_adia_raw = np.maximum(force_adia_raw, 0)

# === Smooth the force signals to reduce noise ===
force_smooth_alpha = 0.4  # Smoothing factor for force (higher = more smooth)

# Initialize smoothed force arrays
force_iso = np.zeros_like(force_iso_raw)
force_adia = np.zeros_like(force_adia_raw)

# Apply exponential smoothing to force signals
force_iso[0] = force_iso_raw[0]
force_adia[0] = force_adia_raw[0]


for n in range(1, len(P)):
    force_iso[n] = force_smooth_alpha * force_iso[n-1] + (1 - force_smooth_alpha) * force_iso_raw[n]
    force_adia[n] = force_smooth_alpha * force_adia[n-1] + (1 - force_smooth_alpha) * force_adia_raw[n]

# Clamp negative force values to zero after smoothing
force_iso = np.maximum(force_iso, 0)
force_adia = np.maximum(force_adia, 0)

# === Improved Event Classification Logic ===
event = ["None"] * len(P)
tap_thresh = 6
push_delta = 1000
min_gap_between_events = 35

# Step 1: Identify all regions with elevated pressure (potential push events)
elevated_pressure_mask = low_pass > P0 + push_delta * 0.2  # Lower threshold for detection

# Step 2: Find continuous push regions
push_regions = []
in_push = False
push_start = 0

for i in range(len(P)):
    if elevated_pressure_mask[i] and not in_push:
        # Start of a push region
        in_push = True
        push_start = i
    elif not elevated_pressure_mask[i] and in_push:
        # End of a push region
        in_push = False
        push_regions.append((push_start, i-1))

# Handle case where push region extends to end of data
if in_push:
    push_regions.append((push_start, len(P)-1))

print(f"Found {len(push_regions)} push regions: {push_regions}")

# Step 3: Mark all samples in push regions as Push events
for region_idx, (start, end) in enumerate(push_regions):
    push_samples_in_region = 0
    for i in range(start, end + 1):
        # Mark all samples in push regions as Push, not just those above main threshold
        if low_pass[i] > P0 + push_delta * 0.3:  # Even lower threshold for marking
            event[i] = "Push"
            push_samples_in_region += 1
    print(f"Region {region_idx + 1} ({start}-{end}): marked {push_samples_in_region} samples as Push")

print(f"Marked push events in all {len(push_regions)} regions")

# Step 4: Find tap events ONLY in regions NOT marked as push
potential_tap_indices = []
i = 0
while i < len(P):
    if tap_intensity[i] > tap_thresh and event[i] == "None":
        # Check if this region is completely isolated from push regions
        is_near_push = False
        for start, end in push_regions:
            if i >= start - 50 and i <= end + 50:  # 50 sample buffer around push regions
                is_near_push = True
                break
        
        if not is_near_push:
            # Find the peak in this region
            peak_start = i
            peak_end = i
            
            # Extend to find the full peak region
            while peak_end < len(P) - 1 and tap_intensity[peak_end + 1] > tap_thresh * 0.5:
                peak_end += 1
            
            # Find the maximum within this region
            if peak_end >= peak_start:
                local_max_idx = peak_start + np.argmax(tap_intensity[peak_start:peak_end + 1])
                potential_tap_indices.append(local_max_idx)
            
            i = peak_end + min_gap_between_events
        else:
            i += 1
    else:
        i += 1

# Step 5: Mark detected taps
for tap_idx in potential_tap_indices:
    event[tap_idx] = "Tap"

# Count events
push_count = sum(1 for e in event if e == "Push")
tap_count = len(potential_tap_indices)

print(f"Final results: {tap_count} taps, {push_count} push samples")
print(f"Tap indices: {potential_tap_indices}")

if tap_count > 0:
    print("WARNING: Taps detected in long touch data - this may indicate classification issues")
else:
    print("SUCCESS: No taps detected - perfect for long touch data!")


# Use model-based force
force_selected = [force_adia[i] if event[i] == "Tap" else force_iso[i] for i in range(len(P))]
# Clamp negative values in force_selected to zero
force_selected = np.maximum(force_selected, 0)

# Add all results to dataframe
df['Low_Pass'] = low_pass
df['Tap_Intensity'] = tap_intensity
df['Force_Isothermal_N'] = force_iso
df['Force_Adiabatic_N'] = force_adia
df['Event'] = event
df['Force_Selected_N'] = force_selected

df.to_csv("processed_force_estimation.csv", index=False)
print("Saved as: processed_force_estimation.csv")