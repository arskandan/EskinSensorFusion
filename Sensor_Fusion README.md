## Overview
This comprehensive system combines **capacitive proximity sensing** with **pressure sensing** to create an intelligent touch detection and analysis pipeline for robotic skin applications.


## Key Features

### Touch Classification
- **IDLE**: No object detected (Low SNR + Low Pressure)
- **PROXIMITY**: Object hovering (High SNR + Low Pressure)  
- **TOUCH**: Active contact with location & force (High SNR + High Pressure)
- **ANOMALY**: Unusual sensor patterns (Low SNR + High Pressure)

### Advanced Signal Processing
- SNR-based touch detection
- Hampel outlier filtering
- Baseline drift correction
- Multi-channel signal fusion


### Comprehensive Visualization
- Touch location heatmaps
- Force detection plots
- SNR analysis charts
- Multi-sensor signal overlays


## Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn joblib scipy
```

### Required Files

You need these additional Python modules in the same directory:
- `CapacitanceModel_Ver1_1.py` - Capacitive sensing model
- `poly2d_*.joblib` files - 2D proximity factor calibration data

### Basic Usage

1. **Update data file paths** in `sensor_fusion_ml.py`:
   ```python
   CAPACITANCE_FILE = "your_capacitance_data.csv"
   PRESSURE_FILE = "your_pressure_data.csv"
   ```

2. **Run the pipeline**:
   ```bash
   python sensor_fusion_ml.py
   ```

3. **Using custom files programmatically**:
   ```python
   from sensor_fusion_ml import run_with_custom_files
   
   results, analysis = run_with_custom_files(
       "my_capacitance_data.csv", 
       "my_pressure_data.csv"
   )
   ```

## Required Data Format

### Capacitance Sensor CSV
Must contain these columns:
- `DATA0_pF` - Channel 0 capacitance (picoFarads)
- `DATA1_pF` - Channel 1 capacitance (picoFarads)
- `DATA2_pF` - Channel 2 capacitance (picoFarads)
- `DATA3_pF` - Channel 3 capacitance (picoFarads)

### Pressure Sensor CSV  
Must contain:
- `Pressure_Pa` - Pressure readings (Pascals)

**Important**: Both files must be synchronized with the same number of samples.

## Output Files Generated

| File | Description |
|------|-------------|
| `sensor_fusion_predictions.csv` | Complete analysis results |
| `touch_location_sure_touch_SNR_heatmap.png` | Touch location visualization |
| `force_peak_detection_model.png` | Force analysis plot |
| `sensor_fusion_comprehensive_results.png` | Multi-panel dashboard |
| `processed_force_estimation.csv` | Detailed force analysis |


## Technical Specifications

### Physics Models
- **Capacitive Model**: 2D polynomial proximity factors
- **Force Model**: Isothermal pressure-to-force conversion
- **Material Properties**: Position-dependent elastic modulus
- **SNR Analysis**: Signal quality assessment


### Modifying Visualization Parameters

```python
# Heatmap resolution and appearance
grid_size = 0.05          # cm - Heatmap grid resolution
sigma = 0.5              # Gaussian blur for touch visualization
sensor_area = (16, 16)   # cm - Sensor dimensions
```

## Troubleshooting

### Common Issues

**FileNotFoundError**: 
- Verify file paths in configuration
- Check file permissions and accessibility

**Missing Columns**: 
- Ensure CSV files contain required column names
- Check for typos in column headers

**Memory Issues**: 
- Reduce data size for initial testing
- Optimize threading settings for your system

**Poor SNR Performance**:
- Check sensor calibration and baseline correction
- Verify proper shielding and grounding













