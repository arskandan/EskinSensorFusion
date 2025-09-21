import pandas as pd
import matplotlib.pyplot as plt

# Load the saved processed data
df = pd.read_csv("processed_force_estimation.csv")

# Extract force data
force_iso = df["Force_Isothermal_N"]
# force_adia = df["Force_Adiabatic_N"]
force_selected = df["Force_Selected_N"]

# === Plot All Force Estimations ===
plt.figure(figsize=(14, 8))

# Plot original force and all viscoelastic models
plt.plot(force_selected, label="Original Force (Selected)", color="black", linewidth=2, alpha=0.8)

plt.title("Force Estimation: All Viscoelastic Models Comparison")
plt.xlabel("Sample Index")
plt.ylabel("Force (N)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
