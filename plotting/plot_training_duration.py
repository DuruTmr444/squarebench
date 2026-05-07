import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

models_dir = "square_lattice_models"
files = sorted(
    [(f[:-len(".safetensors")], os.path.getmtime(os.path.join(models_dir, f)))
     for f in os.listdir(models_dir) if f.endswith(".safetensors")],
    key=lambda x: x[1]
)

names = [f[0] for f in files]
durations = [0] + [int(files[i][1] - files[i-1][1]) / 60 for i in range(1, len(files))]
colors = ["#4C9BE8" if "8q" in n else "#F4A261" if "7q" in n else "#2A9D8F" for n in names]

total = sum(durations)
td, rem = int(total // (60 * 24)), int(total % (60 * 24))
th, tm = int(rem // 60), int(rem % 60)
parts = ([f"{td}d"] if td else []) + ([f"{th}h"] if th else []) + ([f"{tm}m"] if tm or (not td and not th) else [])
total_str = " ".join(parts)

fig, ax = plt.subplots(figsize=(16, 6))
bars = ax.bar(range(len(names)), durations, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace("linear_function_", "") for n in names], rotation=90, fontsize=8)
ax.set_ylabel("Duration (minutes)")
ax.set_title(f"Training Duration per Square Lattice Model  —  Total: {total_str}")

legend_elements = [
    Patch(facecolor="#2A9D8F", label="4–6 qubits"),
    Patch(facecolor="#F4A261", label="7 qubits"),
    Patch(facecolor="#4C9BE8", label="8 qubits"),
]
ax.legend(handles=legend_elements)

def format_hours(x, _):
    h = int(x // 60)
    m = int(x % 60)
    if h == 0:
        return f"{m}m"
    elif m == 0:
        return f"{h}h"
    else:
        return f"{h}h {m}m"

ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_hours))
ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
ax.yaxis.grid(True, linestyle='--', alpha=0.8)
ax.set_axisbelow(True)
plt.tight_layout()
plt.show()
