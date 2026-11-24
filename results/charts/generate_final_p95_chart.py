import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path("results/charts")
data = pd.read_csv("results/final/final_video_summary.csv")

plt.figure(figsize=(11, 6))
plt.barh(data["Model"], data["Mean_Video_P95_ms"])
plt.xlabel("Mean Video P95 Latency (ms)")
plt.ylabel("Model")
plt.title("P95 Latency Comparison")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(folder / "p95_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("P95 chart generated successfully.")
