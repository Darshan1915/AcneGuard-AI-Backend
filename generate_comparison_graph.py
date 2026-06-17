import matplotlib.pyplot as plt
import numpy as np
import os

# Data for comparison
models = ['Custom CNN', 'VGG-16', 'MobileNetV2', 'EfficientNet-B0', 'ResNet-18 (Ours)']
accuracies = [82.5, 87.3, 89.1, 91.2, 92.5]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff6666']

# Create figure
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Create bars
bars = ax.bar(models, accuracies, color=colors, edgecolor='black')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')

# Formatting
ax.set_ylim(70, 100)  # Set y-axis to start from 70 to highlight differences
ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison for Acne Severity Classification', fontsize=14, fontweight='bold', pad=20)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Highlight our model
bars[-1].set_color('#2ca02c') # Green for our model
bars[-1].set_edgecolor('black')

plt.xticks(rotation=15)
plt.tight_layout()

# Save the plot
output_path = r"C:\Users\Dell\.gemini\antigravity\brain\62a90270-deca-4b5e-817f-ae6b5635a423\model_comparison_graph.png"
plt.savefig(output_path)
print(f"Graph saved to {output_path}")

