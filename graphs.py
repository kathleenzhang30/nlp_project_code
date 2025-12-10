import matplotlib.pyplot as plt
import numpy as np

#histogram 

baseline = {'MSE': 1.0328, 'RMSE': 1.0163, 'MAE': 0.8486, 'R2': 0.0243}
rope = {'MSE': 0.9454, 'RMSE': 0.9723, 'MAE': 0.8112, 'R2': 0.0350}
rope_special = {'MSE': 0.8874, 'RMSE': 0.9461, 'MAE': 0.7826, 'R2': 0.0770}

metrics = ['MSE', 'RMSE', 'MAE', 'R2']
baseline_vals = [baseline[m] for m in metrics]
rope_vals = [rope[m] for m in metrics]
rope_special_vals = [rope_special[m] for m in metrics]

x = np.arange(len(metrics))         
width = 0.25                        

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(x - width, baseline_vals, width, label='Baseline')
ax.bar(x,         rope_vals,      width, label='RoPE')
ax.bar(x + width, rope_special_vals, width, label='RoPE + Special Days')

ax.set_xlabel('Metric')
ax.set_ylabel('Value')
ax.set_title('Regression Metrics over 100 Runs (Normalized Test Set)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

ax.grid(axis='y', linestyle='--', alpha=0.4)

fig.tight_layout()
plt.show()

# line graph representing true values and the predicted values 

def to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else x

val_targets = to_numpy(val_targets)
preds_0 = to_numpy(preds_0)
preds_1 = to_numpy(preds_1)
val_preds = to_numpy(val_preds)

start_idx = 0    
end_idx = 100     
time = range(start_idx, end_idx)

plt.figure(figsize=(12, 6))

plt.plot(time, val_targets[start_idx:end_idx], label="Actual", color="green", linewidth=2)
plt.plot(time, val_preds[start_idx:end_idx], label="RoPE + Special Market Days", color = "blue", linewidth=1.5, alpha=0.8)

plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title(f"Predictions vs Actual (time steps {start_idx} to {end_idx})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
