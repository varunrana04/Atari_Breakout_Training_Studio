import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Find the latest CSV log
csv_file = 'checkpoints/history_dueling_double_dqn.csv'
if not os.path.exists(csv_file):
    print("No CSV found.")
    exit(1)

df = pd.read_csv(csv_file)

plt.figure(figsize=(10, 6))
plt.plot(df['episode'], df['avg_reward_100'], label='Avg Reward (100 Ep)')
plt.title('Dueling Double DQN Training Curve')
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.grid(True)
plt.legend()

out_path = r'C:\Users\Varun\.gemini\antigravity-ide\brain\c7192f50-fbfc-4e07-8fca-dcc082bef1ef\curve.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved to {out_path}")
