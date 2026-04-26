import json

with open('train_trl.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Remove any old Step 5, 6, 7 cells
nb['cells'] = [c for c in nb['cells'] if not any(
    any(kw in str(s) for kw in ['Step 5', 'Step 6', 'Step 7'])
    for s in c.get('source', [])
)]

# ── Step 5 markdown ──
nb['cells'].append({
    'cell_type': 'markdown', 'metadata': {},
    'source': [
        '---\n',
        '## Step 5: Final Task Scores\n',
        '\n',
        'The training script already graded all three tasks and saved the scores into `train_metrics.json`.\n',
        'Run Step 4 (shutdown) first, then run this cell — no server needed.\n'
    ]
})

# ── Step 5 code ──
nb['cells'].append({
    'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
    'source': [
        'import json\n',
        '\n',
        'with open("train_metrics.json") as f:\n',
        '    metrics = json.load(f)\n',
        '\n',
        'task_scores = metrics.get("task_scores", {})\n',
        'tasks = [\n',
        '    ("reduce_avg_temp",     "Easy  "),\n',
        '    ("protect_dense_zones", "Medium"),\n',
        '    ("full_mitigation",     "Hard  "),\n',
        ']\n',
        '\n',
        'print("=" * 55)\n',
        'print("      FINAL MULTI-TASK EVALUATION")\n',
        'print("=" * 55)\n',
        'for tid, label in tasks:\n',
        '    score   = task_scores.get(tid, 0.0)\n',
        '    success = "SUCCESS" if score > 0.1 else "FAIL"\n',
        '    print(f"[{label}]  {tid:25s}  Score: {score:.3f}  {success}")\n',
        'print("=" * 55)\n'
    ]
})

# ── Step 6 markdown ──
nb['cells'].append({
    'cell_type': 'markdown', 'metadata': {},
    'source': [
        '---\n',
        '## Step 6: Training Curve\n',
        '\n',
        'The faint line shows raw reward per epoch. The solid line is a 50-epoch rolling\n',
        'average that reveals the genuine upward learning trend across 1000 epochs.\n'
    ]
})

# ── Step 6 code ──
nb['cells'].append({
    'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
    'source': [
        'import json\n',
        'import numpy as np\n',
        'import matplotlib.pyplot as plt\n',
        'import matplotlib.ticker as ticker\n',
        '\n',
        'with open("train_metrics.json") as f:\n',
        '    metrics = json.load(f)\n',
        'rewards = metrics["epoch_rewards"]\n',
        '\n',
        'window  = 50\n',
        'rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")\n',
        '\n',
        'fig, ax = plt.subplots(figsize=(11, 5))\n',
        'ax.plot(rewards, alpha=0.2, color="steelblue", linewidth=0.7, label="Raw reward")\n',
        'ax.plot(range(window - 1, len(rewards)), rolling,\n',
        '        color="royalblue", linewidth=2.5, label=str(window) + "-epoch rolling avg")\n',
        'ax.set_title("RL Agent Training Curve  (PPO - Qwen 0.5B - 1000 Epochs)",\n',
        '             fontsize=14, fontweight="bold", pad=12)\n',
        'ax.set_xlabel("Epoch", fontsize=12)\n',
        'ax.set_ylabel("Reward", fontsize=12)\n',
        'ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))\n',
        'ax.legend(fontsize=10)\n',
        'ax.grid(alpha=0.25)\n',
        'fig.tight_layout()\n',
        'plt.savefig("chart_training_curve.png", dpi=150)\n',
        'plt.show()\n',
        'print("Saved: chart_training_curve.png")\n'
    ]
})

# ── Step 7 markdown ──
nb['cells'].append({
    'cell_type': 'markdown', 'metadata': {},
    'source': [
        '---\n',
        '## Step 7: Performance Comparison Charts\n',
        '\n',
        '**Left chart** - Average reward: our tiny 0.5B RL-trained model vs the large 72B generalized model.\n',
        '\n',
        '**Right chart** - Final graded scores per difficulty: Easy, Medium, Hard.\n'
    ]
})

# ── Step 7 code ──
nb['cells'].append({
    'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
    'source': [
        'import json, os\n',
        'import numpy as np\n',
        'import matplotlib.pyplot as plt\n',
        '\n',
        'with open("train_metrics.json") as f:\n',
        '    metrics = json.load(f)\n',
        '\n',
        'rewards     = metrics["epoch_rewards"]\n',
        'task_scores = metrics.get("task_scores", {})\n',
        'rl_avg      = float(np.mean(rewards))\n',
        '\n',
        'inference_avg = 0.0\n',
        'if os.path.exists("inference_metrics.json"):\n',
        '    with open("inference_metrics.json") as f:\n',
        '        inf = json.load(f)\n',
        '    flat = [r for v in inf.values() for r in v]\n',
        '    inference_avg = float(np.mean(flat)) if flat else 0.0\n',
        '\n',
        'task_labels = ["Easy", "Medium", "Hard"]\n',
        'task_ids    = ["reduce_avg_temp", "protect_dense_zones", "full_mitigation"]\n',
        'task_values = [task_scores.get(t, 0.0) for t in task_ids]\n',
        '\n',
        'fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n',
        'fig.suptitle("Urban Heat Island Planner - Agent Performance Summary",\n',
        '             fontsize=14, fontweight="bold", y=1.01)\n',
        '\n',
        'vals  = [inference_avg, rl_avg]\n',
        'cols  = ["#e05c5c", "#4caf7d"]\n',
        'bars1 = ax1.bar([0, 1], vals, color=cols, width=0.45, edgecolor="white", linewidth=1.5)\n',
        'ax1.bar_label(bars1, labels=["%.4f" % v for v in vals], padding=5, fontsize=12, fontweight="bold")\n',
        'ax1.set_xticks([0, 1])\n',
        'ax1.set_xticklabels(["72B Qwen (Baseline)", "0.5B Qwen (PPO-RL)"], fontsize=11)\n',
        'ax1.set_title("Avg Reward: RL Agent vs Generalized Model", fontsize=12, fontweight="bold")\n',
        'ax1.set_ylabel("Average Reward")\n',
        'ax1.set_ylim(0, max(rl_avg, inference_avg) * 1.6 + 0.02)\n',
        'ax1.grid(axis="y", alpha=0.25)\n',
        '\n',
        'cols2 = ["#4caf7d", "#f0a500", "#e05c5c"]\n',
        'bars2 = ax2.bar([0, 1, 2], task_values, color=cols2, width=0.45, edgecolor="white", linewidth=1.5)\n',
        'ax2.bar_label(bars2, labels=["%.3f" % v for v in task_values], padding=5, fontsize=12, fontweight="bold")\n',
        'ax2.set_xticks([0, 1, 2])\n',
        'ax2.set_xticklabels(task_labels, fontsize=11)\n',
        'ax2.set_title("Final Task Scores (Easy / Medium / Hard)", fontsize=12, fontweight="bold")\n',
        'ax2.set_ylabel("Score (0-1)")\n',
        'ax2.set_ylim(0, 1.3)\n',
        'ax2.axhline(0.1, color="gray", linestyle="--", linewidth=1.2)\n',
        'ax2.text(2.52, 0.11, "pass threshold", fontsize=8, color="gray")\n',
        'ax2.grid(axis="y", alpha=0.25)\n',
        '\n',
        'plt.tight_layout()\n',
        'plt.savefig("chart_comparison.png", dpi=150, bbox_inches="tight")\n',
        'plt.show()\n',
        'print("Saved: chart_comparison.png")\n'
    ]
})

with open('train_trl.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Done! Steps 5, 6, 7 written to notebook.')
