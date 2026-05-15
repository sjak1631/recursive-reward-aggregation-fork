"""
Re-evaluate all configurations using TRUE CVaR (not histogram-based).
3つの統合グラフを生成する。
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

CVAR_ALPHA = 0.05

def compute_true_cvar(returns):
    """真のCVaR（ヒストグラム近似なし）"""
    returns = np.asarray(returns, dtype=np.float64).reshape(-1)
    if returns.size == 0:
        return 0.0
    sorted_returns = np.sort(returns)
    n_tail = max(1, int(np.floor(returns.size * CVAR_ALPHA)))
    return float(np.mean(sorted_returns[:n_tail]))


def load_returns(pkl_path):
    """all_R_*.pkl から報酬リストを読み込んで、エピソードごとにフラット化"""
    with open(pkl_path, 'rb') as f:
        all_R = pickle.load(f)
    episode_returns = []
    for seed_data in all_R:
        for eval_run in seed_data:
            episode_returns.append(np.array(eval_run, dtype=np.float64).flatten())
    return episode_returns


workspace_path = Path('/workspace/RRA/portfolio_cvar/workspace')

# CVaR配置（without_sumsを除外）
cvar_configs = {
    'CVaR bins=25': ('seed4_cvar_bins25',  'log_adapt_reward=False_adapt_state=False_cvar_bins25'),
    'CVaR bins=51': ('seed4_cvar_bins51',  'log_adapt_reward=False_adapt_state=False_cvar_bins51'),
    'CVaR bins=101':('seed4_cvar_bins101', 'log_adapt_reward=False_adapt_state=False_cvar_bins101'),
    'CVaR bins=201':('seed4_cvar_bins201', 'log_adapt_reward=False_adapt_state=False_cvar_bins201'),
    'CVaR bins=401':('seed4_cvar_bins401', 'log_adapt_reward=False_adapt_state=False_cvar_bins401'),
    'CVaR bins=801':('seed4_cvar_bins801', 'log_adapt_reward=False_adapt_state=False_cvar_bins801'),
}

# 他の構成
other_configs = {
    'Mean Return':  ('seed4_mean_return',  'log_adapt_reward=False_adapt_state=False_mean_return'),
    'Sharpe':       ('seed4_sharpe',       'log_adapt_reward=False_adapt_state=False_sharpe'),
    'CVaR bins=801 (without sums)': ('seed4_cvar_bins801_without_sums', 'log_adapt_reward=False_adapt_state=False_cvar_bins801_without_sums'),
}

print("=" * 90)
print(f"{'Configuration':<30} {'Recorded':<15} {'True CVaR':<15} {'Difference':<15}")
print("=" * 90)

results = {}

# Load all data
all_configs = {**cvar_configs, **other_configs}
for label, (folder, log_folder) in all_configs.items():
    pkl_path = workspace_path / folder / log_folder / 'all_R_test.pkl'
    npy_path = workspace_path / folder / log_folder / 'cvar_test.npy'

    if not pkl_path.exists():
        print(f"{label:<30} SKIPPED (no data)")
        continue

    episode_returns = load_returns(pkl_path)
    true_cvars = [compute_true_cvar(ep) for ep in episode_returns]
    true_cvar_mean = np.mean(true_cvars)

    recorded = None
    if npy_path.exists():
        recorded_vals = np.load(npy_path).flatten()
        recorded = float(np.mean(recorded_vals))

    results[label] = {
        'true_cvars': true_cvars,
        'true_mean': true_cvar_mean,
        'recorded_mean': recorded,
    }

    if label.startswith('CVaR') and 'without' not in label:
        rec_str = f"{recorded:.6f}" if recorded is not None else "N/A"
        diff_str = f"{recorded - true_cvar_mean:+.6f}" if recorded is not None else "N/A"
    else:
        rec_str = "N/A"
        diff_str = "N/A"
    
    print(f"{label:<30} {rec_str:<15} {true_cvar_mean:<15.6f} {diff_str:<15}")

print("=" * 90)

# 最高性能のCVaR構成を特定（bins=801 without_sumsを除外）
best_cvar_label = max([k for k in cvar_configs.keys()], 
                      key=lambda x: results[x]['true_mean'])
best_cvar_mean = results[best_cvar_label]['true_mean']
print(f"\nBest CVaR config: {best_cvar_label} (true CVaR = {best_cvar_mean:.6f})")

# ============= グラフ1: CVaR bins系（左：時系列、右：ヒストグラム） =============
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(16, 5.5))

# 左：時系列
colors = plt.cm.tab10(np.linspace(0, 1, len(cvar_configs)))
for (label, data), color in zip([(k, results[k]) for k in cvar_configs.keys()], colors):
    cvars = data['true_cvars']
    x = np.arange(len(cvars))
    ax1a.plot(x, cvars, marker='o', linestyle='-', color=color, alpha=0.8,
              label=f'{label}', linewidth=2, markersize=4)

ax1a.set_xlabel('Step', fontsize=11)
ax1a.set_ylabel('True CVaR (bottom 5% mean)', fontsize=11)
ax1a.set_title('(a) CVaR Time Series - All Bins Configurations', fontsize=12, fontweight='bold')
ax1a.legend(fontsize=9, loc='best')
ax1a.grid(True, alpha=0.3)
ax1a.axhline(y=0, color='black', linewidth=0.5)

# 右：ヒストグラム
cvar_labels = list(cvar_configs.keys())
cvar_true_means = [results[l]['true_mean'] for l in cvar_labels]
cvar_recorded_means = [results[l]['recorded_mean'] for l in cvar_labels]

x = np.arange(len(cvar_labels))
width = 0.35

bars1 = ax1b.bar(x - width/2, cvar_recorded_means, width, label='Recorded (Histogram-based)',
                 color='steelblue', edgecolor='black', linewidth=1.5)
bars2 = ax1b.bar(x + width/2, cvar_true_means, width, label='True CVaR (Re-calculated)',
                 color='coral', edgecolor='black', linewidth=1.5)

ax1b.set_xlabel('Bins Configuration', fontsize=11)
ax1b.set_ylabel('Mean CVaR', fontsize=11)
ax1b.set_title('(b) CVaR: Recorded vs True Values', fontsize=12, fontweight='bold')
ax1b.set_xticks(x)
ax1b.set_xticklabels(cvar_labels, fontsize=9, rotation=45, ha='right')
ax1b.legend(fontsize=9, loc='best')
ax1b.grid(True, alpha=0.3, axis='y')
ax1b.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('/workspace/RRA/portfolio_cvar/imgs/fig1_cvar_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: fig1_cvar_comparison.png")
plt.close()

# ============= グラフ2: bins=401 vs Sharpe (時系列、横点線で平均) =============
fig2, ax2 = plt.subplots(figsize=(12, 6))

for label in ['CVaR bins=401', 'Sharpe']:
    if label not in results:
        continue
    data = results[label]
    cvars = data['true_cvars']
    x = np.arange(len(cvars))
    
    color = '#FF6B6B' if label == 'CVaR bins=401' else '#4ECDC4'
    ax2.plot(x, cvars, marker='o', linestyle='-', color=color, alpha=0.7,
             label=label, linewidth=2.5, markersize=6)
    
    # 平均を横点線で表示
    mean_val = np.mean(cvars)
    ax2.axhline(y=mean_val, color=color, linestyle='--', linewidth=2, alpha=0.6,
                label=f'{label} (mean={mean_val:.6f})')

ax2.set_xlabel('Step', fontsize=12)
ax2.set_ylabel('True CVaR (bottom 5% mean)', fontsize=12)
ax2.set_title('CVaR bins=401 vs Sharpe: Time Series with Mean', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='best')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('/workspace/RRA/portfolio_cvar/imgs/fig2_bins401_vs_sharpe.png', dpi=150, bbox_inches='tight')
print("Saved: fig2_bins401_vs_sharpe.png")
plt.close()

# ============= グラフ3: bins=401 vs bins=801_without_sums (左：時系列、右：バーチャート) =============
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 5.5))

# 左：時系列
for label, color in [('CVaR bins=401', '#FF6B6B'), ('CVaR bins=801 (without sums)', '#FFA500')]:
    if label not in results:
        continue
    data = results[label]
    cvars = data['true_cvars']
    x = np.arange(len(cvars))
    ax3a.plot(x, cvars, marker='o', linestyle='-', color=color, alpha=0.8,
              label=label, linewidth=2.5, markersize=6)

ax3a.set_xlabel('Step', fontsize=11)
ax3a.set_ylabel('True CVaR (bottom 5% mean)', fontsize=11)
ax3a.set_title('(a) Time Series Comparison', fontsize=12, fontweight='bold')
ax3a.legend(fontsize=10, loc='best')
ax3a.grid(True, alpha=0.3)
ax3a.axhline(y=0, color='black', linewidth=0.5)

# 右：バーチャート（記録値 vs 真のCVaR値）
comp_labels_b3 = ['CVaR bins=401', 'CVaR bins=801 (without sums)']
comp_recorded_b3 = []
comp_true_b3 = []

for label in comp_labels_b3:
    if label in results:
        comp_recorded_b3.append(results[label]['recorded_mean'])
        comp_true_b3.append(results[label]['true_mean'])

x = np.arange(len(comp_labels_b3))
width = 0.35

bars1 = ax3b.bar(x - width/2, comp_recorded_b3, width, label='Recorded (Histogram-based)',
                 color='steelblue', edgecolor='black', linewidth=1.5)
bars2 = ax3b.bar(x + width/2, comp_true_b3, width, label='True CVaR (Re-calculated)',
                 color='coral', edgecolor='black', linewidth=1.5)

ax3b.set_xlabel('Configuration', fontsize=11)
ax3b.set_ylabel('Mean CVaR', fontsize=11)
ax3b.set_title('(b) Recorded vs True CVaR Values', fontsize=12, fontweight='bold')
ax3b.set_xticks(x)
ax3b.set_xticklabels(comp_labels_b3, fontsize=10)
ax3b.legend(fontsize=10, loc='best')
ax3b.grid(True, alpha=0.3, axis='y')
ax3b.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('/workspace/RRA/portfolio_cvar/imgs/fig3_bins401_vs_801nosums.png', dpi=150, bbox_inches='tight')
print("Saved: fig3_bins401_vs_801nosums.png")
plt.close()

print("\n✓ All 3 figures generated successfully!")
