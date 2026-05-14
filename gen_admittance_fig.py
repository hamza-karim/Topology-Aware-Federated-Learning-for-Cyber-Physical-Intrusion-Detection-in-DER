"""
Generate IEEE-style cross-zone admittance mismatch figure.
SIDE-BY-SIDE DOUBLE COLUMN VERSION (1x2 Layout)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from datetime import timedelta

# ── Data Loading ──────────────────────────────────────────────────────────────
z2 = pd.read_csv('IDS DATASET/FL_DATA/zone2_test_stealthy.csv', parse_dates=['timestamp'])
z3 = pd.read_csv('IDS DATASET/FL_DATA/zone3_test_stealthy.csv', parse_dates=['timestamp'])

CTX0, CTX1 = 2664, 2808   # 12-hour context window
ATK0, ATK1 = 2724, 2747   # attack block indices (inclusive)

s2  = z2.iloc[CTX0:CTX1 + 1].reset_index(drop=True)
s3  = z3.iloc[CTX0:CTX1 + 1].reset_index(drop=True)
ts  = s2['timestamp']

t0   = ts.iloc[0]
hours = [(t - t0).total_seconds() / 3600 for t in ts]

atk_h0 = (z2.iloc[ATK0]['timestamp'] - t0).total_seconds() / 3600
atk_h1 = (z2.iloc[ATK1]['timestamp'] - t0).total_seconds() / 3600

pre_idx = slice(0, ATK0 - CTX0)
v17_pre = s2['V_bus17'].iloc[pre_idx].mean()
v18_pre = s3['V_bus18'].iloc[pre_idx].mean()

# ── Style Configuration ───────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Serif',
    'font.size':        8,
    'axes.linewidth':   0.6,
    'grid.linewidth':   0.4,
    'grid.alpha':       0.5,
    'lines.linewidth':  1.2,
})

COLORS = {
    'z2':  '#d62728',   # Red  — Zone 2 (Compromised)
    'z3':  '#1f77b4',   # Blue — Zone 3 (Real-time)
    'atk': '#fdbe85',   # Orange fill — Attack window
}

# Expand figsize to 7.16 inches (IEEE standard double-column width)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2))

# ── (a) Voltage (Left Panel) ──────────────────────────────────────────────────
ax1.axvspan(atk_h0, atk_h1, color=COLORS['atk'], alpha=0.4, zorder=0)

line_v17, = ax1.plot(hours, s2['V_bus17'], color=COLORS['z2'], label='Bus 17 (Replayed Data)')
line_v18, = ax1.plot(hours, s3['V_bus18'], color=COLORS['z3'], ls='--', label='Bus 18 (Real Data)')

ax1.set_ylabel('Voltage (p.u.)')
# ax1.set_xlabel('Time of day, 15 Jan 2026')
ax1.set_ylim(0.945, 1.01) 
ax1.grid(True, which='major')
ax1.spines[['top', 'right']].set_visible(False)

mid_h = (atk_h0 + atk_h1) / 2
v17_atk = s2['V_bus17'].iloc[ATK0 - CTX0 : ATK1 - CTX0 + 1].mean()
v18_atk = s3['V_bus18'].iloc[ATK0 - CTX0 : ATK1 - CTX0 + 1].mean()

ax1.annotate(
    f'$\\Delta V={abs(v18_atk - v17_atk)*1000:.1f}$ mp.u.\n(Mismatch)',
    xy=(mid_h, (v17_atk + v18_atk) / 2),
    xytext=(mid_h + 1.5, 0.955), 
    fontsize=8,
    arrowprops=dict(arrowstyle='->', color='#333', lw=0.8, connectionstyle="arc3,rad=-0.2"),
    ha='center',
)
ax1.text(0.02, 0.96, '(a)', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top')

# ── (b) Active Power (Right Panel) ────────────────────────────────────────────
ax2.axvspan(atk_h0, atk_h1, color=COLORS['atk'], alpha=0.4, zorder=0)

# Bus 17 (Forced to bottom half)
ax2.plot(hours, s2['P_bus17'], color=COLORS['z2'])
ax2.set_ylabel(r'$P_{17}$ (p.u.)', color=COLORS['z2'])
ax2.tick_params(axis='y', colors=COLORS['z2'], direction='in')
ax2.set_ylim(-0.55, 0.1) 

# Bus 18 (Forced to top half)
ax2r = ax2.twinx()
ax2r.plot(hours, s3['P_bus18'], color=COLORS['z3'], ls='--')
ax2r.set_ylabel(r'$P_{18}$ (p.u.)', color=COLORS['z3'])
ax2r.tick_params(axis='y', colors=COLORS['z3'], direction='in')
ax2r.set_ylim(-0.02, 0.12) 
ax2r.spines['top'].set_visible(False)

ax2.set_xlabel('Time of day, 15 Jan 2026')
ax2.grid(True, which='major')
ax2.spines[['top', 'right']].set_visible(False)

p17_atk = s2['P_bus17'].iloc[ATK0 - CTX0 : ATK1 - CTX0 + 1].mean()
ax2.annotate(
    f'Target Attack\nTrajectory',
    xy=(mid_h, p17_atk),
    xytext=(mid_h - 2.5, -0.45), 
    fontsize=8,
    arrowprops=dict(arrowstyle='->', color='#333', lw=0.8),
    ha='center',
)
ax2.text(0.02, 0.96, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top')

# ── X-axis Formatting (Applied to both axes) ──────────────────────────────────
for ax in [ax1, ax2]:
    ax.set_xlim(hours[0], hours[-1])
    ax.xaxis.set_major_locator(MultipleLocator(2))
    hour_ticks = [h for h in ax.get_xticks() if hours[0] <= h <= hours[-1]]
    clock_labels = [(t0 + timedelta(hours=h)).strftime('%H:%M') for h in hour_ticks]
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels(clock_labels, rotation=0)

# ── The Master External Legend ────────────────────────────────────────────────
atk_patch = mpatches.Patch(color=COLORS['atk'], alpha=0.4, label='Attack Window')
fig.legend(
    handles=[line_v17, line_v18, atk_patch],
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.08), 
    ncol=3, 
    fontsize=9,
    frameon=False 
)

# ── Save ──────────────────────────────────────────────────────────────────────
fig.tight_layout(pad=0.8, w_pad=2.0)

out_base = 'fig_admittance_mismatch_side_by_side'
fig.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{out_base}.pdf', bbox_inches='tight')
print(f'Saved Side-by-Side Figure!')