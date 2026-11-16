"""Draw the frozen primary contrast and all prescribed old-domain seed curves."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

from audit_panel_alignment import require_matplotlib_panel_alignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    blob = args.summary.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == 'f187d1e29351e7321ab5ab534bff38eb2704adf0156b6c491b4251372dbfa30b'
    s = json.loads(blob)
    args.out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .65, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})

    def save(fig, name, multi=False):
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=str(args.out/f'{name}.alignment.json'),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=multi, strict=True)
        for ext in ('pdf', 'svg', 'png'):
            fig.savefig(args.out/f'{name}.{ext}', dpi=300, facecolor='white')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(183/25.4, 90/25.4))
    fig.subplots_adjust(left=.29, right=.96, top=.91, bottom=.22)
    fields = ['O', 'F', 'A_acq', 'N', 'Z']
    labels = ['Old-domain final O', 'Forgetting F', 'Acquisition A_acq', 'New-domain learning N', 'Last-domain final Z']
    for i, field in enumerate(fields):
        row = s['contrasts']['Full-ER']['raw'][field]['map']
        point = 100*row['mean']; low, high = [100*x for x in row['conditional_interval']]
        color = '#3B6FB6' if field in ('O', 'F') else '#737373'
        ax.errorbar(point, i, xerr=[[point-low], [high-point]], fmt='o', color=color,
                    markersize=4.5, capsize=3, elinewidth=1.2, capthick=.9)
    ax.axvline(0, color='#B0B0B0', lw=.8, ls='--', zorder=0)
    ax.set_yticks(range(5), labels)
    ax.set_ylim(4.6, -.6)
    ax.set_xlabel('Full minus ER (percentage points)')
    ax.set_title('Old-domain gains with a small negative acquisition point', loc='left', pad=10)
    ax.set_xlim(-3.7, 3.7)
    fig.text(.29, .055, 'Points: nine fixed paths. Bars: 95% paired group-conditional intervals.\nLower F is better; higher O, A_acq, N and Z is better.', fontsize=7, ha='left')
    save(fig, 'primary_full_er')

    colors = {'Full': '#3B6FB6', 'ER': '#777777', 'KD': '#2A8A7B', 'SEQ': '#BE7A42'}
    markers = {4: 'o', 5: 's', 6: '^'}
    styles = {4: '-', 5: '--', 6: ':'}
    fig, axes = plt.subplots(2, 3, figsize=(183/25.4, 145/25.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.97, bottom=.18, top=.80, hspace=.52, wspace=.23)
    mi = s['metric_columns'].index('map')
    source_rows = []
    for col, order in enumerate(('ABC', 'BCA', 'CAB')):
        for row, position in enumerate((1, 2)):
            ax = axes[row, col]
            for arm in ('SEQ', 'ER', 'KD', 'Full'):
                for seed in (4, 5, 6):
                    stages = list(range(position, 4))
                    values = [100*s['per_cell'][f'{order}_s{seed}_{arm}_raw_{stage}_{position}']['mean'][mi] for stage in stages]
                    ax.plot(stages, values, color=colors[arm], marker=markers[seed], ls=styles[seed],
                            lw=1.15 if arm == 'Full' else .85, markersize=3, alpha=.9)
                    source_rows.extend({'order': order, 'domain': order[position-1], 'position': position,
                                        'method': arm, 'seed': seed, 'stage': stage, 'raw_map_percent': value}
                                       for stage, value in zip(stages, values))
            ax.set_title(f'{order}: old domain {order[position-1]}', loc='left', pad=10)
            label = chr(ord('a') + row*3 + col)
            transform = ax.transAxes + ScaledTranslation(-18/72, 11/72, fig.dpi_scale_trans)
            ax.text(0, 1, label, transform=transform, fontsize=8, fontweight='bold', ha='left', va='bottom')
            ax.set_xticks([1, 2, 3])
            ax.set_xlim(.90, 3.10)
            ax.grid(axis='y', color='#E9E9E9', lw=.5)
            if col == 0:
                ax.set_ylabel('Raw MAP (%)')
            if row == 1:
                ax.set_xlabel('Training stage')
    values = [row['raw_map_percent'] for row in source_rows]
    import math
    axes[0, 0].set_ylim(math.floor((min(values)-1)/5)*5, math.ceil((max(values)+1)/5)*5)
    handles = [Line2D([0],[0],color=colors[a],lw=1.5,label=a) for a in ('Full','ER','KD','SEQ')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53,.965), ncol=4, frameon=False)
    handles = [Line2D([0],[0],color='#555555',ls=styles[x],marker=markers[x],markersize=3,label=f'Seed {x}') for x in (4,5,6)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53,.91), ncol=3, frameon=False)
    fig.text(.10,.045,'Each line is one fixed seed; each point averages 40 confirmation groups.\nAll three orders and both old domains are shown. No smoothing or new resampling.',fontsize=7,ha='left')
    assert len(source_rows) == 180
    (args.out/'stage_curve_source.json').write_text(json.dumps({'summary_sha256':hashlib.sha256(blob).hexdigest(),'rows':source_rows},indent=2)+'\n')
    save(fig, 'old_domain_seed_curves', multi=True)


if __name__ == '__main__':
    main()
