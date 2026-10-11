"""Publication-style static figures from the reviewer's verified saved results.
No project imports, labels, training, or network.
"""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=True)
    e=json.loads((a.results/'recomputed_endpoints.json').read_text())
    colors={'LOGIT0.1':'#64748b','R0':'#2864c7','S':'#148269'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    comparisons=('R0-LOGIT0.1','S-R0','S-LOGIT0.1')
    fig,axes=plt.subplots(1,3,figsize=(13.4,4.1))
    titles=('Final old domains (O)','New-domain post-stage level (N)','Final latest domain (Z)')
    for ax,ep,title in zip(axes,('O','N','Z'),titles):
        for j,name in enumerate(comparisons):
            v=e['difference'][name]['primary'][ep]['map'];mean=v['mean']*100;lo,hi=[x*100 for x in v['conditional_95pct_interval']]
            color='#148269' if lo>0 else '#64748b'
            ax.errorbar(mean,j,xerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=4,lw=2,markersize=6)
            ax.annotate(f'{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]',(mean,j),xytext=(0,13),textcoords='offset points',ha='center',fontsize=8)
        ax.axvline(0,color='#9ca3af',lw=1,ls='--');ax.set_yticks(range(3),comparisons);ax.set_ylim(2.55,-.6)
        ax.set_title(title,fontsize=11,pad=14);ax.set_xlabel('MAP difference (percentage points)');ax.grid(axis='x',alpha=.17)
        ax.margins(x=.15)
    fig.suptitle('Three-arm attribution: means and conditional 95% intervals',fontsize=14,y=.98)
    fig.text(.5,.018,'Fixed s0 and three cyclic orders. Resampling pairs whole groups within actual A/B/C domains; no overall pass gate.',ha='center',fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,.06,1,.92));fig.savefig(a.out/'map_attribution.png',dpi=175);plt.close(fig)
    rows=list(csv.DictReader((a.results/'recomputed_curves.csv').open(encoding='utf-8-sig')))
    by={(r['point'],r['role'],r['domain'],r['metric']):r for r in rows}
    panels=[]
    for order in ('ABC','BCA','CAB'):
        panels.extend([(order,1,'fit',order[0]),(order,2,'fit',order[1]),(order,2,'cache',order[0]),
                       (order,3,'fit',order[2]),(order,3,'cache',order[0]),(order,3,'cache',order[1])])
    names={'map':'MAP','recall_at_5':'Recall@5','log_loss':'Raw log loss'}
    for metric,label in names.items():
        fig,axes=plt.subplots(6,3,figsize=(13,18),sharex=True,sharey=True)
        maxval=max(float(r[f'{role}_epoch{i}']) for r in rows if r['metric']==metric for role in ('train','valid') for i in range(1,7))
        for ax,(o,s,role,d) in zip(axes.flat,panels):
            first=by.get((f'{o}_R0_stage{s}',role,d,metric))
            if first is None:
                ax.set_facecolor('#f1f5f9');ax.text(.5,.5,'N/A: no surviving C cache groups',ha='center',va='center',transform=ax.transAxes,fontsize=10)
                ax.set_title(f'{o} / stage {s} / {role} {d}',fontsize=10);continue
            for arm in ('LOGIT0.1','R0','S'):
                r=by[f'{o}_{arm}_stage{s}',role,d,metric]
                for pop,style in (('train','--'),('valid','-')):
                    vals=[float(r[f'{pop}_epoch{i}']) for i in range(1,7)]
                    ax.plot(range(1,7),vals,style,color=colors[arm],lw=1.6,marker='o' if pop=='valid' else None,markersize=3,alpha=.95)
            ax.set_title(f'{o} / stage {s} / {role} {d} (n={first["train_groups"]}; valid=20)',fontsize=10)
            ax.set_xticks(range(1,7));ax.grid(alpha=.16);ax.set_ylim(0,1 if metric!='log_loss' else maxval*1.08)
        for ax in axes[-1]:ax.set_xlabel('Epoch (fixed 1 to 6)')
        for ax in axes[:,0]:ax.set_ylabel(label)
        legend=[Line2D([0],[0],color=c,lw=2,label=arm) for arm,c in colors.items()]
        legend += [Line2D([0],[0],color='#334155',lw=2,ls='--',label='Fit / surviving cache'),Line2D([0],[0],color='#334155',lw=2,label='Same-domain valid')]
        fig.suptitle(f'All six-epoch curves: {label}',fontsize=16,y=.994)
        fig.legend(handles=legend,ncol=5,loc='upper center',bbox_to_anchor=(.5,.98),frameon=False,fontsize=10)
        fig.text(.5,.013,'All fixed stage/domain panels shown. First stages have no cache. No best-epoch selection; no absence-of-overfitting claim.',ha='center',fontsize=10,color='#475569')
        fig.tight_layout(rect=(0,.03,1,.96));fig.savefig(a.out/f'all_six_epoch_{metric}.png',dpi=150);plt.close(fig)
    print('Created four figures from independently recomputed results.')


if __name__=='__main__':main()
