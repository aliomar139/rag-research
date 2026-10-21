"""Export release figures from completed measurements. No retrieval or training."""
import csv
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/submission/assets'
COLORS = dict(re.findall(r'--color-([\w-]+):\s*(#[0-9a-fA-F]{6})', (ROOT/'tokens.css').read_text()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for filename in ('IBMPlexSans-Regular.ttf', 'IBMPlexSans-SemiBold.ttf', 'Newsreader-400.ttf'):
        font_manager.fontManager.addfont(OUT/'fonts'/filename)
    plt.rcParams.update({'font.family': 'IBM Plex Sans', 'font.size': 12,
                         'text.color': COLORS['ink'], 'axes.labelcolor': COLORS['muted'],
                         'xtick.color': COLORS['muted'], 'ytick.color': COLORS['ink'],
                         'svg.fonttype': 'path', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.spines.left': False,
                         'axes.spines.bottom': False, 'savefig.facecolor': COLORS['paper']})
    with (ROOT/'docs/figures/all_measured_speedups.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    order = ['static_rrf', 'ridge11_r99', 'ridge11_r95', 'ridge11_r92', 'ridge11_r90']
    for dataset, title in [('trec-covid', 'TREC-COVID: all 50 queries'), ('squad', 'SQuAD: fixed 256-question sample')]:
        selected = {r['method']: r for r in rows if r['dataset']==dataset}
        fig, axes = plt.subplots(1, 2, figsize=(11.5,4.7), gridspec_kw={'wspace':.65})
        fig.set_facecolor(COLORS['paper'])
        for ax, reference, caption in zip(axes, ['speedup_vs_original_fusion','speedup_vs_indexed_fusion'],
                                         ['vs our original fusion', 'vs equally indexed fusion']):
            ax.set_facecolor(COLORS['paper'])
            values = [float(selected[m][reference]) for m in order]
            labels = ['Indexed fusion', 'r99 · primary', 'r95 · secondary', 'r92 · secondary', 'r90 · secondary']
            for i,(method,value) in enumerate(zip(order, values)):
                retention = float(selected[method]['retention'])*100
                fill = COLORS['good'] if method=='ridge11_r99' else COLORS['neutral']
                hatch = '///' if retention<90 else None
                ax.barh(i,value,height=.43,color=fill,hatch=hatch,edgecolor=COLORS['ink'],linewidth=.2)
                ax.text(value+max(values)*.025,i, f'{value:.2f}x',va='center',fontsize=11,fontweight='bold')
            ax.set_yticks(range(5),labels)
            ax.invert_yaxis()
            ax.set_xlim(0,max(values)*1.30)
            ax.set_ylim(4.65,-.7)
            ax.set_title(caption,loc='left',fontsize=13,fontweight='bold',pad=16)
            ax.tick_params(axis='both',length=0,pad=8)
            ax.grid(axis='x',color=COLORS['rule'],alpha=.4,linewidth=.5)
            ax.set_axisbelow(True)
            ax.set_xlabel('Speedup (x)',fontsize=11,labelpad=10)
        fig.suptitle(title,x=.04,y=.98,ha='left',fontsize=16,fontweight='bold')
        ret=' / '.join(f'{float(selected[m]["retention"])*100:.2f}%' for m in order)
        fig.text(.04,.025, 'Recall retained, in row order: '+ret,fontsize=11,color=COLORS['muted'])
        fig.subplots_adjust(left=.17,right=.98,top=.80,bottom=.18)
        for extension in ('png','svg'):
            fig.savefig(OUT/f'speedup_{dataset}.{extension}',dpi=180)
        plt.close(fig)
    # A technical pipeline figure for the paper and HTML, not a decorative asset.
    fig,ax=plt.subplots(figsize=(11.5,3.5))
    fig.set_facecolor(COLORS['paper']); ax.set_facecolor(COLORS['paper'])
    ax.set_xlim(0,11.5); ax.set_ylim(0,3.5); ax.axis('off')
    nodes=[(.2,1.3,1.35,.8,'Query'),(2.0,1.3,2,.8,'Indexed\nBest Fields BM25'),
           (4.5,1.3,2,.8,'11-feature\nRidge gate'),(7.2,2.2,3.9,.8,'Low predicted gain\nReturn BM25 top 100'),
           (7.2,.25,3.9,1.15,'Higher predicted gain\nDense + SPLADE\nRRF with existing BM25')]
    for x,y,w,h,label in nodes:
        ax.add_patch(Rectangle((x,y),w,h,facecolor=COLORS['paper'],edgecolor=COLORS['rule'],linewidth=1))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=12)
    for start,end in [((1.55,1.7),(2,1.7)),((4,1.7),(4.5,1.7)),((6.5,1.9),(7.2,2.6)),((6.5,1.5),(7.2,.85))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,color=COLORS['good'],linewidth=1.2))
    fig.tight_layout(pad=.25)
    for extension in ('png','svg'):
        fig.savefig(OUT/f'pipeline.{extension}',dpi=180)
    plt.close(fig)
    print('Publication figures exported from recorded CSV values.')


if __name__ == '__main__':
    main()
