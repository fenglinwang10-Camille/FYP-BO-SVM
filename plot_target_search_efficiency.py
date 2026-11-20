"""Exploratory target-attainment curves, retaining searches that miss a target."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures/chapter5/search_efficiency'
TARGETS=(87.,88.,89.)
RUNS={
    'original':('Original tie rule','10_nested_cv_shared_initialization'),
    'score_tiebreak':('Decision-score tie rule','13_nested_score_tiebreak'),
}
COLORS={'bo':'#096398','rs':'#AD4E24'}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    curves,rows={},[]
    for protocol,(_,directory) in RUNS.items():
        run=ROOT/'results'/directory/'split_42_inner_1729/search_42'
        for fold in range(1,6):
            frames={m:pd.read_csv(run/f'outer_fold_{fold}'/f'{m}_history.csv',float_precision='round_trip')
                    for m in ['bo','rs']}
            np.testing.assert_allclose(frames['bo'].score.iloc[:16],frames['rs'].score.iloc[:16],rtol=0,atol=1e-12)
            for method,frame in frames.items():
                assert frame.iteration.tolist()==list(range(1,51))
                best=np.maximum.accumulate(frame.score.to_numpy())*100
                for target in TARGETS:
                    reached=best>=target-1e-9
                    hits=np.flatnonzero(reached)
                    first=int(hits[0]+1) if len(hits) else None
                    rows.append(dict(protocol=protocol,outer_fold=fold,method=method,
                                     target_percent=target,reached=first is not None,
                                     first_evaluation=first,budget=50,
                                     not_reached_note='' if first is not None else 'Not reached within 50; right-censored'))
                    curves[protocol,fold,method,target]=reached
    detail=pd.DataFrame(rows)
    detail['first_evaluation']=detail.first_evaluation.astype('Int64')
    detail.to_csv(OUT/'evaluations_to_target_per_fold.csv',index=False,na_rep='')
    table=detail.pivot(index=['protocol','outer_fold'],columns=['target_percent','method'],values='first_evaluation')
    table.to_csv(OUT/'evaluations_to_target_table.csv',na_rep='Not reached')
    count_rows=[]
    for protocol in RUNS:
        for target in TARGETS:
            for method in ['bo','rs']:
                count_rows.append(dict(protocol=protocol,target_percent=target,method=method,
                    reached_folds=int(sum(curves[protocol,k,method,target][-1] for k in range(1,6))),total_folds=5))
    counts=pd.DataFrame(count_rows)
    counts.to_csv(OUT/'target_attainment_counts.csv',index=False)

    def draw(ax,protocol,target):
        for method in ['bo','rs']:
            number=np.stack([curves[protocol,k,method,target] for k in range(1,6)]).sum(axis=0)
            locations=[i for i in range(50) if i+1 in ([1,16,30,50] if method=='bo' else [10,20,40])]
            ax.step(np.arange(1,51),number,where='post',color=COLORS[method],
                    linestyle='-' if method=='bo' else (0,(4,3)),linewidth=1.8,
                    marker='s' if method=='bo' else 'o',markevery=locations,markersize=3.5,
                    markerfacecolor='white',zorder=3 if method=='bo' else 4)
        ax.axvline(16.5,color='#67707A',linestyle=(0,(3,3)),linewidth=.8)
        final={m:sum(curves[protocol,k,m,target][-1] for k in range(1,6)) for m in ['bo','rs']}
        ax.set_title(f'Target {target:.0f}% | BO {final["bo"]}/5, RS {final["rs"]}/5',loc='left',fontsize=9)
        ax.set(xlim=(.5,50.8),ylim=(-.2,5.4),xticks=[1,16,30,50],yticks=range(6))
        ax.grid(axis='y',color='#DFE6EC',linewidth=.6)
        ax.set_axisbelow(True)

    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8,
                         'axes.spines.top':False,'axes.spines.right':False}):
        handles=[Line2D([],[],color=COLORS['bo'],label='BO',marker='s',markerfacecolor='white'),
                 Line2D([],[],color=COLORS['rs'],label='RS',linestyle='--',marker='o',markerfacecolor='white')]
        fig,axes=plt.subplots(2,3,figsize=(9.3,5.1),sharex=True,sharey=True,layout='constrained')
        for row,(protocol,(title,_)) in enumerate(RUNS.items()):
            for col,target in enumerate(TARGETS):
                draw(axes[row,col],protocol,target)
                if col==0:axes[row,col].set_ylabel(f'{title}\nFolds reaching target')
                if row==1:axes[row,col].set_xlabel('Candidate evaluations')
        fig.legend(handles=handles,loc='outside lower center',ncols=2,frameon=False)
        fig.savefig(OUT/'target_attainment_comparison.png',dpi=320,facecolor='white')
        plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(9.3,2.8),sharey=True,layout='constrained')
        for ax,target in zip(axes,TARGETS):
            draw(ax,'score_tiebreak',target)
            ax.set_xlabel('Candidate evaluations')
        axes[0].set_ylabel('Folds reaching target')
        fig.legend(handles=handles,loc='outside lower center',ncols=2,frameon=False)
        fig.savefig(OUT/'target_attainment_score_tiebreak.png',dpi=320,facecolor='white')
        plt.close(fig)
    print(table.to_string(na_rep='Not reached'))
    print(counts.to_string(index=False))
    print('Saved PNG and CSV files:',OUT)


if __name__=='__main__':
    main()
