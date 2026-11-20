"""Detailed saved-history comparison; flat tails do not establish convergence."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures/chapter5/search_detail'
RUNS={
    'original':('Original tie rule','10_nested_cv_shared_initialization'),
    'score_tiebreak':('Decision-score tie rule','13_nested_score_tiebreak'),
}
COLORS={'bo':'#096398','rs':'#AD4E24'}


def draw(ax,x,values,method,marks=True):
    kw=dict(color=COLORS[method],linewidth=1.7,
            linestyle='-' if method=='bo' else (0,(4,3)),
            zorder=3 if method=='bo' else 4)
    if marks:
        locations=[i for i,v in enumerate(x) if int(v) in
                   ([16,25,35,45,50] if method=='bo' else [20,30,40,48])]
        kw.update(marker='s' if method=='bo' else 'o',markevery=locations,
                  markersize=3.7,markerfacecolor='white',markeredgewidth=.9)
    ax.step(x,values,where='post',**kw)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    history,stats,export={},[],[]
    for protocol,(_,dirname) in RUNS.items():
        run=ROOT/'results'/dirname/'split_42_inner_1729/search_42'
        for fold in range(1,6):
            data={m:pd.read_csv(run/f'outer_fold_{fold}'/f'{m}_history.csv',float_precision='round_trip')
                  for m in ['bo','rs']}
            columns=['f_low','f_high','window_size','C','gamma','score']
            np.testing.assert_allclose(data['bo'][columns].iloc[:16],data['rs'][columns].iloc[:16],rtol=0,atol=1e-12)
            for method,frame in data.items():
                assert frame.iteration.tolist()==list(range(1,51))
                score=frame.score.to_numpy()*100
                best=np.maximum.accumulate(score)
                history[protocol,fold,method]=best
                improving=np.r_[True,np.diff(best)>1e-9]
                last=int(np.flatnonzero(improving)[-1]+1)
                stats.append(dict(protocol=protocol,outer_fold=fold,method=method,
                                  best_at_16=best[15],best_at_50=best[-1],
                                  guided_gain_pp=best[-1]-best[15],last_improvement=last,
                                  evaluations_since_last_improvement=50-last))
                export.extend(dict(protocol=protocol,outer_fold=fold,method=method,evaluation=i+1,
                                   candidate_accuracy=score[i],best_so_far_accuracy=best[i]) for i in range(50))
    stats=pd.DataFrame(stats)
    stats.to_csv(OUT/'search_progress_summary.csv',index=False)
    pd.DataFrame(export).to_csv(OUT/'candidate_and_best_history.csv',index=False)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8,
                         'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(5,2,figsize=(9.3,8.8),sharex=True,layout='constrained')
        x=np.arange(16,51)
        for row,fold in enumerate(range(1,6)):
            all_values=np.concatenate([history[p,fold,m][15:] for p in RUNS for m in ['bo','rs']])
            low=np.floor(all_values.min()*2)/2-.25
            high=np.ceil(all_values.max()*2)/2+.25
            if high-low<1.5:
                center=(high+low)/2
                low,high=center-.75,center+.75
            for col,(protocol,(title,_)) in enumerate(RUNS.items()):
                ax=axes[row,col]
                for method in ['bo','rs']:
                    values=history[protocol,fold,method]
                    draw(ax,x,values[15:],method)
                    item=stats[(stats.protocol==protocol)&(stats.outer_fold==fold)&(stats.method==method)].iloc[0]
                    last=int(item.last_improvement)
                    if last>=17:
                        ax.scatter(last,values[last-1],marker='*',s=65,color=COLORS[method],
                                   edgecolors='white',linewidths=.5,zorder=6)
                bo=history[protocol,fold,'bo'][-1]
                rs=history[protocol,fold,'rs'][-1]
                ax.set_title(f'Fold {fold} | BO {bo:.2f}%, RS {rs:.2f}%',loc='left',fontsize=8.5)
                ax.set(xlim=(15.5,50.8),ylim=(low,high),xticks=[16,20,30,40,50])
                ax.yaxis.set_major_locator(MultipleLocator(.5))
                ax.grid(axis='y',color='#DFE6EC',linewidth=.6)
                ax.set_axisbelow(True)
                if col==0:ax.set_ylabel('Best inner-CV (%)')
                if row==4:ax.set_xlabel('Candidate evaluations')
                if row==0:
                    ax.text(.5,1.32,title,ha='center',transform=ax.transAxes,fontsize=11)
        handles=[Line2D([],[],color=COLORS['bo'],label='BO',marker='s',markerfacecolor='white'),
                 Line2D([],[],color=COLORS['rs'],linestyle='--',label='RS',marker='o',markerfacecolor='white'),
                 Line2D([],[],color='#555555',marker='*',linestyle='none',label='Last improvement after initialisation')]
        fig.legend(handles=handles,loc='outside lower center',ncols=3,frameon=False,fontsize=8)
        fig.savefig(OUT/'five_fold_guided_search_detail.png',dpi=320,facecolor='white')
        plt.close(fig)

        fig,axes=plt.subplots(2,2,figsize=(9.3,5.5),layout='constrained')
        for col,(protocol,(title,_)) in enumerate(RUNS.items()):
            for row,start in enumerate([0,15]):
                ax=axes[row,col]
                x=np.arange(start+1,51)
                for method in ['bo','rs']:
                    values=np.stack([history[protocol,k,method] for k in range(1,6)])
                    mean=values.mean(axis=0)
                    sd=values.std(axis=0,ddof=1)
                    ax.fill_between(x,(mean-sd)[start:],(mean+sd)[start:],step='post',color=COLORS[method],alpha=.10,linewidth=0)
                    draw(ax,x,mean[start:],method)
                ax.set_xlim(start+.5,50.8)
                if row==0:
                    ax.axvline(16.5,color='#67707A',linestyle='--',linewidth=.8)
                    ax.set_title(title,loc='left',fontsize=10)
                    ax.set_ylim(74,90)
                    ax.set_xticks([1,10,16,20,30,40,50])
                else:
                    ax.set_ylim(85.5,90)
                    ax.set_xticks([16,20,30,40,50])
                    ax.yaxis.set_major_locator(MultipleLocator(.5))
                    ax.set_xlabel('Candidate evaluations')
                if col==0:ax.set_ylabel('Mean best inner-CV (%)')
                ax.grid(axis='y',color='#DFE6EC',linewidth=.6)
                ax.set_axisbelow(True)
        fig.legend(handles=handles[:2],loc='outside lower center',ncols=2,frameon=False)
        fig.savefig(OUT/'mean_search_full_and_zoom.png',dpi=320,facecolor='white')
        plt.close(fig)
    print(stats.to_string(index=False))
    print('Saved PNG and CSV files:',OUT)


if __name__=='__main__':
    main()
