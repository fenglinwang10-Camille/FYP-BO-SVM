"""Exact paper-size SVG authored with FigGenie metrics; no bitmap editing."""
from pathlib import Path
import json
import sys
from html import escape

SKILL = Path.home()/'.codex/skills/figgenie-paper-diagram'
sys.path.insert(0, str(SKILL/'scripts'))
from place_text import place, svg_text

OUT = Path(__file__).resolve().parent
INK, DATA, CONTROL, BLUE, GROUND = '#20364B', '#365F82', '#B86B32', '#E8F0F8', '#F7F9FC'
FONT = "'TeX Gyre Heros', 'Helvetica Neue', Helvetica, Arimo, Arial, sans-serif"
nodes = [
    ('recordings','Bonn EEG segments\n(five classes)',(8,8,88,30),None),
    ('split','Segment-level\ndata split',(8,60,88,30),None),
    ('training','Training segments\n(fixed cross-validation)',(8,101,88,38),None),
    ('test','Held-out test\nsegments',(8,174,88,32),None),
    ('candidate','Candidate parameters',(172,48,116,26),'bo'),
    ('preprocessing','Filtering and\nwindowing',(126,106,76,28),'bo'),
    ('features','Feature extraction',(208,106,70,28),'bo'),
    ('svm','Standardisation\nand RBF-SVM',(286,106,78,28),'bo'),
    ('validation','Mean segment-level\nvalidation accuracy',(260,162,104,30),'bo'),
    ('gp-ei','GP update and EI',(140,162,104,30),'bo'),
    ('selected','Selected\nconfiguration',(388,39,84,30),None),
    ('refit','Refit on all\ntraining data',(388,88,84,30),None),
    ('evaluation','Final test\nevaluation',(388,137,84,32),None),
]
edges = [
    ('recordings-split','recordings','split','data',[(52,38.5),(52,59.5)],None),
    ('split-training','split','training','data',[(52,90.5),(52,100.5)],None),
    ('split-test','split','test','data',[(7.5,75),(2,75),(2,190),(7.5,190)],None),
    ('training-preprocessing','training','preprocessing','data',[(96.5,120),(125.5,120)],None),
    ('preprocessing-features','preprocessing','features','data',[(202.5,120),(207.5,120)],None),
    ('features-svm','features','svm','data',[(278.5,120),(285.5,120)],None),
    ('candidate-preprocessing','candidate','preprocessing','control',[(171.5,61),(122,61),(122,93),(158,93),(158,105.5)],('Filter cutoffs and\nwindow length',(129,70,83,21))),
    ('candidate-svm','candidate','svm','control',[(288.5,61),(325,61),(325,105.5)],('C and γ',(330,79,40,13))),
    ('svm-validation','svm','validation','data',[(324,134.5),(324,161.5)],('Majority voting',(254,140,65,13))),
    ('validation-gp','validation','gp-ei','feedback',[(259.5,177),(244.5,177)],None),
    ('next-candidate','gp-ei','candidate','feedback',[(192,192.5),(192,202),(382,202),(382,35),(230,35),(230,47.5)],None),
    ('select','validation','selected','control',[(364.5,177),(372,177),(372,54),(387.5,54)],('After budget\nexhaustion',(322,136,46,25))),
    ('selected-refit','selected','refit','data',[(430,69.5),(430,87.5)],None),
    ('refit-evaluation','refit','evaluation','data',[(430,118.5),(430,136.5)],None),
    ('test-evaluation','test','evaluation','data',[(52,206.5),(52,220),(430,220),(430,168.5)],None),
]

spec = {
 'meta': {'id':'eeg-bo-overview','title':'EEG classification implementation overview',
          'system':{'name':'BO-controlled EEG classification','one_liner':'Training-only parameter selection followed by held-out testing.'},
          'family':'systems','kind':'architecture','figure_kind':'whole-system','column':'double',
          'width_pt':480,'max_height_pt':260,'template':'control-loop','direction':'left-right',
          'palette':'blue-outline','font':'helvetica','language':'en','step_markers':'none','version':1,
          'caption_draft':'Implementation workflow of the Bayesian optimisation-controlled EEG classification pipeline. Solid blue arrows show data processing and evaluation; dashed orange arrows show parameter selection and feedback.',
          'open_questions':[],
          'notes':'A4 thesis width, 169.3 mm. The diagram is a concise implementation overview. Solid blue arrows show processing/evaluation; dashed orange arrows show selection/feedback. Held-out segments remain outside the optimisation loop.'},
 'containers':[{'id':'bo','label':'Bayesian optimisation','kind':'solid','role':'boundary','title_position':'top-center'}],
 'nodes':[dict(id=i,label=t,kind='module',role='existing',**({'parent':p} if p else {})) for i,t,b,p in nodes],
 'edges':[dict(id=i,**{'from':a,'to':b},kind=k,style='dashed' if k in ('control','feedback') else 'solid',weight='main' if i in ('preprocessing-features','features-svm','svm-validation') else 'normal',head='filled',route='orthogonal',**({'label':l[0]} if l else {})) for i,a,b,k,pts,l in edges],
 'legend':{'mode':'explicit','placement':'below','title':'','items':[{'id':'data-flow','label':'Data processing and evaluation','swatch':{'kind':'line','role':'stroke','line_style':'solid','head':'filled'}},{'id':'parameter-flow','label':'Parameter selection and feedback','swatch':{'kind':'line','role':'accent','line_style':'dashed','head':'filled'}}]},
 'notes':'Solid blue arrows show data processing and evaluation. Dashed orange arrows show parameter selection and feedback. Held-out test segments do not enter the BO loop.'
}
(OUT/'spec.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')

parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="480pt" height="260pt" viewBox="0 0 480 260" font-family="{escape(FONT,quote=True)}" font-size="8" fill="{INK}" data-spec="eeg-bo-overview" data-spec-version="1">',
         f'<defs><marker id="arrow-data" markerWidth="5" markerHeight="5" refX="4.7" refY="2.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L5 2.5L0 5Z" fill="{DATA}"/></marker><marker id="arrow-control" markerWidth="5" markerHeight="5" refX="4.7" refY="2.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L5 2.5L0 5Z" fill="{CONTROL}"/></marker></defs>',
         f'<g id="c:bo" data-kind="container" data-role="boundary"><rect x="118" y="8" width="250" height="202" fill="{GROUND}" stroke="#A7B7C7" stroke-width="0.6"/>'+svg_text(place('Bayesian optimisation',(145,12,196,17),size=9,font='helvetica',bold=True),attrs='font-weight="bold"')+'</g>']

for i,a,b,k,pts,label in edges:
    d = 'M'+' L'.join(f'{x:g} {y:g}' for x,y in pts)
    c = DATA if k == 'data' else CONTROL
    dash = '' if k == 'data' else ' stroke-dasharray="3 2"'
    marker = 'arrow-data' if k == 'data' else 'arrow-control'
    parts.append(f'<g id="e:{i}" data-kind="edge" data-from="{a}" data-to="{b}" data-edge-kind="{k}"><path d="{d}" fill="none" stroke="{c}" stroke-width="{0.8 if k == "data" else 0.75}"{dash} marker-end="url(#{marker})"/>')
    if label:
        p=place(label[0],label[1],size=7,font='helvetica')
        assert p['fits'],(i,p['short'])
        parts.append(svg_text(p))
    parts.append('</g>')

for i,text,rect,parent in nodes:
    x,y,w,h=rect
    fill=BLUE if i in ('candidate','gp-ei') else '#FFFFFF'
    p=place(text,rect,size=8,font='helvetica')
    assert p['fits'],(i,p['short'])
    par=f' data-parent="{parent}"' if parent else ''
    parts.append(f'<g id="n:{i}" data-kind="node" data-role="existing"{par}><rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{DATA}" stroke-width="0.65"/>'+svg_text(p)+'</g>')
parts.append(f'<g id="legend" data-kind="legend"><path d="M150 245 H178" fill="none" stroke="{DATA}" stroke-width="0.8" marker-end="url(#arrow-data)"/>'+svg_text(place('Data processing and evaluation',(184,238,103,15),size=7,font='helvetica'))+f'<path d="M305 245 H333" fill="none" stroke="{CONTROL}" stroke-width="0.75" stroke-dasharray="3 2" marker-end="url(#arrow-control)"/>'+svg_text(place('Parameter selection and feedback',(339,238,132,15),size=7,font='helvetica'))+'</g>')
parts.append('</svg>')
(OUT/'fig.svg').write_text('\n'.join(parts)+'\n',encoding='utf-8')
(OUT/'caption.txt').write_text(spec['meta']['caption_draft']+'\n',encoding='utf-8')
print('Wrote brief-aligned spec and editable 480 x 260 pt SVG.')
