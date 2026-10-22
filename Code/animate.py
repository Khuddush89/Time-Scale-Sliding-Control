"""Generate original GIFs. Discrete states are markers, never interpolated curves."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parent))
from tsmc.control import known_value
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'assets';OUT.mkdir(exist_ok=True)
BG='#0b1224';INK='#e5efff';CYAN='#39d9ef';PINK='#ff6caa';GOLD='#ffce70'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK,'axes.labelcolor':INK,'xtick.color':'#98aeca','ytick.color':'#98aeca','axes.edgecolor':'#31425d','axes.facecolor':BG,'figure.facecolor':BG,'savefig.facecolor':BG})
def radii(t,r=1.):
    out=[r]
    for h in np.diff(t):out.append(known_value(out[-1],h,.2,1.))
    return np.array(out)
def frame(fig):
    fig.canvas.draw();return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]).convert('P',palette=Image.Palette.ADAPTIVE,colors=128)
def save(images,name):images[0].save(OUT/name,save_all=True,append_images=images[1:],duration=[90]*(len(images)-1)+[1700],loop=0,optimize=True)
def style(ax):
    ax.grid(alpha=.14);ax.spines[['top','right']].set_visible(False)
def comparison():
    times=[np.r_[0,.5*np.cumsum(1/np.arange(1,55))],np.sqrt(np.arange(55)),1.12**np.arange(30)]
    titles=['HARMONIC  /  shrinking gaps','SQUARE ROOT  /  vanishing radius','GEOMETRIC  /  growing gaps']
    fig,axs=plt.subplots(1,3,figsize=(12,3.9),dpi=85);fig.subplots_adjust(left=.06,right=.98,bottom=.18,top=.76,wspace=.31)
    fig.suptitle('GRAININESS SHAPES THE ROBUST RADIUS',fontsize=17,fontweight='bold',y=.96)
    arts=[]
    for ax,t,title,col in zip(axs,times,titles,[CYAN,PINK,GOLD]):
        r=radii(t,0 if title.startswith('GEOM') else 1);ax.set_title(title,fontsize=10,color=col);style(ax);ax.set(xlim=(t[0]-.05,t[-1]*1.03),ylim=(-.025,max(r)*1.12),xlabel='time-scale coordinate t');pts=ax.scatter([],[],s=19,c=col);arts.append((pts,t,r));
    axs[0].set_ylabel('worst-case radius')
    imgs=[]
    for k in range(42):
        for pts,t,r in arts:
            n=max(1,int(len(t)*(k+1)/42));pts.set_offsets(np.c_[t[:n],r[:n]])
        imgs.append(frame(fig))
    save(imgs,'graininess.gif');plt.close(fig)
def accumulation():
    t=1-.72**np.arange(42);r=radii(t,.5)
    fig,ax=plt.subplots(figsize=(10,3.8),dpi=90);fig.subplots_adjust(bottom=.2,top=.8);style(ax)
    ax.set_title('INFINITELY MANY JUMPS · A FINITE ACCUMULATION POINT',fontsize=14,fontweight='bold',pad=18)
    ax.set(xlim=(-.03,1.06),ylim=(-.02,.56),xlabel='t_k = 1 − 0.72^k',ylabel='worst-case radius')
    ax.axvline(1,color=GOLD,ls=':',alpha=.7);ax.scatter([1],[0],marker='*',s=130,color=GOLD,label='exact limit (1, 0)');ax.legend(facecolor=BG,labelcolor=INK,edgecolor='#31425d')
    dots=ax.scatter([],[],c=CYAN,s=28);imgs=[]
    for k in range(1,43):dots.set_offsets(np.c_[t[:k],r[:k]]);imgs.append(frame(fig))
    save(imgs,'accumulation.gif');plt.close(fig)
def mixed():
    fig,ax=plt.subplots(figsize=(10,3.8),dpi=90);fig.subplots_adjust(bottom=.2,top=.8);style(ax)
    ax.set_title('MIXED TIME SCALE · FLOW ON INTERVALS, JUMPS ACROSS GAPS',fontsize=13,fontweight='bold',pad=18)
    ax.set(xlim=(-.1,4.1),ylim=(-.025,.83),xlabel='t in union [n, n + 0.6]',ylabel='worst-case radius')
    arcs=[];r=.8
    for n in range(4):
        x=np.linspace(n,n+.6,36);y=np.maximum(0,r-.8*(x-n));line,=ax.plot([],[],color=CYAN,lw=2.5);arcs.append((line,x,y));ax.axvspan(n,n+.6,color=CYAN,alpha=.05);r=max(.08,float(y[-1])-.8*.4)
    imgs=[]
    for k in range(48):
        for j,(line,x,y) in enumerate(arcs):
            count=np.clip((k-j*12+1)*3,0,36);line.set_data(x[:count],y[:count])
        imgs.append(frame(fig))
    save(imgs,'mixed.gif');plt.close(fig)
def header():
    from matplotlib.font_manager import findfont, FontProperties
    font=findfont(FontProperties(family='DejaVu Sans'));bold=findfont(FontProperties(family='DejaVu Sans',weight='bold'))
    imgs=[]
    t=np.r_[0,.5*np.cumsum(1/np.arange(1,75))];r=radii(t)
    for k in range(48):
        im=Image.new('RGB',(1200,390),BG);d=ImageDraw.Draw(im)
        for x in range(0,1200,40):d.line((x,0,x,390),fill='#14213a')
        for y in range(0,390,40):d.line((0,y,1200,y),fill='#14213a')
        d.text((52,36),'DYNAMIC EQUATIONS  /  ROBUST CONTROL',font=ImageFont.truetype(bold,15),fill=CYAN)
        d.text((50,77),'SLIDING MODE',font=ImageFont.truetype(bold,54),fill=INK)
        d.text((50,141),'ON TIME SCALES',font=ImageFont.truetype(bold,54),fill=INK)
        d.text((53,225),'Sharp bands. Timing uncertainty. Exact computations.',font=ImageFont.truetype(font,20),fill='#a9bddd')
        d.text((53,330),'Mahammad Khuddush  •  Mathematical control research',font=ImageFont.truetype(font,17),fill=CYAN)
        d.text((815,45),'HARMONIC TIME SCALE',font=ImageFont.truetype(bold,16),fill=GOLD)
        d.line((805,300,1150,300),fill='#48607f',width=2)
        d.line((805,80,805,300),fill='#48607f',width=2)
        n=1+int(73*(k+1)/48)
        for a,b in zip(t[:n],r[:n]):
            x=815+325*a/t[-1];y=295-200*b;d.ellipse((x-3,y-3,x+3,y+3),fill=CYAN)
        d.text((820,317),'POINTS OF T · NO INTERPOLATION',font=ImageFont.truetype(font,12),fill='#a9bddd')
        imgs.append(im.convert('P',palette=Image.Palette.ADAPTIVE,colors=128))
    save(imgs,'header.gif')
if __name__=='__main__':
    header();comparison();accumulation();mixed();print('Created four original animated assets.')
