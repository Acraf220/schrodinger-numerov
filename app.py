"""
Resolution numerique de l'equation de Schrodinger — Methode de Numerov
Unites atomiques (Hartree) dans tous les systemes.
Systemes : boite infinie | oscillateur harmonique | anharmonique | hydrogene
"""

import math, numpy as np, plotly.graph_objects as go, streamlit as st
from numba import njit

# ── Couleurs et textes theoriques ─────────────────────────────────────────────

PAL = ("#002147","#1F618D","#1E8449","#B7770D","#8E44AD","#B03A2E","#117A65","#784212")

TH = {
"box": "### Particule dans une boite\n$$E_n=\\frac{n^2\\pi^2}{2L^2}\\text{ (Ha)},\\quad\\psi_n=\\sqrt{\\frac{2}{L}}\\sin\\frac{n\\pi x}{L}$$",
"ho" : "### Oscillateur harmonique\n$$V=\\tfrac12\\omega^2x^2,\\quad E_n=\\hbar\\omega(n+\\tfrac12)\\text{ (Ha)}$$",
"anh": "### Oscillateur anharmonique\n$$V=\\tfrac12\\omega^2x^2+\\lambda x^4$$\nReference : perturbations ordre 2 (Ha).",
"H"  : "### Atome d'hydrogene\n$$V(r)=-\\tfrac1r,\\quad E_n=-\\tfrac{1}{2n^2}\\text{ (Ha)}$$",
}

# ── BLOC 1 : Methode de Numerov ───────────────────────────────────────────────
# Resout numeriquement  psi'' = f(x) * psi  point par point sur la grille.
# c0 = h^2/12  ou h est le pas de la grille.

@njit(cache=True)
def numerov(f,c0):
    # Integre psi vers l'avant depuis le bord gauche
    n=f.size; p=np.zeros(n); p[1]=1e-10
    for i in range(1,n-1):
        d=1+c0*f[i+1]
        p[i+1]=(2*p[i]*(1-5*c0*f[i])-p[i-1]*(1+c0*f[i-1]))/d if abs(d)>1e-30 else 0
    return p

@njit(cache=True)
def d5(p,i,dx):
    # Derivee centree d'ordre 5 au point i (utilisee pour le raccord)
    return (p[i-2]-8*p[i-1]+8*p[i+1]-p[i+2])/(12*dx)

@njit(cache=True)
def raccord(E,x,V,c0,h2):
    # Integre depuis les deux bords et retourne la discontinuite au centre.
    # Vaut zero quand E est un autovaleur.
    n=x.size; dx=x[1]-x[0]; m=n//2; f=np.empty(n)
    for i in range(n): f[i]=2*(E-V[i])/h2
    g=numerov(f,c0); d=numerov(f[::-1].copy(),c0)[::-1].copy()
    return d5(g,m,dx)/(g[m] if abs(g[m])>1e-30 else 1e-30)-d5(d,m,dx)/(d[m] if abs(d[m])>1e-30 else 1e-30)

@njit(cache=True)
def bord_puits(E,x,c0):
    # Pour la boite : V=0, donc f=2E partout. Condition : psi(L)=0.
    return numerov(np.full(x.size,2*E),c0)[-1]

# ── BLOC 2 : Recherche de racines (bisection) ─────────────────────────────────
# Trouve l'energie E qui annule la fonction de raccord.
# Etape 1 : balayage grossier pour localiser un changement de signe.
# Etape 2 : bisection fine jusqu'a la tolerance demandee.

def racine(fn,c,tol,a,b,w):
    w=max(w,tol*10)
    for _ in range(9):
        lo,hi=max(a,c-w),min(b,c+w); best=None; e0,f0=lo,fn(lo)
        for i in range(1,2400):
            e=lo+(hi-lo)*i/2399; f=fn(e)
            if np.isfinite(f0) and np.isfinite(f) and abs(f0)<1e8 and abs(f)<1e8 and f0*f<=0:
                s=abs(.5*(e0+e)-c); best=(e0,e,s) if best is None or s<best[2] else best
            e0,f0=e,f
        if best:
            x0,x1=best[0],best[1]; f0=fn(x0)
            for _ in range(120):
                xm=.5*(x0+x1); fm=fn(xm)
                if abs(x1-x0)<tol*1e-3: break
                if f0*fm<=0: x1=xm
                else: x0,f0=xm,fm
            return .5*(x0+x1)
        w*=1.7
    raise RuntimeError("Racine non trouvee")

# ── BLOC 3 : Oscillateur harmonique ──────────────────────────────────────────
# Fonctions propres analytiques phi_n et correction perturbative anharmonique.
# Energies en Hartree : E_n = hbar*omega*(n+1/2).

def hermite(n,x):
    # Polynomes de Hermite par recurrence : H_{n+1} = 2x*H_n - 2n*H_{n-1}
    if n==0: return np.ones_like(x)
    if n==1: return 2*x
    h0,h1=np.ones_like(x),2*x
    for k in range(2,n+1): h0,h1=h1,2*x*h1-2*(k-1)*h0
    return h1

def ho_phi(n,x,w):
    # Fonction propre normalisee de l'oscillateur harmonique
    return (w/math.pi)**.25/math.sqrt((2**n)*math.factorial(n))*np.exp(-w*x*x/2)*hermite(n,np.sqrt(w)*x)

def psi_anh_pert(n,x,w,l):
    # Correction perturbative d'ordre 1 de la fonction propre anharmonique
    B=[ho_phi(k,x,w) for k in range(n+5)]; psi=B[n].copy(); x4=x**4; En=lambda k:w*(k+.5)
    for k in range(n+5):
        if k!=n: psi+=l*np.trapz(B[k]*x4*B[n],x)/(En(n)-En(k))*B[k]
    return psi/math.sqrt(np.trapz(psi*psi,x))

def psi_build(E,x,V,c0,mode,n,h2):
    # Reconstruit psi normalise en raccordant les solutions gauche et droite
    f=2*(E-V)/h2; g=numerov(f,c0); d=numerov(f[::-1].copy(),c0)[::-1].copy(); N=x.size
    if mode=="box": m=max(2,min(N-3,int(round(x[-1]/(2*max(n,1)*(x[1]-x[0]))))))
    else:
        cls=np.where(E-V>0)[0]; c=(int(cls[0])+int(cls[-1]))//2 if cls.size>=4 else N//2
        m=max(2,min(N-3,c if n%2==0 else ((int(cls[0])+c)//2 if cls.size else c)))
    d*=g[m]/(d[m] if abs(d[m])>1e-30 else 1e-30)
    psi=np.empty(N); psi[:m]=g[:m]; psi[m:]=d[m:]
    nrm=math.sqrt(np.trapz(psi*psi,x)); return psi/nrm if nrm>1e-15 else psi

def align(psi,ref): return -psi if float(np.dot(psi,ref))<0 else psi

def Epert(n,w,l):
    # Energie perturbative d'ordre 2 pour l'anharmonique (en Hartree)
    return w*(n+.5)+l*(3/(4*w*w))*(2*n*n+2*n+1)-l*l/(8*w**4)*(34*n**3+51*n*n+59*n+21)

# ── BLOC 4 : Atome d'hydrogene ────────────────────────────────────────────────
# On resout l'equation radiale en posant u(r) = r*R(r).
# Energies en Hartree : E_n = -1/(2n^2).

@njit(cache=True)
def fH(E,r,l): return 2*(E+1/r-l*(l+1)/(2*r*r))

@njit(cache=True)
def Hf(E,r,l,c0,m):
    # Integration vers l'avant de l'equation radiale
    u=np.zeros(m+3); u[1]=r[1]**(l+1)
    for i in range(1,m+2):
        f0,fp,fm=fH(E,r[i],l),fH(E,r[i+1],l),fH(E,r[i-1],l); d=1+c0*fp
        u[i+1]=(2*u[i]*(1-5*c0*f0)-u[i-1]*(1+c0*fm))/d if abs(d)>1e-30 else 0
    return u

@njit(cache=True)
def Hb(E,r,l,c0):
    # Integration vers l'arriere de l'equation radiale
    N=r.size; u=np.zeros(N); u[N-2]=1e-6
    for i in range(N-2,0,-1):
        f0,fp,fm=fH(E,r[i],l),fH(E,r[i+1],l),fH(E,r[i-1],l); d=1+c0*fm
        u[i-1]=(2*u[i]*(1-5*c0*f0)-u[i+1]*(1+c0*fp))/d if abs(d)>1e-30 else 0
    return u

def errH(E,r,l,c0,m):
    # Discontinuite de la derivee logarithmique au raccord
    dr=r[1]-r[0]; g=Hf(E,r,l,c0,m); d=Hb(E,r,l,c0)
    return 1e10 if abs(g[m])<1e-30 or abs(d[m])<1e-30 else d5(g,m,dr)/g[m]-d5(d,m,dr)/d[m]

def matchH(n,l,r):
    # Point de raccord optimal pres du tournant classique
    ro=n*(n+math.sqrt(max(0,n*n-l*(l+1)))); rm=max(2,min(.6*ro,r[-1]*.35))
    return max(4,min(r.size-3,int(round(rm/(r[1]-r[0])))))

def uH(E,r,l,c0,m):
    # Reconstruit u(r) normalise
    g=Hf(E,r,l,c0,m); d=Hb(E,r,l,c0)
    d*=g[m]/(d[m] if abs(d[m])>1e-30 else 1e-30)
    u=np.empty_like(r); u[:m]=g[:m]; u[m:]=d[m:]
    nrm=math.sqrt(np.trapz(u*u,dx=r[1]-r[0])); u=u/nrm if nrm>1e-15 else u
    for i in range(1,u.size):
        if abs(u[i])>1e-8: return -u if u[i]<0 else u
    return u

def Y(l,m,th,ph):
    # Harmoniques spheriques reelles Y_l^m pour la visualisation 3D
    c,s,pi=np.cos(th),np.sin(th),math.pi
    if l==0: return np.full_like(th,.5/math.sqrt(pi))
    if l==1:
        d={0:math.sqrt(3/(4*pi))*c,
           1:math.sqrt(3/(4*pi))*s*np.cos(ph),
          -1:math.sqrt(3/(4*pi))*s*np.sin(ph)}
        return d.get(m,np.zeros_like(th))
    if l==2:
        d={0:.25*math.sqrt(5/pi)*(3*c*c-1),
           1:.5*math.sqrt(15/pi)*s*c*np.cos(ph),
          -1:.5*math.sqrt(15/pi)*s*c*np.sin(ph),
           2:.25*math.sqrt(15/pi)*s*s*np.cos(2*ph),
          -2:.25*math.sqrt(15/pi)*s*s*np.sin(2*ph)}
        return d.get(m,np.zeros_like(th))
    if l==3:
        d={0:.25*math.sqrt(7/pi)*c*(5*c*c-3),
           1:.125*math.sqrt(42/pi)*s*(5*c*c-1)*np.cos(ph),
          -1:.125*math.sqrt(42/pi)*s*(5*c*c-1)*np.sin(ph),
           2:.25*math.sqrt(105/pi)*s*s*c*np.cos(2*ph),
          -2:.25*math.sqrt(105/pi)*s*s*c*np.sin(2*ph),
           3:.125*math.sqrt(70/pi)*s**3*np.cos(3*ph),
          -3:.125*math.sqrt(70/pi)*s**3*np.sin(3*ph)}
        return d.get(m,np.zeros_like(th))
    return np.zeros_like(th)

def interpR(r,u):
    R=np.where(r>1e-10,u/r,0.0); return lambda rr:np.interp(rr,r,R,left=R[0],right=0.0)

def nom_H(n,l,m):
    if l==0: o="s"
    elif l==1: o={0:"pz",1:"px",-1:"py"}[m]
    elif l==2: o={0:"dz2",1:"dxz",-1:"dyz",2:"dx2-y2",-2:"dxy"}[m]
    elif l==3: o={0:"fz3",1:"fxz2",-1:"fyz2",2:"fz(x2-y2)",-2:"fxyz",3:"fx(x2-3y2)",-3:"fy(3x2-y2)"}[m]
    else: o=f"l{l},m{m}"
    return f"{n}{o} (m={m}, l={l}, n={n})"

# ── BLOC 5 : Solveurs (un par systeme) ───────────────────────────────────────
# Chaque solveur construit la grille, cherche les energies avec racine(),
# reconstruit psi, et retourne une liste de dictionnaires de resultats.

def solve_box(L,n_max,tol,N):
    # Boite infinie : V=0 dans [0,L], energies en Hartree
    x=np.linspace(0,L,N); V=np.zeros_like(x); c0=(x[1]-x[0])**2/12
    Eu=((n_max+2)**2)*math.pi**2/(2*L*L)*1.5; S=[]
    for i in range(n_max):
        n=i+1; Ea=n*n*math.pi**2/(2*L*L)
        E=racine(lambda e:bord_puits(e,x,c0),Ea,tol,.01,Eu,max(Ea*.35,.5))
        ref=math.sqrt(2/L)*np.sin(n*math.pi*x/L)
        S.append(dict(n=n,label=f"n={n}",E=E,Ea=Ea,psi=align(psi_build(E,x,V,c0,"box",n,1.0),ref),psi_ref=ref,color=PAL[i%len(PAL)]))
    return x,V,S

def solve_ho(omega,x_max,n_max,tol,N,hbar=1.0):
    # Oscillateur harmonique : E_n = hbar*omega*(n+1/2) en Hartree
    x=np.linspace(-min(x_max,28),min(x_max,28),N); V=.5*omega*omega*x*x
    c0=(x[1]-x[0])**2/12; fn=lambda E:raccord(E,x,V,c0,hbar*hbar); Eu=hbar*omega*(n_max+2); S=[]
    for i in range(n_max+1):
        Ea=hbar*omega*(i+.5); E=racine(fn,Ea,tol,hbar*omega*.02,Eu,.72*hbar*omega)
        ref=ho_phi(i,x,omega)
        S.append(dict(n=i,label=f"n={i}",E=E,Ea=Ea,psi=align(psi_build(E,x,V,c0,"ho",i,hbar*hbar),ref),psi_ref=ref,color=PAL[i%len(PAL)]))
    return x,V,S

def solve_anh(omega,lamb,x_max,n_max,tol,N,hbar=1.0):
    # Oscillateur anharmonique : V = omega^2*x^2/2 + lambda*x^4
    x=np.linspace(-min(x_max,28),min(x_max,28),N); V=.5*omega*omega*x*x+lamb*x**4
    c0=(x[1]-x[0])**2/12; fn=lambda E:raccord(E,x,V,c0,hbar*hbar)
    Eu=hbar*omega*(n_max+2)+lamb*(x_max*.7)**4; S=[]
    for i in range(n_max+1):
        Ea=Epert(i,omega,lamb); E=racine(fn,Ea,tol,hbar*omega*.02,Eu,max(.85*hbar*omega,.35*abs(Ea)+.25))
        ref=psi_anh_pert(i,x,omega,lamb)
        S.append(dict(n=i,label=f"n={i}",E=E,Ea=Ea,psi=align(psi_build(E,x,V,c0,"anh",i,hbar*hbar),ref),psi_ref=ref,color=PAL[i%len(PAL)]))
    return x,V,S

def solve_H(n_max,r_max,N,tol):
    # Hydrogene : E_n = -1/(2*n^2) en Hartree
    r=np.linspace(1e-4,r_max,N); c0=(r[1]-r[0])**2/12; S=[]
    for n in range(1,n_max+1):
        for l in range(n):
            try:
                m0=matchH(n,l,r); Ea=-1/(2*n*n)
                E=racine(lambda e:errH(e,r,l,c0,m0),Ea,tol,Ea*1.6,Ea*.7,abs(Ea)*.3)
                u=uH(E,r,l,c0,m0)
                for m in range(-l,l+1):
                    S.append(dict(n=n,l=l,m=m,label=nom_H(n,l,m),E=E,Ea=Ea,r=r.copy(),u=u.copy(),color=PAL[(n-1+abs(m))%len(PAL)]))
            except RuntimeError: pass
    return S

# ── BLOC 6 : Statistiques et figures ─────────────────────────────────────────
# Calcul des ecarts numerique/analytique et generation des graphiques Plotly.

def stats(S):
    e=np.array([abs(s["E"]-s["Ea"])/abs(s["Ea"])*100 for s in S if s.get("Ea") is not None and abs(s["Ea"])>1e-30],float)
    return e,float(np.mean(e) if e.size else 0),float(np.std(e,ddof=1) if e.size>1 else 0)

def statsH(S):
    g={}
    for s in S: g.setdefault(s["n"],[]).append(abs(s["E"]-s["Ea"])/abs(s["Ea"])*100)
    e=np.array([np.mean(v) for _,v in sorted(g.items())],float)
    return e,float(np.mean(e) if e.size else 0),float(np.std(e,ddof=1) if e.size>1 else 0)

def ferr(x): return f"{x:.2e} %" if x<0.01 else f"{x:.4f} %"

def fig_ondes(x,V,S,prob,off,ref):
    # Graphique des fonctions d'onde psi(x) ou densites |psi|^2
    fig=go.Figure(); vn=(V-V.min())/(np.ptp(V) if np.ptp(V)>0 else 1.0)
    amp=max(float(np.max(np.abs(s["psi"]))) for s in S) if S else 1.0
    fig.add_trace(go.Scatter(x=x,y=vn*amp*.55-amp*.7 if not prob else vn*.25,mode="lines",name="V(x)",line=dict(color="#C8C4BC",width=2,dash="dot")))
    for i,s in enumerate(S):
        y=(s["psi"]**2 if prob else s["psi"])+i*off
        fig.add_trace(go.Scatter(x=x,y=y,mode="lines",name=f"{s['label']} numerique",line=dict(color=s["color"],width=2)))
        if ref and s.get("psi_ref") is not None:
            yr=(s["psi_ref"]**2 if prob else s["psi_ref"])+i*off
            fig.add_trace(go.Scatter(x=x,y=yr,mode="lines",name=f"{s['label']} analytique",line=dict(color=s["color"],width=1.4,dash="dash")))
    fig.update_layout(template="plotly_white",height=380,margin=dict(l=20,r=20,t=40,b=20),xaxis_title="x",yaxis_title="|psi|^2 (Ha)" if prob else "psi(x)")
    return fig

def fig_spectre(S,titre):
    # Spectre des energies : numerique vs reference analytique
    x=list(range(len(S))); fig=go.Figure()
    fig.add_trace(go.Scatter(x=x,y=[s["E"] for s in S],mode="markers+text",text=[s["label"] for s in S],textposition="top center",marker=dict(size=10,color=[s["color"] for s in S]),name="Numerique"))
    if any(s.get("Ea") is not None for s in S):
        fig.add_trace(go.Scatter(x=x,y=[s.get("Ea") for s in S],mode="markers",marker=dict(size=9,symbol="diamond-open",color="#444"),name="Reference"))
    fig.update_layout(template="plotly_white",title=titre,height=360,margin=dict(l=20,r=20,t=50,b=20),xaxis_title="Etat",yaxis_title="Energie (Ha)")
    return fig

def table(S):
    # Tableau comparatif E numerique / E analytique / erreur relative
    return [{"Etat":s["label"],"E_num (Ha)":s["E"],"E_ref (Ha)":s.get("Ea"),
             "Erreur (%)":abs(s["E"]-s["Ea"])/abs(s["Ea"])*100 if s.get("Ea") is not None and abs(s["Ea"])>1e-30 else None} for s in S]

def meshH(s,nt=56,npf=112,nr=220):
    # Maillage 3D de l'orbitale pour la visualisation
    ir=interpR(s["r"],s["u"]); rmax=max(10.0,s["n"]*s["n"]*5+8.0)
    th=np.linspace(0,math.pi,nt+1); ph=np.linspace(0,2*math.pi,npf+1)
    THH,PH=np.meshgrid(th,ph,indexing="ij"); YY=Y(s["l"],s["m"],THH,PH)
    rad=np.zeros_like(THH); rs=np.linspace(0,1,nr+1)**.82*rmax
    base=np.abs(ir(rs)); pm=float(base.max()*np.abs(YY).max()); iso=pm*.12 if pm>1e-10 else 0.0
    for i in range(THH.shape[0]):
        for j in range(THH.shape[1]):
            if abs(YY[i,j])<1e-10 or iso<=0: continue
            ok=np.where(base*abs(YY[i,j])-iso>=0)[0]; rad[i,j]=rs[int(ok[-1])] if ok.size else 0.0
    for _ in range(2): rad=.52*rad+.16*np.roll(rad,1,1)+.16*np.roll(rad,-1,1)+.08*np.vstack([rad[:1],rad[:-1]])+.08*np.vstack([rad[1:],rad[-1:]])
    X,Yc,Z=rad*np.sin(THH)*np.cos(PH),rad*np.sin(THH)*np.sin(PH),rad*np.cos(THH)
    sg=np.where(YY.ravel()>=0,1,-1); w=npf+1; I,J,K,T=[],[],[],[]
    for i in range(nt):
        for j in range(npf):
            a,b,c,d=i*w+j,i*w+j+1,(i+1)*w+j,(i+1)*w+j+1
            I.extend((a,b)); J.extend((b,d)); K.extend((c,c))
            T.extend((1 if sg[a]+sg[b]+sg[c]>=0 else -1,1 if sg[b]+sg[d]+sg[c]>=0 else -1))
    return dict(x=X.ravel(),y=Yc.ravel(),z=Z.ravel(),i=np.array(I),j=np.array(J),k=np.array(K),t=np.array(T))

def fig_H(s,h=620,ttl=True,mini=False):
    # Visualisation 3D de l'orbitale de l'hydrogene
    m=meshH(s,28,56,120) if mini else meshH(s); fig=go.Figure()
    for sign,color,name in ((1,"#2A78A8","psi > 0"),(-1,"#C66843","psi < 0")):
        q=m["t"]==sign
        if np.any(q): fig.add_trace(go.Mesh3d(x=m["x"],y=m["y"],z=m["z"],i=m["i"][q],j=m["j"][q],k=m["k"][q],color=color,opacity=1.0,name=name,flatshading=False,lighting=dict(ambient=.55,diffuse=.85,specular=.35,roughness=.35,fresnel=.1),lightposition=dict(x=120*sign,y=100,z=180)))
    fig.update_layout(template="plotly_white",height=h,margin=dict(l=0,r=0,t=40 if ttl else 10,b=0),title=f"Orbitale 3D : {s['label']}" if ttl else None,scene=dict(aspectmode="data",xaxis_title="x",yaxis_title="y",zaxis_title="z",camera=dict(eye=dict(x=1.8,y=1.6,z=1.3))),legend=dict(orientation="h"))
    return fig

# ── BLOC 7 : Interface Streamlit ──────────────────────────────────────────────
# Sidebar : choix du systeme et des parametres.
# Corps   : affichage des graphiques et du tableau de resultats.

def main():
    st.set_page_config(page_title="Schrodinger — Numerov",page_icon="⚛️",layout="wide")
    st.markdown("""<style>
    :root{--primary-color:#002147;}
    body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stSidebar"],section[data-testid="stSidebar"]{background:#F7F7F2 !important;}
    [data-testid="stStatusWidget"],[data-testid="stDecoration"],[data-testid="stToolbarActions"],[data-testid="stMainMenu"]{display:none !important;}
    header[data-testid="stHeader"] div:has(> button[kind="header"]) + div{display:none !important;}
    [data-testid="stSidebar"] button,[data-testid="stSidebar"] [role="slider"]{accent-color:#002147;}
    div[data-baseweb="select"] *{border-color:#002147 !important;}
    </style>""",unsafe_allow_html=True)
    st.title("Resolution numerique de l'equation de Schrodinger")
    st.caption("Methode de Numerov — unites atomiques (Hartree).")

    with st.sidebar:
        sys=st.selectbox("Systeme",[("box","Particule dans une boite"),("ho","Oscillateur harmonique"),("anh","Oscillateur anharmonique"),("H","Atome d'hydrogene")],format_func=lambda t:t[1],key="sys")[0]
        st.markdown(TH[sys])
        if sys=="box":
            p=dict(L=st.slider("Longueur L",0.5,6.0,2.22,.01,key="box_L"),n_max=st.slider("n_max",1,8,5,key="box_n"),N=st.slider("Points N",500,6000,2000,250,key="box_N"),tol=float(st.selectbox("Tolerance",("1e-6","1e-8","1e-10"),1,key="box_tol")))
        elif sys=="ho":
            p=dict(omega=st.slider("omega",0.1,3.0,1.0,.05,key="ho_w"),hbar=st.slider("hbar",0.1,3.0,1.0,.05,key="ho_h"),x_max=st.slider("x_max",5.0,22.0,12.0,.5,key="ho_x"),n_max=st.slider("n_max",0,8,4,key="ho_n"),N=st.slider("Points N",500,6000,2000,250,key="ho_N"),tol=float(st.selectbox("Tolerance",("1e-6","1e-8","1e-10"),1,key="ho_tol")))
        elif sys=="anh":
            p=dict(omega=st.slider("omega",0.1,3.0,1.0,.05,key="anh_w"),lamb=st.slider("lambda",0.0,0.2,0.1,.01,key="anh_l"),hbar=st.slider("hbar",0.1,3.0,1.0,.05,key="anh_h"),x_max=st.slider("x_max",5.0,22.0,12.0,.5,key="anh_x"),n_max=st.slider("n_max",0,8,4,key="anh_n"),N=st.slider("Points N",500,6000,2000,250,key="anh_N"),tol=float(st.selectbox("Tolerance",("1e-6","1e-8","1e-10"),1,key="anh_tol")))
        else:
            p=dict(n_max=st.slider("n_max",1,3,3,key="H_n"),r_max=st.slider("r_max",20.0,120.0,60.0,5.0,key="H_r"),N=st.slider("Points radiaux N",1000,8000,3500,250,key="H_N"),tol=float(st.selectbox("Tolerance",("1e-8","1e-10","1e-12"),1,key="H_tol")))
        if sys!="H":
            off=st.slider("Decalage vertical",0.0,2.5,0.0,.05,key="off")
            show_ref=st.checkbox("Afficher la reference analytique",value=(sys in ("box","ho")),key="show_ref")
        else: off,show_ref=0.0,False

    # Cache : evite de recalculer si les parametres n'ont pas change
    if "res" not in st.session_state: st.session_state.res=None; st.session_state.last_sys=None; st.session_state.last_p=None
    if st.session_state.res is None or st.session_state.last_sys!=sys or st.session_state.last_p!=p:
        with st.spinner("Calcul en cours..."):
            st.session_state.res={"box":lambda:solve_box(**p),"ho":lambda:solve_ho(**p),"anh":lambda:solve_anh(**p),"H":lambda:solve_H(**p)}[sys]()
            st.session_state.last_sys=sys; st.session_state.last_p=p
    res=st.session_state.res

    # Affichage hydrogene
    if sys=="H":
        S=res; _,m,sd=statsH(S); a,b=st.columns([1.2,1])
        a.metric("Ecart moyen",ferr(m)); b.metric("Ecart-type",ferr(sd))
        labs=[s["label"] for s in S]
        if "h_sel" not in st.session_state or st.session_state.h_sel not in labs: st.session_state.h_sel=labs[0]
        sel=next(s for s in S if s["label"]==st.session_state.h_sel)
        c1,c2,c3=st.columns(3)
        c1.metric("E numerique (Ha)",f"{sel['E']:.8f}"); c2.metric("E theorique (Ha)",f"{sel['Ea']:.8f}"); c3.metric("Erreur",ferr(abs(sel['E']-sel['Ea'])/abs(sel['Ea'])*100))
        st.plotly_chart(fig_H(sel), use_container_width=True)
        with st.expander("Galerie 3D des orbitales"):
            for pack in [S[i:i+3] for i in range(0,len(S),3)]:
                cols=st.columns(3)
                for c,s in zip(cols,pack):
                    with c:
                        st.plotly_chart(fig_H(s,220,False,True), use_container_width=True)
                        if st.button(s["label"],key=f"h_{s['label']}"): st.session_state.h_sel=s["label"]
        st.plotly_chart(fig_spectre(S,"Spectre de l'atome d'hydrogene (Ha)"), use_container_width=True)
        st.dataframe(table(S),hide_index=True)
    # Affichage box / ho / anh
    else:
        x,V,S0=res
        choix=st.multiselect("Etats n visibles",sorted({s["n"] for s in S0}),default=sorted({s["n"] for s in S0}),key="n_vis")
        S=[s for s in S0 if s["n"] in choix] if choix else S0[:1]
        _,m,sd=stats(S); a,b=st.columns(2)
        a.metric("Ecart moyen",ferr(m)); b.metric("Ecart-type",ferr(sd))
        if sys=="anh" and show_ref: st.info("Reference : fonction propre perturbative d'ordre 1.")
        c1,c2=st.columns(2)
        c1.plotly_chart(fig_ondes(x,V,S,False,off,show_ref))
        c2.plotly_chart(fig_ondes(x,V,S,True,off,show_ref))
        st.plotly_chart(fig_spectre(S,"Spectre des valeurs propres (Ha)"), use_container_width=True)
        st.dataframe(table(S),hide_index=True)

if __name__=="__main__": main()
