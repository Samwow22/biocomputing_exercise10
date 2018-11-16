#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:22:19 2018

@author: Samuel_Clarin
"""
###load packages
import numpy as np
import pandas as pd
import scipy
import scipy.integrate as spint
from plotnine import *
from scipy import stats

#problem 1

x=data.x
y=data.y
df=pd.DataFrame({'x':x, 'y':y})

def line(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*obs.x
    lans=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return lans

initialGuess=np.array([1,1,1])
lfit=minimize(line,initialGuess,method="Nelder-Mead",options={'disp':True},args=df)
print(lfit.x)

def quad(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    B2=p[3]
    expected=B0+B1*obs.x+(B2*obs.x*obs.x)
    qans=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return qans

quadGuess=np.array([1,10,1,0.11])
qfit=minimize(quad,quadGuess,method="Nelder-Mead",options={'disp':True},args=df)

print(qfit.x)

teststat=2*(lfit.fun-qfit.fun)
df2=len(qfit.x)-len(lfit.x)
1-stats.chi2.cdf(teststat,df2)

#problem 2: solving Lotka-Volterra

def LV (y,t0,R1,R2,N1,N2,a11,a12,a22,a21):
   #unpacking
    N1=y[0]
    N2=y[1]
    
    #calculate change given parameters
    dN1dt=R1*(1-N1*a11-N2*a12)*N1
    dN2dt=R2*(1-N2*a22-N2*a21)*N2
    #return list with change in state variables by time
    return [dN1dt,dN2dt]

times=range(0,100)
y0=[0.1,0.1]
params=(0.5,10,0.5,10,0.05,0.05,5,0.05)

#simulate model using odeint

Sim=spint.odeint(func=LV,y0=y0,t=times,args=params)

#put model output in a dataframe for plotting purposes
MO1=pd.DataFrame({"t":times,"N1": Sim[:,0],"N2":Sim[:,1]})

#plot simulation output
ggplot(MO1, aes(x="t",y="N1"))+geom_line()+geom_line(MO1, aes(x="t", y="N2"), color='red')+theme_classic()



#case 1 where all alphas are the same
times1=range(0,100)
y01=[0.1,0.1]
params1=(0.5,0.5,0.5,0.5,0.05,0.05,0.05,0.05)

Sim1=spint.odeint(func=LV,y0=y01,t=times1,args=params1)
MO1=pd.DataFrame({"t":times1,"N1": Sim1[:,0],"N2":Sim1[:,1]})
ggplot(MO1, aes(x="t",y="N1"))+geom_line()+geom_line(MO1, aes(x="t", y="N2"), color='red')+theme_classic()

#case 2 where a11 and a22 are larger than a12 and a21
times2=range(0,100)
y02=[0.01,0.01]
params2=(0.5,0.5,0.5,0.5,0.1,0.05,0.1,0.05)

Sim2=spint.odeint(func=LV,y0=y02,t=times2,args=params2)
MO2=pd.DataFrame({"t":times2,"N1": Sim2[:,0],"N2":Sim2[:,1]})
ggplot(MO2, aes(x="t",y="N1"))+geom_line()+geom_line(MO2, aes(x="t", y="N2"), color='red')+theme_classic()

#case 3 where a12 and a21 are larger than a11 and a22
times3=range(0,100)
y03=[0.01,0.01]
params3=(0.5,0.5,0.5,0.5,0.05,0.08,0.05,0.08)

Sim3=spint.odeint(func=LV,y0=y03,t=times3,args=params3)
MO3=pd.DataFrame({"t":times3,"N1": Sim3[:,0],"N2":Sim3[:,1]})
ggplot(MO3, aes(x="t",y="N1"))+geom_line()+geom_line(MO3, aes(x="t", y="N2"), color='red')+theme_classic()

