# -*- coding: utf-8 -*-
"""
Created on Sun May 27 16:54:08 2018

@author: Administrator
"""
import pandas as pd
import rpy2.robjects as rob
from rpy2.robjects.packages import importr
import matplotlib.pyplot as plt
import numpy as np
importr("effsize","C:/Program Files/R/R-3.4.3/library")
def WDL(x1,x2):
    """
    x1的wdl值
    """
    x1,x2=list(x1),list(x2)
    w,d,l=0,0,0
    for i,j in zip(x1,x2):
        i,j=round(i,3),round(j,3)
        if(i<j):
            l+=1
        elif(i>j):
            w+=1
        else:
            d+=1
    return str(w)+"/"+str(d)+"/"+str(l)
def WDL_Wilcox(x1,x2):
    """
    利用显著性分析做WDL
    """
    x1,x2=list(x1),list(x2)
    w,d,l=0,0,0  
    t1,t2=[],[]
    for i,j in zip(x1,x2):
        i,j=round(i,3),round(j,3)
        t1.append(i)
        t2.append(j)
        if(len(t1)==25):#开始做检验
            wilValue=RWilcox(t1,t2)
            
            if(wilValue<0.05):#有显著性差异
                if(sum(t1)<sum(t2)):
                    l+=1
                else:
                    w+=1
            else:
                d+=1
            t1,t2=[],[]
    return str(w)+"/"+str(d)+"/"+str(l)        
        
        
def RWilcox(x1,x2):
    x1,x2=list(x1),list(x2)
    a,b=rob.FloatVector(x1),rob.FloatVector(x2)
    return rob.r["wilcox.test"](a,b,paired = True)[2][0]           
def TopRank(X,modelNames,batch=1):
    data=np.empty(shape=(int(len(X)/batch),len(modelNames)))
    res=[0]*len(modelNames)
    for i in range(len(data)):
        data[i]=X[list(range(i*batch,i*batch+batch)),:].mean(axis=0)
    data=data.round(3)
#    print(data)
    for i in range(len(data)):
        maxValue=data[i].max()
        for j in range(len(modelNames)):
            if(data[i,j]==maxValue):
                res[j]+=1    
    for i in res:
        print(i,end=',')
def RCliffDelta(x1,x2):   
    x1,x2=list(x1),list(x2)
    a,b=rob.FloatVector(x1),rob.FloatVector(x2)
    delta=rob.r["cliff.delta"](a,b,0.95)[0][0]
    if(abs(delta)>=0.474):
        effectsize="Large"
    elif(abs(delta)>=0.33):
        effectsize="Medium"
    elif(abs(delta)>=0.147):
        effectsize="Small"
    else:
        effectsize="Negligible"
    return delta,effectsize






