# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:09:47 2017

@author: Zhdun
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def Minkowski(u,v,**kwargs):
#    u,v=np.array(u),np.array(v)
    p=kwargs["P"]
    d=(np.abs(u-v)**p).sum()**(1.0/p)
    if(d==0):
        return float("inf")
    return d
class SMOTEND():
    def __init__(self,k=10,m=3,r=2):
        self.k,self.m,self.r=k,m,r
    def fit(self,x=None,y=None):
        pass
    def transform(self,x,y):
        k,m,r=self.k,self.m,self.r
        if(m==0):
            return (x,y)
        pIns=[]
        for i in range(len(y)):
            if(y[i]!=0):
                pIns.append(i)
        xPositive,yPositive=x[pIns],y[pIns]
        
        ######################################################
        if(k>len(yPositive)-3):
            k=len(yPositive)-3
        k=int(k)
        ind=NearestNeighbors(n_neighbors=k,metric=Minkowski, metric_params={"P":r}).fit(xPositive).kneighbors(return_distance=False)
        numInstances=int((len(x)-2*len(pIns))/6.0*m)
        count=[int(numInstances/len(pIns))]*len(pIns)
        for i  in range(numInstances % len(pIns)):
            count[i]+=1
        neighboursX,neighboursY,targetsX,targetsY=[],[],[],[]
        for i in range(len(ind)):#对于每一个少数类
            t=ind[i][np.random.randint(low=0,high=k,size=count[i])]
            neighboursX.append(xPositive[t])
            neighboursY.append(yPositive[t].reshape(count[i],1))#要与目标实例生成新的实例，最近邻向量组成的矩阵
            targetsX.append(xPositive[[i]*count[i]])
            targetsY.append(yPositive[[i]*count[i]].reshape(count[i],1))#重复该少数类组成矩阵
         
        neighboursX,neighboursY,targetsX,targetsY=np.vstack(neighboursX),np.vstack(neighboursY),np.vstack(targetsX),np.vstack(targetsY)
         #生成新向量  
        newX=targetsX+(neighboursX-targetsX)*np.random.random_sample(targetsX.shape)
        d1=(np.abs(neighboursX-newX)**r).sum(axis=1)**(1.0/r)#与newX的距离
        d2=(np.abs(targetsX-newX)**r).sum(axis=1)**(1.0/r)#与targets的距离
        newY=(d1.reshape(len(d1),1)*targetsY+d2.reshape(len(d2),1)*neighboursY)/(d1+d2).reshape(len(d1),1)
            
        return (np.vstack((x,newX)),np.vstack((y.reshape(len(y),1),newY)).ravel())


if (__name__=="__main__"):
    data=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    t=data.sum(axis=1)
#    q1,q2=NearestNeighbors(n_neighbors=int(3),p=3).fit(data).kneighbors(data)
    q1,q2=NearestNeighbors(n_neighbors=4,metric=Minkowski, metric_params={"p": 2}).fit(data).kneighbors(data)
#    sklearn.neighbors.dist_metrics.MinkowskiDistance(0.5)
    
    data=np.array(pd.read_csv("bugs/ant-1.7.csv"))
    x,y=data[:,0:-1],data[:,-1]
#    for i in range(len(y)):
#        if(y[i]!=0):
#            y[i]=1
    sm=SMOTEND()
    sm.fit()
    xx,yy,_=sm.transform(x,y)
#    pass






