# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 19:47:39 2017

@author: Administrator
"""
from sklearn import linear_model
from sklearn.svm import SVR  
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
import UnsupervisedModel
from CBS import CBS
from OneWay import OneWay
def buildModel(x,y,modelName,metric=None):
    """
    构建模型
    
    Parameters
    ----------
    trainingSet : 训练集, array-like
    
    modelType :模型类型 LR(Linear Regression),BRR(Bayesian Ridge Regression),SVR(Support Vector Regression),
    NNR(Nearest Neighbours Regression),DTR(Decision Tree Regression),GBR(GGradient Boosting Regression)
    
    Returns
    -------
    model
    """
    # In[] 建立模型
    model=None
    if(modelName == "EALR"):
        model = linear_model.LinearRegression()
    elif(modelName == "BRR"):
        model = linear_model.BayesianRidge(compute_score=True)
    elif(modelName == "SVR"):
        model = SVR()
    elif(modelName == "NNR"):
        model = KNeighborsRegressor()
    elif(modelName == "DTR"):
        model = DecisionTreeRegressor()
    elif(modelName == "GBR"):
        model = GradientBoostingRegressor() 
    elif(modelName=="RF"):
        model=RandomForestRegressor()
    elif("U_" in modelName):#无监督算法
        if("LocRfc" in modelName):
            model=UnsupervisedModel.LocRfc(sign=modelName[-1])
        else:
            model=UnsupervisedModel.UnsupervisedModel(attrName=modelName[2:])
    elif("OneWay" in modelName):
        model=OneWay(sign=modelName[-1],metric=metric)
    elif ("CBS" in modelName):
        model=CBS(attrName="loc_"+modelName[-1])
    elif("AdaBoostR2" in modelName):
        base_m=modelName.split("_")[1]
        if(base_m=="LR"):
            model=AdaBoostRegressor(base_estimator= linear_model.LinearRegression())
        elif(base_m=="DTR"):
            model=AdaBoostRegressor(base_estimator=DecisionTreeRegressor())
        elif(base_m=="BRR"):
            model=AdaBoostRegressor(base_estimator=linear_model.BayesianRidge(compute_score=True))
        
    # In[] 训练模型    
    model.fit(x,y)
    return model
        
        
        
