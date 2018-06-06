# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from SMOTENDDE import SMOTENDDE
from SMOTEND import SMOTEND
from Data import Data
import BuildingModel as BM
import Evaluate as E
if(__name__=="__main__"):
    # In[some parameters] 
    datasetsPaths={#Windows 下的路径
                   "ant":["bugs\\ant-1.3.csv",
                           "bugs\\ant-1.4.csv",
                           "bugs\\ant-1.5.csv",
                           "bugs\\ant-1.6.csv",
                           "bugs\\ant-1.7.csv"],
                   "camel":["bugs\\camel-1.0.csv",
                           "bugs\\camel-1.2.csv",
                           "bugs\\camel-1.4.csv",
                           "bugs\\camel-1.6.csv"],
                   "jedit":["bugs\\jedit-3.2.csv",
                           "bugs\\jedit-4.0.csv",
                           "bugs\\jedit-4.1.csv",
                           "bugs\\jedit-4.2.csv"],
                   "synapse":["bugs\\synapse-1.0.csv",
                           "bugs\\synapse-1.1.csv",
                           "bugs\\synapse-1.2.csv"],
                   "ivy":["bugs\\ivy-1.1.csv",
                          "bugs\\ivy-1.4.csv",
                          "bugs\\ivy-2.0.csv"],
                   "xalan":["bugs\\xalan-2.4.csv",
                            "bugs\\xalan-2.5.csv",
                            "bugs\\xalan-2.6.csv"],
                    "xerces":["bugs\\xerces-1.2.csv",
                              "bugs\\xerces-1.3.csv"]
                   }
    models=["EALR","BRR","DTR","NNR","GBR","RF","AdaBoostR2_LR","AdaBoostR2_DTR","AdaBoostR2_BRR"]
    testSets=["ant-1.7","ant-1.6","ant-1.5","camel-1.6","camel-1.4","camel-1.2","ivy-2.0",
              "jedit-4.2","jedit-4.1","jedit-4.0","synapse-1.1","synapse-1.2",
              "xalan-2.5","xalan-2.6","xerces-1.3"]
#    
#    models=["EALR","BRR","DTR"]
#    testSets=["ant-1.5","camel-1.2","ivy-2.0"]
    trainingProjs=["ant","camel","jedit","synapse","ivy","xalan","xerces"]
    unsupervisedModels=['U_LocRfc_A','U_LocRfc_D','U_amc_A','U_amc_D','U_avg_cc_A','U_avg_cc_D','U_ca_A','U_ca_D','U_cam_A','U_cam_D','U_cbm_A',
                        'U_cbm_D','U_cbo_A','U_cbo_D','U_ce_A','U_ce_D','U_dam_A','U_dam_D','U_dit_A','U_dit_D','U_ic_A','U_ic_D','U_lcom3_A',
                        'U_lcom3_D','U_lcom_A','U_lcom_D','U_loc_A','U_loc_D','U_max_cc_A','U_max_cc_D','U_mfa_A','U_mfa_D','U_moa_A','U_moa_D',
                        'U_noc_A','U_noc_D','U_npm_A','U_npm_D','U_rfc_A','U_rfc_D','U_wmc_A','U_wmc_D']
    #i will give you a simple example about how to use our codes
    # In[]Step 1 get traning set and test set
    x_train,y_train,x_test,y_test=Data(paths=datasetsPaths, scenario="Cross-Version", testSetName="ant-1.7", trainingProjName="").getTrainingAndTestSet()

    # In[] step 2 build and train a model 
    smo=SMOTEND(k=10, m=3, r=2)#you can use SMOTEND or SMOTEND+DE, NOTICE:the smotendde return the last generation
    x_train,y_train=smo.transform(x_train,y_train)
    m=BM.buildModel(x_train,y_train,modelName="RF")
    # In[] step3 model selection
    FPAResult=E.Eva(m,x_test,y_test,metric="FPA")
    print(FPAResult)
    
    





























    
    
    
