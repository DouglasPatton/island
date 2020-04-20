import numpy as np
import data_geo as dg

class Model():
    def __init__(self,modeldict=None:
        if modeldict is None:
            modeldict={
                combine_pre_post=1,
                modeltype='SEM'
            }
        self.modeldict=modeldict
        if self.modeldict['combine_pre_post']==1:
            df_idx_0_list=[slice(None)]
                 
                 
                 
    def buildSpatial_W_FromDF(self,df=None):
        if df is None:
            try:self.df
            except: self.arrayListToPandasDF()
        lonarray=self.df[]