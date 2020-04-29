import numpy as np
import data_geo as dg
import pysal
#import libpysal
#import geopandas
import geopy

class SpatialModel():
    def __init__(self,modeldict=None):
        if modeldict is None:
            modeldict={
                'combine_pre_post':0,
                'modeltype':'SEM',
                'nneighbor':[10,15,20],
                'crs':'epsg:4326'
            }
        self.modeldict=modeldict
        
    
        
    def buildGeoDF(self,df=None):
        if df is None:
            try:self.df
            except: self.arrayListToPandasDF()
            df=self.df
        try:
            points=geopandas.points_from_xy(df['longitude'],df['latitude'])
        except:
            points=geopandas.points_from_xy(df[:]['longitude'],df[:]['latitude'])
        gdf=geopandas.GeoDataFrame(df,geometry=points,crs={'init':self.crs})
        return gdf         
                 
                 
    def run(self,df=None):
        # https://pysal.org/libpysal/generated/libpysal.weights.W.html#libpysal.weights.W
        nn=self.modeldict['nneighbor']
        if type(nn) is int:
            klist=[nn]
        elif type(nn) is list:
            klist=nn
        else:assert False,f'halt, unrecongized nn:{nn}'
            
        if df is None:
            try:self.df
            except: self.arrayListToPandasDF()
            df=self.df
        
        if self.modeldict['combine_pre_post']==1:
            df_idx_0_list=[slice(None)]
        else:
            df_idx_0_list=[0,1]
        wtlistlist=[];resultslistlist=[]
        for idx0 in df_idx_0_list:
            dfi=df.loc[idx0]
            #gdfi=self.buildGeoDF(df=dfi)
            
            wtlist=self.makeInverseDistanceWeights(dfi,klist=klist)
            wtlistlist.append(wtlist)
            resultslistlist.append([self.runSpatialErrorModel(dfi,w) for w in wtlist])
        return resultslistlist
    
    def runSpatialErrorModel(self,df,w):
    
        y=np.log10(df.loc['saleprice_real-2015'].to_numpy())
        xvarlist=[
            'secchi','bayfront','wateraccess','wqbayfront','wqwateraccess',
            'totalbathroomsedited','totallivingarea','saleacres',
            'distance_park','distance_nyc','distance_golf',
            'shorelinedistancedv3_1000','shorelinedistancedv1000_2000',
            'shorelinedistancedv2000_3000','shorelinedistancedv3000_4000',
            'wqshorelinedistancedv3_1000','wqshorelinedistancedv1000_2000',
            'wqshorelinedistancedv2000_3000','wqshorelinedistancedv3000_4000',
            'education','income_real-2015','povertylevel','pct_white']
                  
        #xvarlist=[varlist.pop(varlist.index(var)) for var in excludevars]
        
        x=df.loc[xvarlist].to_numpy()
        sem=pysal.ML_Error(y,x,w,name_y=None,name_x=None,name_w=None)
        return sem
    
    
    def makeInverseDistanceWeights(self,dfi,klist=[20],distmat=None):
        '''
        from libpysal.weights import W
        neighbors = {0: [3, 1], 1: [0, 4, 2], 2: [1, 5], 3: [0, 6, 4], 4: [1, 3, 7, 5], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]}
        weights = {0: [1, 1], 1: [1, 1, 1], 2: [1, 1], 3: [1, 1, 1], 4: [1, 1, 1, 1], 5: [1, 1, 1], 6: [1, 1], 7: [1, 1, 1], 8: [1, 1]}
        w = W(neighbors, weights)
        '''
        n=dfi.shape[0
        neighbors_dictlist=[{} for _ in range(len(klist))]
        weights_dictlist=[{} for _ in range(len(klist))]
        if distmat:
            distmat=self.makeDistMat(dfi)
            for i in range(n):
                dist,jlist=zip(*sorted(zip(distmat[i],list(range(n)))))
                for k_idx in range(len(klist)):
                    neighbors_dictlist[k_idx][i]=jlist[:klist[k_idx]]
                    weights_dictlist[k_idx][i]=dist[:klist[k_idx]]**(-1)
        else:
            pointlist=self.makePointListFromDF(dfi)
            for idx in range(n):
                distlist=self.makeDistList(idx,pointlist)
                dist,j_idx=zip(*sorted(zip(distlist,list(range(n))))) 
                #    j_idx indexes positions of other points, dist is their distance
                for k_idx in range(len(klist)):
                    neighbors_dictlist[k_idx][idx]=j_idx[:klist[k_idx]]
                    weights_dictlist[k_idx][idx]=dist[:klist[k_idx]]**(-1)    
                
        wlist=[]
        for k_idx in range(len(klist)):
            wlist.append(pysal.weights.W(neighbors_dictlist[k_idx],weights_dictlist[k_idx]))
        return wlist
    
    def makePointListFromDF(self,df):
        lon_array=df['longitude'].to_numpy()
        lat_array=df['latitude'].to_numpy()
        n=lon_array.size
        pointlist=[(lon_array[i],lat_array[i]) for i in range(n)]
        return pointlist
   
    def makeDistList(self,idx,pointlist):
        otherpoints=pointlist.copy()
        thispoint=otherpoints.pop(idx)
        distlist=[[10**299] for _ in range(len(otherpoints))]            
        for p_idx in otherpoints:
            thatpoint=otherpoints[p_idx]
            distlist[p_idx]=geopy.distance.geodesic(thispoint,thatpoint)
        return distlist
            
            
        
        
                           
    def makeDistMat(self,df):
        lon_array=df['longitude'].to_numpy()
        lat_array=df['latitude'].to_numpy()
        n=lon_array.size
        pointlist=[(lon_array[i],lat_array[i]) for i in range(n)]
        distmat=[[[10**299] for _ in range(n)] for _ in range(n)] #
        for i0 in range(n):
            for i1 in range(i0+1,n):
                distance=geopy.distance.geodesic(pointlist[i0],pointlist[i1])
                distmat[i0][i1]=distance#not taking advantage of symmetry of distance for storage, assuming ample ram
                distmat[i1][i0]=distance
        self.distmat=distmat
                