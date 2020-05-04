import island
import pickle
import os

def savepickle(thing,path):
    with open(path,'wb') as f:
        pickle.dump(thing,f)
        
if __name__=='__main__':
    
    
    idata_obj=island.IslandData()
    idata_obj.makeTimeListArrayList()
    idata_obj.addRealByCPI(to_year=2015)
    idata_obj.arrayListToPandasDF()
    resultslistlist=idata_obj.runSpatialModel()
    savepickle(resultslistlist,idata_obj.results_path)
    