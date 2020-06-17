import island
import pickle
import os

def savepickle(thing,path):
    with open(path,'wb') as f:
        pickle.dump(thing,f)
        
if __name__=='__main__':
    
    
    idata_obj=island.IslandData()
    idata_obj.makeTimeListArrayList()
    idata_obj.addRealByCPI()
    idata_obj.arrayListToPandasDF()
    idata_obj.runSpatialModel()
    idata_obj.printModelSummary()
    