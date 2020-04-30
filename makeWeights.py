import island
import pickle

def savepickle(thing,path):
    with open(path,'wb') as f:
        pickle.dump(thing,f)
        
if __name__=='__main__':
    idata_obj=island.IslandData()
    idata_obj.makeTimeListArrayList()
    idata_obj.addRealByCPI(to_year=2015)
    idata_obj.arrayListToPandasDF()
    idata_obj.runSpatialModel(justW=1)
    savepickle(resultslistlist,'resultslistlist.pickle')
    savepickle(idata_obj,'idata_obj.pickle')
    