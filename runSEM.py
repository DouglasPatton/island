import island

if __name__==__main__:
    idata_obj=island.IslandData()
    idata_obj.makeTimeListArrayList()
    idata_obj.addRealByCPI(to_year=2015)
    idata_obj.arrayListToPandasDF()
    resultslistlist=idata_obj.runSpatialModel()
    