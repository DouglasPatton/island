from geopy import distance

class DataGeoTool():
    def __init__(self,):
        pass
    
    
    def makeGeoDistMat(self,lon_array,lat_array):
        # https://geopy.readthedocs.io/en/latest/#module-geopy.distance
        # default geodesic distance assuming WGS-84
        n=lon_array.size
        assert lon_array.size==lat_array.size,f'mismatch in lat lon, \
            (lon_array.shape,lat_array.shape):{(lon_array.shape,lat_array.shape)}'
        pointlist=self.makePointList(lon_array,lat_array)
        
        
        distmat=[[[0.0]for _ in range(n)] for _ in range(n)]
        for idx0 in range(n):
            for idx1 in range(idx0+1,n):# plus 1 to skip self.
                dist01=distance.geodesic(pointlist[idx0],pointlist[idx1]) # default:WGS-84
                distmat[idx0][idx1]=dist01
                distmat[idx1][idx0]=dist01
                self.logger.info(f'(idx0,idx1,dist01):{(idx0,idx1,dist01)}')
        return distmat
            
            
    def makePointList(self,lon_array,lat_array):
        n=lon_array.size
        pointlist=[distance.lonlat(lon_array[i],lat_array[i]) for i in range(n)]
        return pointlist
            