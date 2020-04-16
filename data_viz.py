import logging
import numpy as np

'''from bokeh.io import show, output_notebook,curdoc,save, output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d, BBoxTileSource
from bokeh.layouts import row
output_notebook()
#output_file("map1.html",'bokeh graphs')'''
'''
import matplotlib.pyplot as plt
#%matplotlib inline
#the above line must be included and commented out
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class DataView:
    def __init__(self):
        self.logger=logging.getLogger(__name__)
        self.logger.debug('DataView object started')
        plt.rc_context({'axes.edgecolor':'orange', 
                        'xtick.color':'red', 'ytick.color':'red', 
                        'figure.facecolor':'white'})
        #self.fig=plt.figure(figsize=[14,80])
        
        
          
    def my2dbarplot(self,x,y,xlabel='x',ylabel='y',title=None,fig=None,subplot_idx=(1,1,1)):
        if not fig: fig=plt.figure(dpi=self.dpi,figsize=[14,10])
        ax=fig.add_subplot(*subplot_idx)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        width=0.8*(x.max()-x.min())/x.size
        if title:
            ax.set_title(title)
        ax.bar(x,y,color='darkslategrey', alpha=0.8,width=width)
        ax.grid(True)
        '''self.ax.xaxis.label.set_color('blue')
        self.ax.yaxis.label.set_color('blue')                                  
        self.ax.spines['top'].set_color('blue') '''
        return fig
        
                                  
    def my3dHistogram(self,arraylist,varname,dim3list,subplot_idx=[1,1,1],fig=None,norm_hist=1,ylabel='time'): 
        if fig is None:
            fig=plt.figure(dpi=self.dpi,figsize=[self.figwidth,self.figheight])
            
        uniquecount=max([np.unique(nparray).size for nparray in arraylist])
        if uniquecount<4:nbins=8
        elif uniquecount<10:nbins=30
        else:nbins=min([50,uniquecount])
        maxrange=max([nparray.max() for nparray in arraylist])-min([nparray.min() for nparray in arraylist])
        #width=0.7*(np.log10(nbins**1.5)-1)*(maxrange)/nbins    
        width=.8*maxrange/nbins
        yarray=np.array(dim3list,dtype=int)
        left=min([nparray.min() for nparray in arraylist])
        right=max([nparray.max() for nparray in arraylist])
        #right=max([np.percentile(nparray,99) for nparray in arraylist])
        ax=fig.add_subplot(*subplot_idx,projection='3d')
        ax.set_xlabel(varname)
        ax.set_ylabel(ylabel)
        if norm_hist:
            ax.set_zlabel(f'relative frequency by {ylabel}')
            ax.set_zticklabels([])
        else: ax.set_zlabel(f'count per bin of {nbins} bins')
        ax.set_xlim(left,right)
        histtuplist=[]
        for i,nparray in enumerate(arraylist):
            y=yarray[i]
            freq,bins=np.histogram(nparray,bins=nbins,density=norm_hist)#dim0 is time
            x=0.5*(bins[1:]+bins[:-1])
            z=freq
            ax.view_init(elev=40,azim=-35)
            ax.bar(x,z,zs=y,zdir='y',alpha=0.95,width=width)
            #ax.bar(y,z,zs=x,zdir='x',alpha=0.95,width=width)
            histtuplist.append((freq,bins))

        return fig,{'varname':varname,'hist_tup_list':histtuplist,'norm_hist':norm_hist}
        
        

        
            
    def plotGeog(self,arraylist,yearlist=None):
        lat_idx=self.varlist.index('latitude')
        lon_idx=self.varlist.index('longitude')
        latlist=[nparray[:,lat_idx] for nparray in arraylist]
        longlist=[nparray[:,lon_idx] for nparray in arraylist]
        pass
        
