from bokeh.io import show, output_notebook,curdoc,save, output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d, BBoxTileSource
from bokeh.layouts import row
output_notebook()
#output_file("map1.html",'bokeh graphs')
'''
import matplotlib.pyplot as plt
#%matplotlib inline
#the above line must be included and commented out
'''


class DataView:
    def __init__(self):
        