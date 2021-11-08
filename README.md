# island
## long island hedonic analysis
 -  [draft paper](https://github.com/DouglasPatton/island/blob/master/Long%20Island%20Hedonic%20Paper%20Draft.pdf)
### build the environment
 1. `conda create -c conda-forge --name island python=3 pysal jupyterlab pandas scipy matplotlib statsmodels`
 2. `pip install haversine`

### Explore/visualize
time series of histograms
 - [most variables](https://nbviewer.jupyter.org/github/DouglasPatton/island/blob/master/island-data_summary_hist.ipynb)
 - [dollar variables](https://nbviewer.jupyter.org/github/DouglasPatton/island/blob/master/island-data_dollarvars.ipynb)
 - [sale acres](https://nbviewer.jupyter.org/github/DouglasPatton/island/blob/master/island-data_saleacres.ipynb)
  
  
### Estimation  
 - [main models](https://nbviewer.jupyter.org/github/DouglasPatton/island/blob/master/island-data_main.ipynb)
 - [effects (marginal and treatment)](https://nbviewer.jupyter.org/github/DouglasPatton/island/blob/master/island-data_effects.ipynb)
 - [sem with more neighbors](https://nbviewer.jupyter.org/github/DouglasPatton/island/blob/master/island-data_main_other-sem.ipynb)
 
### Under the Hood
Python Files
 - [manipulate the data and call the models](https://github.com/DouglasPatton/island/blob/master/island.py)
 - [create spatial weights and run models](https://github.com/DouglasPatton/island/blob/master/models.py)
 - [create visualizations](https://github.com/DouglasPatton/island/blob/master/data_viz.py)
 - [estimate marginal and treatment effects](https://github.com/DouglasPatton/island/blob/master/island_effects.py)
