import csv
import os
import logging

class Helper:
    def __init__(self):
        self.logger=logging.getLogger(__name__)
        self.logger.debug('Helper object started')
            
    def getcsvfile(self,filename):
        thisdir=os.getcwd()
        datadir=os.path.join(thisdir,'fishfiles',filename)
        #if os.path.exists(datadir):
        
        with open(datadir, 'r') as f:
            datadict=[row for row in csv.DictReader(f)]
        print(f'opening {filename} with length:{len(datadict)} and type:{type(datadict)}')
        self.logger.info(f'opening {filename} with length:{len(datadict)} and type:{type(datadict)}')
        keylist=[key for key in datadict]
        self.logger.debug(f'for filestem:{os.path.split(filename)[1]} keylist:{keylist}')
        return datadict
    
    def printDFtoSumStats(self,df=None,varlist=None):
        if df is None:
            df=self.df
        if varlist is None:
            varlist=self.varlist

        
    
    def print_model_save(self,filename=None,directory=None,model_save_list=None,shortlist=[],species=''):
        pd.set_option('display.max_colwidth', -1)
        if model_save_list is None:
            
            if directory is None:
                directory=os.getcwd()
            if filename==None:
                filename='model_save'
            
            file_loc=os.path.join(directory,filename)
            for i in range(10):
                try:
                    exists=os.path.exists(file_loc)
                    if not exists:
                        print(f'file:{file_loc} has os.path.exists value:{exists}')
                        return
                    with open(file_loc,'rb') as model_save:
                        model_save_list=pickle.load(model_save)
                except:
                    if i==9:
                        self.logger.info(f'could not open{file_loc}')
                        self.logger.exception(f'error in {__name__}')
                        return
        if len(model_save_list)==0:
            self.logger.info(f'no models in model_save_list for printing')
            return
        for model_save in model_save_list:
            if type(model_save['loss']) is str:
                self.logger.warning(f'model_save has string loss: {model_save}')
        
        try:
            model_save_list.sort(key=lambda savedicti: savedicti['loss']/savedicti['naiveloss'])
            
        except:
            model_save_list.sort(key=lambda savedicti: savedicti['loss'])
              #sorts by loss

        output_loc=os.path.join(directory,'output')
        if not os.path.exists(output_loc):
            os.mkdir(output_loc)

        filecount=len(os.listdir(output_loc))
        output_filename = os.path.join(output_loc,f'{species}'+'.html')
        output_filename=self.helper.getname(output_filename)

        modeltablehtml=''
        #keylist = ['loss','params', 'modeldict', 'when_saved', 'datagen_dict']#xdata and ydata aren't in here
        
        self.logger.info(f'len(model_save_list:{len(model_save_list)}')
        for j,model in enumerate(model_save_list):
            keylistj=[key for key in model]
            simpledicti=self.myflatdict(model,keys=keylistj)
            this_model_df=pd.DataFrame(simpledicti)
            if shortlist:
                print_this_model_df=this_model_df.loc[:, this_model_df.columns.isin(shortlist)].T
            else:
                print_this_model_df=this_model_df.T
            
            this_model_html_string=print_this_model_df.to_html()
            modeltablehtml=modeltablehtml+f'model:{j+1}<br>'+this_model_html_string+"<br>"
        for i in range(10):
            try:
                with open(output_filename,'w') as _htmlfile:
                    _htmlfile.write(modeltablehtml)
                return
            except:
                if i==9:
                    self.logger.critical(f'could not write modeltablehtml to location:{output_filename}')
                    self.logger.exception(f'error in {__name__}')
                    return