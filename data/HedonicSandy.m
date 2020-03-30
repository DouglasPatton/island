%%  Muluken Muche
%      This program runs spatial error model for hedonic analysis related to hurricane Sandy in Nassau
%      County, Long Island, Newyork

%  This code is created partly based on code from Patrick Walsh, Megan kung, and Dennis Guignet
%  The SEM model is adoped from James P. LeSage
% (https://www.spatial-econometrics.com/), which is publicly available resource

clear all; close all, clc;
tic
DataF = 'C:\Users\Mmuche\Desktop\SEM_hedonic';
suffix = '02-25-2020';
ResultsF = 'C:\Users\Mmuche\Desktop\SEM_hedonic';
cd(DataF);
NYData = 'PostSandy.csv';
importfile1(NYData);

%%

Nobs = length(saleprice);
Constant = ones(Nobs,1);

% Setup basic RHS variable set
YearDums = [];                             namesYearDums1 = {};
%YearDums = [YearDums dv_2002];             namesYearDums1 = [namesYearDums1, 'dv_2002'];
%YearDums = [YearDums dv_2004];             namesYearDums1 = [namesYearDums1, 'dv_2004'];
%YearDums = [YearDums dv_2005];             namesYearDums1 = [namesYearDums1, 'dv_2005'];
%YearDums = [YearDums dv_2006];             namesYearDums1 = [namesYearDums1, 'dv_2006'];
%YearDums = [YearDums dv_2007];             namesYearDums1 = [namesYearDums1, 'dv_2007'];
%YearDums = [YearDums dv_2008];             namesYearDums1 = [namesYearDums1, 'dv_2008'];
%YearDums = [YearDums dv_2009];             namesYearDums1 = [namesYearDums1, 'dv_2009'];
%YearDums = [YearDums dv_2010];             namesYearDums1 = [namesYearDums1, 'dv_2010'];
%YearDums = [YearDums dv_2011];             namesYearDums1 = [namesYearDums1, 'dv_2011'];
YearDums = [YearDums dv_2012];             namesYearDums1 = [namesYearDums1, 'dv_2012'];
YearDums = [YearDums dv_2013];             namesYearDums1 = [namesYearDums1, 'dv_2013'];
YearDums = [YearDums dv_2014];             namesYearDums1 = [namesYearDums1, 'dv_2014'];
%YearDums = [YearDums dv_2015];             namesYearDums1 = [namesYearDums1, 'dv_2015'];
namesYearDums = char( namesYearDums1);

structural = [];                             namesStructural1 = {};
structural = [structural, bayfront];         namesStructural1 = [namesStructural1, 'bayfront'];
%structural = [structural, fullbathrooms];    namesStructural1 = [namesStructural1, 'fullbathrooms'];
%structural = [structural, halfbathrooms];    namesStructural1 = [namesStructural1, 'halfbathrooms'];
%structural = [structural, parcel_area];      namesStructural1 = [namesStructural1, 'parcel_area'];
structural = [structural, wateraccess];      namesStructural1 = [namesStructural1, 'wateraccess'];
structural = [structural, totalbathrooms];   namesStructural1 = [namesStructural1, 'totalbathrooms'];
structural = [structural, totallivingarea];  namesStructural1 = [namesStructural1, 'totallivingarea'];
structural = [structural, saleacres];        namesStructural1 = [namesStructural1, 'saleacres'];
namesStructural = char(namesStructural1);
 
location = [];                                        namesLocation1={};
location = [location, distance_park];                 namesLocation1=[namesLocation1, 'distance_park'];
%location = [location, distance_shoreline];            namesLocation1=[namesLocation1, 'distance_shoreline'];
%location = [location, parcel_distance];               namesLocation1=[namesLocation1, 'parcel_distance'];
location = [location, distance_nyc];                  namesLocation1=[namesLocation1, 'distance_nyc'];
location = [location, distance_golf];                 namesLocation1=[namesLocation1, 'distance_golf'];
location = [location, shorelinedistancedv3_1000];     namesLocation1=[namesLocation1, 'shorelinedistancedv3_1000'];
location = [location, shorelinedistancedv1000_2000];  namesLocation1=[namesLocation1, 'shorelinedistancedv1000_2000'];
location = [location, shorelinedistancedv2000_3000];  namesLocation1=[namesLocation1, 'shorelinedistancedv2000_3000'];
% location = [location, shorelinedistancedv3000_4000];  namesLocation1=[namesLocation1, 'shorelinedistancedv3000_4000'];
namesLocation = char(namesLocation1);
 
Demography = [];                            namesDemography1={};
Demography = [Demography, education];       namesDemography1=[namesDemography1, 'education'];
Demography = [Demography, income];          namesDemography1=[namesDemography1, 'income'];
%Demography = [Demography, nl_income];          namesDemography1=[namesDemography1, 'nl_income'];
Demography = [Demography, povertylevel];    namesDemography1=[namesDemography1, 'povertylevel'];
Demography = [Demography, pct_white];       namesDemography1=[namesDemography1, 'pct_white'];
%Demography = [Demography, pct_black];       namesDemography1=[namesDemography1, 'pct_black'];
namesDemography = char(namesDemography1);
 
WQshoreDistance = [];                                               namesWQshoreDistance1={};
WQshoreDistance = [WQshoreDistance, wqshorelinedistancedv3_1000];     namesWQshoreDistance1=[namesWQshoreDistance1, 'wqshorelinedistancedv3_1000'];
WQshoreDistance = [WQshoreDistance, wqshorelinedistancedv1000_2000];  namesWQshoreDistance1=[namesWQshoreDistance1, 'wqshorelinedistancedv1000_2000'];
WQshoreDistance = [WQshoreDistance, wqshorelinedistancedv2000_3000];  namesWQshoreDistance1=[namesWQshoreDistance1, 'wqshorelinedistancedv2000_3000'];
% WQshoreDistance = [WQshoreDistance, wqshorelinedistancedv3000_4000];  namesWQshoreDistance1=[namesWQshoreDistance1, 'wqshorelinedistancedv3000_4000'];
% 
namesWQshoreDistance = char(namesWQshoreDistance1);


% Preliminary definition

RHS1 = [Constant, structural, location, secchi, Demography, YearDums, WQshoreDistance, wqbayfront, wqwateraccess];
RHS1names1 = ['lnsaleprice' 'Constant' namesStructural1 namesLocation1 'secchi' namesDemography1 namesYearDums1 namesWQshoreDistance1 'wqbayfront' 'wqwateraccess'];

RHS1names = char(RHS1names1);
% Estimate spatial weights matrices
% longitude considered the x coordinate
% Nearest neighbor


%%
NN10 = nnSWM(longitude,latitude,10,1,1);
NN15 = nnSWM(longitude,latitude,15,1,1);
NN20 = nnSWM(longitude,latitude,20,1,1);

prefix = 'HedonicReg';
Intermedprefix = 'HedonicIntermed';
Pregprefix = 'HedonicPreReg';
FinalPrefix = 'HedonicPostReg';

   
SWMs = {NN10 NN15 NN20};
SWMnames = {'NN10' 'NN15' 'NN20'};

sems = {};
cd(ResultsF);

%%
resultsNameSEM = strcat('LnPostSandy_SEM',suffix,'.txt');


fidNSEM = fopen(resultsNameSEM, 'a');
% Begin loop over SWMs
for t=1:length(SWMs)
    disp(SWMnames{t})
    % Output results to a file
    % setup name of regression to be displayed
    RegNameSEM = sprintf('SEM  SWM1: %s', SWMnames{t});
   
    fprintf(fidNSEM, RegNameSEM);
    
    % Run regressions
    sems.(SWMnames{t}) = sem(lnsaleprice,RHS1,SWMs{t});
    prt(sems.(SWMnames{t}),RHS1names,fidNSEM);
    
end

fclose('all');

toc



