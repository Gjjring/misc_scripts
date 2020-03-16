"""
Script for taking loading a pandas dataframe and plotting data in either 1D or
2D, depending on defined critera.
"""

import os, sys
import numpy as np
import pandas
import argparse
import matplotlib
import matplotlib.ticker as ticker
import pickle
import copy
# Force matplotlib to not use any Xwindows backend.
#if 'DISPLAY' not in os.environ:
#    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler
plt.rcParams['image.cmap'] = 'magma'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

from matplotlib import cm
from cycler import cycler
Dark2 = cm.get_cmap('Dark2', 12)
Paired = cm.get_cmap('Paired',12)
colors = []
#for row in [0,2,3,5,6,7,9,10]:
    #colors.append(Dark2.colors[row,:])
for row in range(8):
    colors.append(Paired.colors[row,:])
plt.rcParams['axes.prop_cycle'] = cycler('color',colors)


parser = argparse.ArgumentParser(description='plot simulations that have been run')
parser.add_argument('DataSetName',metavar='DataSet',nargs='+',type=str,help ='Name of the dataset to plot')
parser.add_argument('--IV',metavar='IndependentVariables',nargs='+',default=['vacuum_wavelength'],help='indepedent variables when plotting')
parser.add_argument('--DV',metavar='DepdentVariables',nargs='+',default=['conservation_1'],help='dependent variables when plotting')
parser.add_argument('--SV',metavar='SlicedVariables',nargs='+',default=[],help='independent variables that should be sliced not plotted')
parser.add_argument('--stacked',action="store_true",help='whether to plot curves stacked, only valid for 1 Independent Variable')
parser.add_argument('--logy',action="store_true",help='whether the y axis should be log scale')
parser.add_argument('--loglog',action="store_true",help='whether the both axes should be log scale')
parser.add_argument('--display_max',action="store_true",help='display maximum value and argument')
parser.add_argument('--savefig',action="store_true",help='save figure to file')
parser.add_argument('--noplot',action="store_true",help='show the plot')
parser.add_argument('--averageDVs',action="store_true",help="take the average of given dependent variables")
parser.add_argument('--sumDVs',action="store_true",help="sum the given dependent variables")
parser.add_argument('--exponentiateDVs',action="store_true",help="sum the given dependent variables")
@ticker.FuncFormatter
def format_func_GM(value,tick_number):
    if value == 0:
        return r"$\Gamma$"
    elif value == 60:
        return r"$M \leftarrow $"
    else:
        return "%i" % value

@ticker.FuncFormatter
def format_func_GK(value,tick_number):
    if value == 0:
        return r"$\Gamma$"
    elif value == 60:
        return r"$\rightarrow K$"
    else:
        return "%i" % value


@ticker.FuncFormatter
def format_func_GM_Left(value,tick_number):
    if value == 0:
        return r"$\Gamma$"
    elif value == 60:
        return r"$M \leftarrow $"
    else:
        return "%i" % value

@ticker.FuncFormatter
def format_func_GM_Right(value,tick_number):
    if value == 0:
        return r"$\Gamma$"
    elif value == 60:
        return r"$\rightarrow M$"
    else:
        return "%i" % value

@ticker.FuncFormatter
def format_func_GK_Right(value,tick_number):
    if value == 0:
        return r"$\Gamma$"
    elif value == 60:
        return r"$\rightarrow K$"
    else:
        return "%i" % value

@ticker.FuncFormatter
def format_func_GK_Left(value,tick_number):
    if value == 0:
        return r"$\Gamma$"
    elif value == 60:
        return r"$K \leftarrow$"
    else:
        return "%i" % value

def makeArgs(argDict):
    args = []
    args.append(argDict['DataSetName'])
    if 'IV' in argDict:
        args.append('--IV')
        for ii in range(len(argDict['IV'])):
            args.append(argDict['IV'][ii])
    if 'DV' in argDict:
        args.append('--DV')
        for ii in range(len(argDict['DV'])):
            args.append(argDict['DV'][ii])
    if 'SV' in argDict:
        args.append('--SV')
        for item in argDict['SV'].items():
            args.append("{}={}".format(item[0],item[1]))
    args += argDict['extraArgs']
    return args



def plotData(args,axes=None):
    if not axes:
        axes = []
    else:
        axcount = 0;
    args = parser.parse_args(args)
    IndepVars = args.IV
    DepVars = args.DV
    slicedVars = []
    for i in range(len(args.SV)):
        slicer = args.SV[i].split('=')
        if slicer[0] in IndepVars:
            raise ValueError('sliced variables cannot independent variables')
        slicedVars.append((slicer[0],float(slicer[1])))
    projectNames = args.DataSetName
    if projectNames[0] == 'PlotData.py':
        projectNames.pop(0)
    #DataFrames = []
    for pN in projectNames:
        if pN.endswith('.df'):
            with open('results/{}'.format(pN),'r') as f:
                DataFrame = pickle.load(f)
        elif pN.endswith('.csv'):
            DataFrame = pandas.read_csv('results/{}'.format(pN))
        else:
            raise ValueError("projectName {} invalid".format(pN))

        if "phi" in DataFrame.columns:
            df_phis = np.unique(DataFrame.phi.values)
            for sv in slicedVars:
                if sv[0] == 'phi':
                    phi = sv[1]
                    if phi == 90.0 and phi not in df_phis:
                        DataFrame.loc[ DataFrame.phi.values==30.0,'phi'] = 90.0



        DFNames = DepVars;
        Alliases = DepVars;
        DimensionNames = IndepVars
        Dimensions = {}
        DFCols = list(DataFrame)
        '''
        UnusedVars = []
        for i in range(len(DFCols)):
            if not any(ext in DFCOls[i] for ext in IndepVars):
                UnusedVars.append(DFCols[i])
        '''
        for i in range(len(DimensionNames)):
            dimensionNumpy = DataFrame[DimensionNames[i]].values
            uniqueNumpy = np.unique(dimensionNumpy);
            Dimensions[DimensionNames[i]] = uniqueNumpy;


        #Dimensions['vacuum_wavelengths'] = np.arange(400,710,10)*1e-9
        #Dimensions['theta'] = np.arange(0.0,82.5,2.5)
        #Dimensions['phi'] = np.array([0.,45.0])

        DimensionLengths = []
        for i in range(len(DimensionNames)):
            DimensionLengths.append(len(Dimensions[DimensionNames[i]]))

        Arrays = {}
        for i in range(len(Alliases)):
            Arrays[Alliases[i]] = np.zeros(DimensionLengths)

        #print(slicedVars)
        #print(DataFrame.head(5))
        DataFrame = SliceDataFrame(DataFrame,slicedVars)
        CreateArrays(DataFrame,copy.copy(IndepVars),Arrays,DFNames,Alliases,DimensionLengths)


        if len(Dimensions) == 1:
            if axes is None:
                f,ax = plt.subplots()
                axes.append(ax)
            else:
                plt.sca(axes[axcount])
                axcount +=1
            ''' 1-D Data '''
            if IndepVars[0] == 'vacuum_wavelength':
                Dimensions[IndepVars[0]] *= 1e9


            if args.averageDVs == False and args.sumDVs is False:
                if args.stacked:
                    stackedData = np.zeros( (len(DepVars),Dimensions[IndepVars[0]].size))
                    plotLabels = []
                iDV = 0
                for dv in DepVars:
                    if args.logy:
                        #plt.semilogy(Dimensions[IndepVars[0]],Arrays[dv],'o')
                        plt.semilogy(Dimensions[IndepVars[0]],np.power(10,Arrays[dv]))
                    elif args.loglog:
                        plt.loglog(Dimensions[IndepVars[0]],Arrays[dv])
                    else:
                        if args.stacked:
                            stackedData[iDV,:] = Arrays[dv]
                            plotLabels.append(dv)
                            iDV += 1
                        else:
                            plt.plot(Dimensions[IndepVars[0]],Arrays[dv])
                if args.stacked:
                    plt.stackplot(Dimensions[IndepVars[0]],stackedData,
                                  labels=plotLabels)
            else:
                y = np.zeros(Arrays[DepVars[0]].shape)
                for dv in DepVars:
                    if args.averageDVs:
                        y += Arrays[dv]/len(DepVars)
                    elif args.sumDVs:
                        y += Arrays[dv]
                if args.logy:
                    plt.semilogy(Dimensions[IndepVars[0]],np.power(10,y))
                elif args.loglog:
                    plt.loglog(Dimensions[IndepVars[0]],y)
                else:
                    plt.plot(Dimensions[IndepVars[0]],y)



            plt.grid(True)
            plt.xlabel(IndepVars[0])
            plt.ylabel(DepVars[0])
            if args.savefig:
                plt.savefig('figures/{}.pdf'.format(".".join(projectNames[0].split('.')[0:-1])))
            if args.display_max:
                print(np.max(Arrays[DepVars[0]]))
                print(Dimensions[IndepVars[0]][np.argmax(Arrays[DepVars[0]])])

            if 'DISPLAY' in os.environ and not args.noplot:
                plt.show()


        elif len(Dimensions) == 2:
            #[X,Y] = np.meshgrid( Dimensions[IndpVars[0]], Dimensions[IndpVars[1]])
            if IndepVars[0] == 'vacuum_wavelength':
                Dimensions[IndepVars[0]] *= 1e9
            if IndepVars[1] == 'vacuum_wavelength':
                Dimensions[IndepVars[1]] *= 1e9
            xmin = np.min(Dimensions[IndepVars[0]])
            xmax = np.max(Dimensions[IndepVars[0]])
            ymin = np.min(Dimensions[IndepVars[1]])
            ymax = np.max(Dimensions[IndepVars[1]])
            X,Y = np.meshgrid(Dimensions[IndepVars[0]],Dimensions[IndepVars[1]])
            Extent = (xmin,xmax,ymin,ymax)
            if args.averageDVs == False and args.sumDVs == False:
                for dv in DepVars:
                    if axes is None:
                        f,ax = plt.subplots()
                        axes.append(ax)
                    else:
                        plt.sca(axes[axcount])
                        axcount +=1
                    #plt.imshow(np.transpose(Arrays[dv]),extent=Extent,aspect='auto',origin='lower')
                    z = np.transpose(Arrays[dv][:-1,:-1])
                    plt.pcolormesh(X,Y,z)
                    plt.xlim(xmin,xmax)
                    plt.ylim(ymin,ymax)
                    #range(np.shape(Arrays[dv])[0]):
                    if args.savefig:
                        #print pN
                        #print pN.split('.')
                        #print pN.split('.')[0:-1]
                        #print ".".join(projectNames[0].split('.')[0:-1])
                        plt.savefig('figures/{}_{}.pdf'.format(".".join(projectNames[0].split('.')[0:-1]),dv))
                    if 'DISPLAY' in os.environ and not args.noplot:
                        plt.show()
            else:
                z = np.zeros(Arrays[DepVars[0]].shape)

                for dv in DepVars:
                    z_contribution = Arrays[dv]
                    if args.exponentiateDVs:
                        z_contribution = np.power(10,z_contribution)
                    if args.averageDVs:
                        z += z_contribution/len(DepVars)
                    elif args.sumDVs:
                        z += z_contribution

                if not axes:
                    f,ax = plt.subplots()
                    axes.append(ax)
                else:
                    plt.sca(axes[0])
                #plt.imshow(np.transpose(z),extent=Extent,aspect='auto',origin='lower')
                if args.exponentiateDVs:
                    z = np.log10(z)
                plt.pcolormesh(X,Y,np.transpose(z[:-1,:-1]))
                plt.xlim(xmin,xmax)
                plt.ylim(ymin,ymax)
                #range(np.shape(Arrays[dv])[0]):
                if args.savefig:
                    #print pN
                    #print pN.split('.')
                    #print pN.split('.')[0:-1]
                    #print ".".join(projectNames[0].split('.')[0:-1])
                    plt.savefig('figures/{}_{}.pdf'.format(".".join(projectNames[0].split('.')[0:-1]),dv))
                if 'DISPLAY' in os.environ and not args.noplot:
                    plt.show()


def SliceDataFrame(DFrame,slicedVars):
    for i in range(len(slicedVars)):
        DFrame = DFrame[np.isclose(DFrame[slicedVars[i][0]],slicedVars[i][1],rtol=1e-12,atol=1e-20)].copy()
    return DFrame


def CreateArrays(DFrame,Vars,Arrays,DFNames,Alliases,DimensionLengths):
    #print Vars
    DFrame = DFrame.sort_values(by=Vars)
    for a in range(len(Alliases)):
        Arrays[Alliases[a]] = DFrame[DFNames[a]].values.reshape(DimensionLengths)
    return Arrays


if __name__=="__main__":
    ax = plotData(sys.argv)
