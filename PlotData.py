"""
Script for taking loading a pandas dataframe and plotting data in either 1D or
2D, depending on defined critera.
"""

import os, sys
import numpy as np
import numpy.matlib
import pandas
import argparse
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.gridspec
import pickle
import copy
from collections import OrderedDict
from Current import getCurrent_trapz
# Force matplotlib to not use any Xwindows backend.
#if 'DISPLAY' not in os.environ:
#    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler
plt.rcParams['image.cmap'] = 'magma'
#plt.rcParams['axes.titlesize'] = 20
#plt.rcParams['axes.labelsize'] = 20
#plt.rcParams['lines.linewidth'] = 2
#plt.rcParams['xtick.labelsize'] = 16
#plt.rcParams['ytick.labelsize'] = 16

from matplotlib import cm
from cycler import cycler
Dark2 = cm.get_cmap('Dark2', 12)
Paired = cm.get_cmap('Paired', 12)
colors = []
#for row in [0,2,3,5,6,7,9,10]:
    #colors.append(Dark2.colors[row,:])
for row in range(8):
    colors.append(Paired.colors[row, :])
plt.rcParams['axes.prop_cycle'] = cycler('color', colors)


parser = argparse.ArgumentParser(description='plot simulations that have been run')
parser.add_argument('DataSetName',metavar='DataSet',nargs='+',type=str,help ='Name of the dataset to plot')
parser.add_argument('--IV',metavar='IndependentVariables',nargs='+',default=['vacuum_wavelength'],help='independent variables when plotting')
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
parser.add_argument('--calcCurrent',action="store_true",help="sum the given dependent variables")
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


def gaussian(wavelengths,peak,fwhm):
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    #/(sigma*np.sqrt(np.pi*2))
    return np.exp( -(wavelengths-peak)**2 /(2.0*sigma**2))


class Plotter(object):

    def __init__(self, filename, aliases=None):
        self.filename = filename
        self.load_data()
        self.variables = {}
        self.variables['independent'] = []
        self.variables['dependent'] = []
        self.variables['sliced'] = {}

        # used for constructing arrays
        self.init_dimensions()


        # arrays to be plotted
        self.init_plot_data()


        #self.aliases['arrays'] = {}
        self.aliases = {}
        self.aliases['independent'] = {}
        self.aliases['dependent'] = {}

        self.mpl_data = {}
        #self.mpl_data['aliases'] = {}
        self.mpl_data['axes_list'] = []
        self.mpl_data['figure'] = None
        self.mpl_data['line_styles'] = {}
        self.mpl_data['colors'] = {}
        self.mpl_data['user_line_styles'] = {}
        self.mpl_data['user_colors'] = {}
        self.mpl_data['user_colormap'] = "magma"
        self.mpl_data['colorbar'] = None
        self.mpl_data['mappables'] = {}

        self.options = {}
        self.options['expand_sources'] = False
        self.options['exponentiateDVs'] = False
        self.options['logDVs'] = False
        self.options['logy'] = False
        self.options['loglog'] = False
        self.options['data_preprocess'] = None
        #self.options['log_color'] = False
        self.options['average_polarization'] = False
        self.options['average_sliced'] = False
        self.options['average_DVs'] = False
        self.options['sum_DVs'] = False
        self.options['plot_stacked'] = False
        self.options['scaling_factor'] = None
        self.options['scale_IVs'] = 1.0
        self.options['root_data'] = False
        self.options['vmin'] = None
        self.options['vmax'] = None
        self.options['colorbar'] = True
        self.options['calculate_current'] = False
        self.options['print_current'] = False
        self.options['plot_current_stack'] = False
        self.options['integrate_out_current'] = False
        self.options['plot_mesh'] = False
        self.options['average_independent_variables'] = False
        self.options['user_preprocess'] = None
        self.make_dimensions()

        #self.make_array_names()
        #self.make_arrays()
        #self.fill_arrays()
    def load_data(self):
        (head,tail) = os.path.split(self.filename)
        if head == "":
            head = 'results'
        file_path = os.path.join(head,tail)
        self.data_frame = pandas.read_csv(file_path)

    def init_plot_data(self):
        self.plot_data = {}
        self.plot_data['arrays'] = {}
        self.plot_data['aliases'] = {}


    def init_dimensions(self):
        self.dimensions = {}
        self.dimensions['vectors'] = []
        self.dimensions['names'] = []
        self.dimensions['lengths'] = []
        #self.dimensions['aliases'] = {}

    def setup_axes(self, axes_list):
        if (axes_list is None or \
                len(axes_list) == 0):
            self.mpl_data['axes_list'] = []
            f,ax = plt.subplots()
            self.mpl_data['figure'] = f
            self.mpl_data['axes_list'].append(ax)
        else:
            plt.sca(axes_list[0])
            self.mpl_data['axes_list'] = axes_list

    def normalise_indepvar_units(self):
        #self.aliases['dimensions'] = OrderedDict()
        for index, dimension in enumerate(self.dimensions['names']):
            vals = self.dimensions['vectors'][index]
            if dimension == 'vacuum_wavelength':
                self.dimensions['vectors'][index] = vals*1e9
            else:
                self.dimensions['vectors'][index] = vals

    def exponentiate_data(self):
        for name in self.plot_data['arrays']:
            self.plot_data['arrays'][name] = np.power(10,self.plot_data['arrays'][name])

    def root_data(self):
        for name in self.plot_data['arrays']:
            self.plot_data['arrays'][name] = np.sqrt(self.plot_data['arrays'][name])

    def log_data(self):
        for name in self.plot_data['arrays']:
            self.plot_data['arrays'][name] = np.log10(self.plot_data['arrays'][name])

    def apply_custom_preprocess(self):
        func = self.options['data_preprocess']
        for name in self.plot_data['arrays']:
            self.plot_data['arrays'][name] = func(self.plot_data['arrays'][name])


    def _preprocess(self, aliases, axes_list):
        self.mpl_data['axes_list'] = []
        self.init_plot_data()
        if aliases is not None:
            self.aliases['dependent'] = aliases
        self.setup_axes(axes_list)
        self.make_dimensions()
        self.normalise_indepvar_units()
        #self.make_array_names()
        if self.options['expand_sources']:
            self.add_source_to_depvars()

        self.make_arrays()
        self.fill_arrays()
        if self.options['average_polarization']:
            self.average_polarizations()

        if self.options['integrate_out_current']:
            self.convert_to_current()

        if self.options['root_data']:
            self.root_data()

        if self.options['exponentiateDVs']:
            self.exponentiate_data()

        if self.options['logDVs']:
            self.log_data()

        if self.options['average_independent_variables']:
            self.average_indepvars()

        if self.options['average_DVs']:
            self.average_depvars()
        if self.options['sum_DVs']:
            self.sum_depvars()

        if len(self.plot_data['aliases']) == 0:
            self.make_aliases()

        if self.options['data_preprocess'] is not None:
            self.apply_custom_preprocess()

        self.setup_styles()



    def convert_to_current(self):
        keys = list(self.plot_data['arrays'].keys())
        x_data = self.get_x_data() # some other variable
        y_data = self.get_y_data() # wavelengths
        for name in keys:
            data = self.plot_data['arrays'].pop(name)
            currents = np.zeros(x_data.shape)
            for i in range(data.shape[0]):
                line_data = data[i,:]
                current = getCurrent_trapz(y_data, line_data,
                                           np.amin(y_data),
                                           np.amax(y_data),
                                           0.)
                currents[i] = current
            self.plot_data['arrays'][name] = currents


    def add_source_to_depvars(self):
        total_dep_vars = []
        for depvar in self.variables['dependent']:
            total_dep_vars.append("{}_1".format(depvar))
            total_dep_vars.append("{}_2".format(depvar))
        self.variables['dependent'] = total_dep_vars

    def average_polarizations(self):
        all_keys = list(self.plot_data['arrays'].keys())
        reduced_keys = []
        for key in all_keys:
            tmp_key = key[:-2]
            if tmp_key not in reduced_keys:
                reduced_keys.append(tmp_key)
        for key in reduced_keys:
            key_1 = "{}_1".format(key)
            key_2 = "{}_2".format(key)
            data_1 = self.plot_data['arrays'].pop(key_1)
            data_2 = self.plot_data['arrays'].pop(key_2)
            self.plot_data['arrays'][key] = (data_1+data_2)/2.0


    def average_indepvars(self):
        #random_key = next(iter(self.plot_data['arrays']))
        #average_array = np.zeros(self.plot_data['arrays'][random_key].shape)
        #n_arrays = len(self.plot_data['arrays'])
        keys = list(self.plot_data['arrays'].keys())
        for name in keys:
            data = self.plot_data['arrays'].pop(name)
            average_array = data.mean(axis=-1)
            self.plot_data['arrays'][name] = average_array

    def average_depvars(self):
        random_key = next(iter(self.plot_data['arrays']))
        average_array = np.zeros(self.plot_data['arrays'][random_key].shape)
        n_arrays = len(self.plot_data['arrays'])
        keys = list(self.plot_data['arrays'].keys())
        for name in keys:
            data = self.plot_data['arrays'].pop(name)
            average_array += data/n_arrays
        self.plot_data['arrays']['average'] = average_array

    def sum_depvars(self):
        random_key = next(iter(self.plot_data['arrays']))
        sum_array = np.zeros(self.plot_data['arrays'][random_key].shape)
        keys = list(self.plot_data['arrays'].keys())
        for name in keys:
            data = self.plot_data['arrays'].pop(name)
            sum_array += data
        self.plot_data['arrays']['sum'] = sum_array

    def make_aliases(self):
        for name in self.plot_data['arrays']:
            if name in self.aliases['dependent']:
                self.plot_data['aliases'][name] = self.aliases['dependent'][name]
            else:
                self.plot_data['aliases'][name] = name

    def _postprocess(self):
        if (self.options['calculate_current'] and
            self.dimensions['names'][0] == 'vacuum_wavelength'):
            self.calc_current()

        self.mpl_data['axes_list'] = None

        if self.mpl_data['figure'] is not None:
            plt.close(self.mpl_data['figure'])

    def setup_styles(self):
        for name in self.plot_data['arrays']:
            if name in self.mpl_data['user_line_styles']:
                self.mpl_data['line_styles'][name] = self.mpl_data['user_line_styles'][name]
            else:
                self.mpl_data['line_styles'][name] = '-'
            if name in self.mpl_data['user_colors']:
                self.mpl_data['colors'][name] = self.mpl_data['user_colors'][name]
            else:
                ax = plt.gca()
                self.mpl_data['colors'][name] = next(ax._get_lines.prop_cycler)['color']

    def get_x_data(self):
        return self.dimensions['vectors'][0]

    def get_y_data(self):
        return self.dimensions['vectors'][1]

    def plot_lines(self):
        x_data = self.get_x_data()
        for name in self.plot_data['arrays']:
            ls = self.mpl_data['line_styles'][name]
            color = self.mpl_data['colors'][name]
            line_data = self.plot_data['arrays'][name]
            label = self.plot_data['aliases'][name]
            if self.options['logy']:
                plt.semilogy(x_data, line_data, ls=ls, color=color, label=label)
            elif self.options['loglog']:
                plt.loglog(x_data, line_data, ls=ls, color=color, label=label)
            else:
                plt.plot(x_data, line_data, ls=ls, color=color, label=label)

    def calc_current(self):
        currents = {}
        x_data = self.get_x_data()
        if self.options['print_current']:
            print("Current Density (mA/cm2)")
        for name in self.plot_data['arrays']:
            line_data = self.plot_data['arrays'][name]
            current = getCurrent_trapz(x_data, line_data,
                                       np.amin(x_data),
                                       np.amax(x_data),
                                       0.)
            currents[name] = current
            if self.options['print_current']:
                print("{}: {:.2f}".format(name,current))
        if self.options['plot_current_stack']:
            self.plot_current_stack(currents)

    def plot_current_stack(self,currents):
        start = 0.0
        if len(self.mpl_data['axes_list']) > 1:
            ax = self.mpl_data['axes_list'][-1]
        else:
            #gs = {'width_ratios':[1.,0.15]}
            gs = matplotlib.gridspec.GridSpec(1,2,width_ratios=[1., 0.15])
            ax = self.mpl_data['axes_list'][0]
            fig = self.mpl_data['figure']
            print(gs[0:2])
            ax.set_subplotspec(gs[0:2])
            #print(gs[0:2].get_positions(fig))
            ax.set_position(gs[:2].get_position(fig))
            fig.add_subplot(gs[1])
            ax = gs[1]

            #ax.set_subplotspec(gs[0:2])              # only necessary if using tight_layout()

        plt.sca(ax)
        for name,current in currents.items():
            color = self.mpl_data['colors'][name]
            p1 = plt.bar(0.0,current,1.0,bottom=start,
                         color=color,label="{:.2f}".format(current))
            if current > 0.1:
                text_height = start+current/2.0
                plt.text(0.0,text_height,"{:.2f}".format(current),fontsize=18,
                        horizontalalignment='center',
                        verticalalignment='center',c='w')
                start += current
        plt.xlim((-.5,0.5))
        plt.xticks([])
        ax.yaxis.tick_right()
        #ax1.yaxis.set_label_position("right")
        plt.ylim([0,start])
        ax.xaxis.set_label_position("top")
        #ax1.xaxis.set_label_postion("top")
        plt.xlabel('$J_{ph}$ (mA/cm$^{2}$)',labelpad=15.0)





    def plot_stacked(self):
        x_data = self.get_x_data()
        colors = []
        stacked_data = []
        labels = []
        for name in self.plot_data['arrays']:
            colors.append(self.mpl_data['colors'][name])
            stacked_data.append(self.plot_data['arrays'][name])
            labels.append(self.plot_data['aliases'][name])
        plt.stackplot(x_data, stacked_data, colors=colors, labels=labels)
        plt.ylim([0.,1.])

    def plot1D(self, plot_names, aliases=None, axes_list=None):
        self.variables['dependent'] = plot_names
        self._preprocess(aliases,axes_list)
        if self.options['plot_stacked']:
            self.plot_stacked()
        else:
            self.plot_lines()
        self.dimension_label('x')
        self.tight_limits('x')
        self._postprocess()



    def gaussian_average_data(self, peak, fwhm):

        wvls = self.get_y_data()
        angles = self.get_x_data()
        illum = gaussian(wvls,peak,fwhm)
        illum = illum / np.trapz(illum,wvls)
        illum2D = np.matlib.repmat(illum,angles.size,1)
        self.plot_data['gaussian_average'] = {}
        for item in self.plot_data['arrays'].items():
            print(illum2D.shape, item[1].shape)
            averaged = illum2D*item[1]
            integrated = np.trapz(averaged,wvls,axis=1)
            self.plot_data['gaussian_average'][item[0]] = integrated

    def tight_limits(self,axis):
        x_data = self.get_x_data()
        x_min = np.amin(x_data)
        x_max = np.amax(x_data)
        plt.xlim((x_min,x_max))

    def dimension_label(self,axis):
        if axis == 'x':
            index = 0
            function = plt.xlabel
        elif axis == 'y':
            index = 1
            function = plt.ylabel

        try:
            simple_name = self.dimensions['names'][index]
            function(self.aliases['independent'][simple_name])
        except KeyError as e:
            function(simple_name)

    def _preprocess2D(self):
        x_data = self.get_x_data()
        y_data = self.get_y_data()
        xmin = np.min(x_data)
        xmax = np.max(x_data)
        ymin = np.min(y_data)
        ymax = np.max(y_data)
        X, Y = np.meshgrid(x_data, y_data)
        self.dimensions['tensors'] = {}
        self.dimensions['tensors']['X'] = X
        self.dimensions['tensors']['Y'] = Y
        self.dimensions['extent'] = (xmin, xmax, ymin, ymax)

    def plot_mesh(self):
        axes_iter = 0

        if (not len(self.plot_data['arrays']) ==
            len(self.mpl_data['axes_list'])):
            raise ValueError("not enough axes for plots")

        for name in self.plot_data['arrays']:
            ax = self.mpl_data['axes_list'][axes_iter]
            plt.sca(ax)
            label = self.plot_data['aliases'][name]
            data = self.plot_data['arrays'][name].T
            #if self.options['log_color']:
            #    data = np.log10(data)
            X = self.dimensions['tensors']['X']
            Y = self.dimensions['tensors']['Y']
            mesh = plt.pcolormesh(X, Y, data[:-1, :-1],
                                 vmin=self.options['vmin'],
                                 vmax=self.options['vmax'],
                                 cmap=self.mpl_data['user_colormap'])
            self.mpl_data['mappables'][name] = mesh
            self.dimension_label('x')
            self.dimension_label('y')
            if self.options['colorbar']:
                self.mpl_data['colorbar'] = plt.colorbar()
            plt.title(label)
            axes_iter += 1


    def plot_image(self):
        axes_iter = 0

        if (not len(self.plot_data['arrays']) ==
            len(self.mpl_data['axes_list'])):
            raise ValueError("not enough axes for plots")

        for name in self.plot_data['arrays']:
            ax = self.mpl_data['axes_list'][axes_iter]
            plt.sca(ax)
            label = self.plot_data['aliases'][name]
            data = self.plot_data['arrays'][name].T
            #if self.options['log_color']:
            #    data = np.log10(data)
            image = plt.imshow(data, extent=self.dimensions['extent'],
                               vmin=self.options['vmin'],
                               vmax=self.options['vmax'],
                               cmap=self.mpl_data['user_colormap'],
                               origin='lower')
            self.mpl_data['mappables'][name] = image
            self.dimension_label('x')
            self.dimension_label('y')
            if self.options['colorbar']:
                self.mpl_data['colorbar'] = plt.colorbar()
            plt.title(label)
            axes_iter += 1

    def plot2D(self, plot_names, aliases=None, axes_list=None):
        self.variables['dependent'] = plot_names
        self._preprocess(aliases,axes_list)
        self._preprocess2D()
        if self.options['plot_mesh']:
            self.plot_mesh()
        else:
            self.plot_image()


        #self.dimension_label('x')
        #self.dimension_label('y')

        self._postprocess()


    def make_legend(self,bbox_to_anchor=None):
        if bbox_to_anchor == None:
            bbox_to_anchor = [1.0,0.5]
        plt.legend(loc='center left',bbox_to_anchor=bbox_to_anchor,fontsize=18)

    def make_dimensions(self):
        # make dimension names and vectors
        self.init_dimensions()
        #self.dimensions = OrderedDict()
        #self.dimension_lengths = []
        for index,indep_var in enumerate(self.variables['independent']):
            dimension_values = self.data_frame[indep_var].values
            unique_values = np.unique(np.round(dimension_values, 12))
            unique_values = unique_values[~np.isnan(unique_values)]*self.options['scale_IVs']
            self.dimensions['vectors'].append( np.array(unique_values))
            self.dimensions['lengths'].append(unique_values.size)
            self.dimensions['names'].append(indep_var)
        if len(self.aliases['independent']) == 0:
            for index,indep_var in enumerate(self.dimensions['names']):
                self.aliases['independent'][indep_var] = indep_var

    """
    def make_array_names(self):
        #self.array_names = []
        for index,dep_var in enumerate(self.variables['dependent']):
            if dep_var in self.aliases['arrays']:
                self.array_names.append(self.aliases['arrays'][dep_var])
            else:
                self.array_names.append(dep_var)
    """

    def make_arrays(self):
        # make empty tensors of rank 1 or 2 for each dep_var
        self.plot_data['arrays'] = {}
        for index,name in enumerate(self.variables['dependent']):
            self.plot_data['arrays'][name] = np.zeros(self.dimensions['lengths'])

    def fill_arrays(self):
        # slice all the sliced vars
        sliced_frame = self.slice_dataframe()
        # write the data into the arrays
        order = copy.copy(self.variables['independent'])
        sliced_frame = sliced_frame.sort_values(by=order)
        for name in self.plot_data['arrays']:
            #dep_var = self.variables['dependent'][index]
            sliced_vals = sliced_frame[name].values
            values = sliced_vals.reshape(self.dimensions['lengths'])
            if self.options['scaling_factor'] is not None:
                values *= self.options['scaling_factor']
            if self.options['user_preprocess'] is not None:
                values = self.options['user_preprocess'](values)
            self.plot_data['arrays'][name] = values


    def slice_dataframe(self):
        data_frame = self.data_frame.copy()
        #print(self.plot_data)
        for sliced_var,value in self.variables['sliced'].items():
            if isinstance(value, str):
                data_frame = data_frame[data_frame[sliced_var] == value]
            else:
                data_frame = data_frame[np.isclose(data_frame[sliced_var],
                                                   value,
                                                   rtol=1e-12, atol=1e-20)]
        for indep_var in self.variables['independent']:
            data_frame = data_frame[~ np.isnan(data_frame[indep_var])]

        return data_frame

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
                    if args.logy:
                        z = np.log10(np.transpose(Arrays[dv][:-1,:-1]))
                    else:
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
