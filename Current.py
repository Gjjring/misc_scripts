import numpy as np
from scipy.interpolate import pchip_interpolate
from csv import reader
import pdb
import inspect, os

def opencsv(filename):
    file_object = open(filename)
    raw = reader(file_object, delimiter=';')
    data = np.array(list(raw))
    data = np.delete(data,0,0)
    data = data.astype(np.float)
    return data

def getCurrent_trapz(wl_abs,abs_data,wl_min,wl_max, theta_in):
    cdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory    
    am15data = opencsv(os.path.join(cdir,'ASTMG173.csv'))
    wl_am15 = am15data[:,0]
    e = 1.6021765e-19
    h = 6.62607004e-34    
    c = 299792458
    am15jphdata = np.multiply(wl_am15,am15data[:,2])
    am15jphdata *= 1.e-9*e/(h*c)/10
    am15jphdata *= np.cos(theta_in) #Correct for incident angles different from 0
    #print "am15 shape " + str(np.shape(wl_am15))
    #print "wl_abs shape " + str(np.shape(wl_abs))
    #print "abs_data shape " + str(np.shape(abs_data))
    abs_fine=pchip_interpolate(wl_abs, abs_data,wl_am15)
    #abs_fine = abs_spline(wl_am15)
    abs_int_data = np.multiply(am15jphdata,abs_fine)
    n_wl = wl_am15.size
    for i in range(0, n_wl):
        if wl_am15[i] < wl_min:
            abs_int_data[i] = 0
        elif wl_am15[i] > wl_max:
            abs_int_data[i] = 0
             
    j_abs = np.trapz(abs_int_data,wl_am15)
    return j_abs

def CalcCurrentWithDerivative(data,Name,AbsorptionCorrection = False):
    #data = np.load('AbsorptionData_Thickness_%.i' % inputvalue*100).item()
    wl_abs = data['wavelength']
    abs_data1 = data['Abs_'+Name]
    abs_data2 = data['d_Abs_'+Name]
    if len(abs_data1) == 2:
        abs_data1 = (abs_data1[0]+abs_data1[1])/2
    else:
        abs_data1 = abs_data1[0]
    if len(abs_data2) == 2:
        abs_data2 = (abs_data2[0]+abs_data2[1])/2
    else:
        abs_data2 = abs_data2[0]
    if AbsorptionCorrection:
        abs_data1= CorrectAbsorption(wl_abs,abs_data1,Name,"Tiedje")
        abs_data2= CorrectAbsorption(wl_abs,abs_data2,Name,"Tiedje")
    wl_min = min(wl_abs)
    wl_max = max(wl_abs)    
    theta_in = 0    
    Jsc = getCurrent_trapz(wl_abs,abs_data1,wl_min,wl_max,theta_in)
    dJsc = getCurrent_trapz(wl_abs,abs_data2,wl_min,wl_max,theta_in)    
    return (Jsc,dJsc)

def CalcCurrent(data,Name,AbsorptionCorrection = False):
    #data = np.load('AbsorptionData_Thickness_%.i' % inputvalue*100).item()
    wl_abs = data['wavelength']
    abs_data1 = data['Abs_'+Name]
    if len(abs_data1) == 2:
        abs_data1 = (abs_data1[0]+abs_data1[1])/2
    else:
        abs_data1 = abs_data1[0]
    if AbsorptionCorrection:
        abs_data1= CorrectAbsorption(wl_abs,abs_data1,Name,"Tiedje")
    wl_min = min(wl_abs)
    wl_max = max(wl_abs)    
    theta_in = 0    
    Jsc = getCurrent_trapz(wl_abs,abs_data1,wl_min,wl_max,theta_in)
    dJsc = getCurrent_trapz(wl_abs,abs_data2,wl_min,wl_max,theta_in)    
    return (Jsc,dJsc)
    
def CorrectAbsorption(wl_abs,abs_data,Name,Type):
    data = np.loadtxt('materials/' + Name + '_nk.txt')
    
    n = pchip_interpolate(data[:,0], data[:,1], wl_abs)
    k = pchip_interpolate(data[:,0], data[:,2], wl_abs)
    t = 160*1e-6    
    t_array = np.ones((len(wl_abs),))*t
    alpha = 4*np.pi * k / (wl_abs*1e-9);
    if Type == "Tiedje":
        return abs_data * alpha / (alpha + (1/(4*(n**2)*t_array)))
    elif Type == "2Pass":
        Tout = abs_data * np.exp(-alpha*2*t_array)    
        return abs_data-Tout


