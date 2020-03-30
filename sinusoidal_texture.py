"""
defines help functions for constructing a sinusoidal texture for meshing
a layer interface with the finite element solver JCMsuite.
"""

import numpy as np
import scipy.spatial
from scipy.interpolate import LinearNDInterpolator

def get_texture(pitch,aspect_ratio,phase):
    '''
    returns the string holding a python evaluation for the texture height Z at
    each position in the x-y plane. Positions are evaluated by the tuple X,
    X[0] is the x position and X[1] is the y position.
    '''
    factor = np.pi * 2/(np.sqrt(3) * pitch)
    phase = np.pi*phase
    phi = np.abs(phase)
    h_peak_valley = get_peak_valley(phase)
    amplitude = pitch*aspect_ratio
    ampl = amplitude / h_peak_valley
    string = "_ampl_ * (cos(X[0] * _factor_+_phase_) * " + \
             "cos(0.5 * _factor_ * (X[0] + sqrt(3.) * X[1])) * " + \
             "cos(0.5 * _factor_ * (X[0] - sqrt(3.) * X[1])))"
    string = string.replace('_factor_', str(factor))
    string = string.replace('_ampl_', str(ampl))
    string = string.replace('_phase_', str(phase))

    hmin = np.abs(0.75*np.cos(phi/3.+2.*np.pi/3.)+0.25*np.cos(phi))
    shift_upper = hmin*ampl
    shift_lower = -shift_upper
    return string,(shift_lower,shift_upper)

def get_peak_valley(phi):
    return 1.29903810568*np.sin(phi/3.+np.pi/3.)

def texture_average(pitch, aspect_ratio, phase):
    pass

def sumtriangles(xy, z, triangles ):
    """ integrate scattered data, given a triangulation
    zsum, areasum = sumtriangles( xy, z, triangles )
    In:
        xy: npt, dim data points in 2d, 3d ...
        z: npt data values at the points, scalars or vectors
        triangles: ntri, dim+1 indices of triangles or simplexes, as from
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    Out:
        zsum: sum over all triangles of (area * z at midpoint).
            Thus z at a point where 5 triangles meet
            enters the sum 5 times, each weighted by that triangle's area / 3.
        areasum: the area or volume of the convex hull of the data points.
            For points over the unit square, zsum outside the hull is 0,
            so zsum / areasum would compensate for that.
            Or, make sure that the corners of the square or cube are in xy.
    """
    # z concave or convex => under or overestimates
    npt, dim = xy.shape
    ntri, dim1 = triangles.shape
    assert npt == len(z), "shape mismatch: xy %s z %s" % (xy.shape, z.shape)
    assert dim1 == dim+1, "triangles ? %s" % triangles.shape
    zsum = np.zeros( z[0].shape )
    areasum = 0
    dimfac = np.prod( np.arange( 1, dim+1 ))
    for tri in triangles:
        corners = xy[tri]
        t = corners[1:] - corners[0]
        if dim == 2:
            area = abs( t[0,0] * t[1,1] - t[0,1] * t[1,0] ) / 2
        else:
            area = abs( np.linalg.det( t )) / dimfac  # v slow
        zsum += area * z[tri].mean(axis=0)
        areasum += area
    return (zsum, areasum)

class SinusoidalTexture(object):

    def __init__(self,pitch,aspect_ratio,phase):
        self.pitch = pitch
        self.aspect_ratio = aspect_ratio
        self.phase = phase*np.pi
        self.phi = np.abs(phase)
        self.h_peak_valley = get_peak_valley(self.phi)
        self.simple_amplitude = pitch*aspect_ratio
        self.amplitude = self.simple_amplitude / self.h_peak_valley

    def eval_texture(self,X,Y):
        factor = np.pi * 2/(np.sqrt(3) * self.pitch)
        height = (self.amplitude * np.cos(factor*X + self.phase) *
                  np.cos(0.5*factor* (X+ np.sqrt(3)*Y)) *
                  np.cos(0.5*factor* (X- np.sqrt(3)*Y)))
        return height



    def get_shift(self):
        theta = np.radians(30.0)

        xy0 = np.array([0,-np.sin(theta)*self.pitch])
        xy1 = np.array([np.cos(theta)*self.pitch,0.0])
        npoints = 100
        x = np.linspace( xy0[0],xy1[0],npoints)
        y = np.linspace( xy0[1],xy1[1],npoints)
        x_total = list(x)
        y_total = list(y)
        for step in range(1,npoints+1):
            step_size = self.pitch/npoints
            xy0 = np.array([0.,0.])
            xy1 = np.array([-np.cos(theta)*step_size,np.sin(theta)*step_size])
            x_new = x + xy1[0]*step
            y_new = y + xy1[1]*step
            x_total += list(np.array(x_new))
            y_total += list(np.array(y_new))

        x = np.array(x_total)
        y = np.array(y_total)
        xy = np.vstack([x,y]).T
        tri = scipy.spatial.Delaunay(xy)
        data = self.eval_texture(xy[:,0],xy[:,1])
        interp = LinearNDInterpolator(tri,data)
        integral = sumtriangles(xy, interp.__call__(xy), tri.simplices)
        return integral[0]/integral[1]
    """
    def get_shift(self):
        hmin = np.abs(0.75*np.cos(self.phi/3.+2.*np.pi/3.)+0.25*np.cos(self.phi))
        shift = hmin*self.amplitude
        return shift
    """

    def texture_string(self):
        factor = np.pi * 2/(np.sqrt(3) * self.pitch)
        string = "_ampl_ * (cos(X[0] * _factor_+_phase_) * " + \
                 "cos(0.5 * _factor_ * (X[0] + sqrt(3.) * X[1])) * " + \
                 "cos(0.5 * _factor_ * (X[0] - sqrt(3.) * X[1])))"
        string = string.replace('_factor_', str(factor))
        string = string.replace('_ampl_', str(self.amplitude))
        string = string.replace('_phase_', str(self.phase))
        return string
