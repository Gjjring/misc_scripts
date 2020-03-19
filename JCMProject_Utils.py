"""Collection of functions that may be used by JCM template files (*.jcmt) to
create the project or that may be useful/necessary to process the results.

Contains a default processing function (`processing_default`).

Authors: Carlo Barth, Phillip Manley

Credit: Partly based on MATLAB-versions written by Sven Burger and Martin
        Hammerschmidt.
"""

import numpy as np
from numpy.linalg import norm
from scipy import constants
import os
# from warnings import warn

Z0 = np.sqrt( constants.mu_0 / constants.epsilon_0 )

# =============================================================================

class Cone(object):
    """Cone with a specific height, diameter at center and side wall angle
    (in degrees)."""
    
    def __init__(self, diameter_center, height, angle_degrees):
        self.diameter_center = diameter_center
        self.height = height
        self.angle_degrees = angle_degrees
        self._angle = np.deg2rad(angle_degrees)
        self._rad = diameter_center/2.
        self._rad0 = self._rad - (height/2. * np.tan(self._angle))
        
    def __repr__(self):
        return 'Cone(d={}, h={}, angle={})'.format(self.diameter_center,
                                                   self.height,
                                                   self.angle_degrees)
    
    def diameter_at_height(self, height):
        """Returns the diameter of the cone at a specific height.
        """
        return 2 * (self._rad0 + height * np.tan(self._angle))



class JCM_Post_Process(object):
    """An abstract class to hold JCM-PostProcess results. Must be subclassed!"""
    STD_KEYS = []

    def __init__(self, jcm_dict, **kwargs):
        self.jcm_dict = jcm_dict
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])
        self._check_keys()
        self.title = jcm_dict['title']
        self.set_values()

    def _check_keys(self):
        """Checks if the jcm_dict represents the results of this post
        process."""
        keys_ = self.jcm_dict.keys()
        for k in self.STD_KEYS:
            if not k in keys_:
                raise ValueError('The provided `jcm_dict` is not a valid '+
                                 'PostProcess result. Key {} '.format(k)+
                                 'is missing.')

    def set_values(self):
        """Overwrite this function to set the post process specific values."""
        pass

    def __repr__(self):
        return self.title


class PP_FourierTransform(JCM_Post_Process):
    """Holds the results of a JCM-FourierTransform post process for the source
    with index `i_src`, as performed in this project. """

    STD_KEYS = ['ElectricFieldStrength', 'title', 'K',
                'header', 'N1', 'N2']
    SRC_IDENTIFIER = 'ElectricFieldStrength'

    def __init__(self, jcm_dict, i_src=0):
        JCM_Post_Process.__init__(self, jcm_dict, i_src=i_src)

    def set_values(self):
        # Extract the info from the dict
        self.E_strength = self.jcm_dict['ElectricFieldStrength'][self.i_src]
        self.K = self.jcm_dict['K']
        self.header = self.jcm_dict['header']
        self.N1 = self.jcm_dict['N1']
        self.N2 = self.jcm_dict['N2']

    def __repr__(self):
        return self.title+'(i_src={})'.format(self.i_src)

    def _cos_factor(self, theta_rad):
        thetas = np.arccos(np.clip( np.abs(self.K[:,-1]) / norm(self.K[0]) ,-1.0,1.0))
        return np.cos(thetas)/np.cos(theta_rad)

    def get_reflection(self, theta_rad):
        """Returns the reflection. `theta_rad` is the incident angle in
        radians!"""
        cos_factor = self._cos_factor(theta_rad)
        rt = np.sum(np.square(np.abs(self.E_strength)), axis=1)
        return np.sum(rt*cos_factor)

    def get_transmission(self, theta_rad, n_subspace, n_superspace):
        """Returns the transmission, which depends on the subspace and
        superspace refractive index. `theta_rad` is the incident angle in
        radians!"""
        cos_factor = self._cos_factor(theta_rad)
        rt = np.sum(np.square(np.abs(self.E_strength)), axis=1)
        return np.sum(rt*cos_factor)*n_subspace/n_superspace


class PP_DensityIntegration(JCM_Post_Process):
    """Holds the results of a JCM-DensityIntegration post process for the source
    with index `i_src`, as performed in this project. """

    def __init__(self, jcm_dict, i_src=0,quantity="ElectricFieldStrength"):
        self.quantity = quantity
        STD_KEYS = [quantity, 'DomainId', 'title']
        SRC_IDENTIFIER = quantity
        JCM_Post_Process.__init__(self, jcm_dict, i_src=i_src)

    def set_values(self):
        # Extract the info from the dict
        self.Quantity = self.jcm_dict[self.quantity][self.i_src]
        self.title = self.jcm_dict['title']
        self.DomainId = self.jcm_dict['DomainId']

    def __repr__(self):
        return self.title+'(i_src={})'.format(self.i_src)


class PP_Absorption(PP_DensityIntegration):
    STD_KEYS = ['ElectromagneticFieldAbsorption', 'DomainId', 'title']
    SRC_IDENTIFIER = 'ElectromagneticFieldAbsorption'
    
    def __init__(self, jcm_dict, i_src=0):
        PP_DensityIntegration.__init__(self,jcm_dict,
                                       i_src=i_src,
                                       quantity='ElectromagneticFieldAbsorption')


class PP_ElectricFieldEnhancement(PP_DensityIntegration):
    STD_KEYS = ['ElectricFieldEnergy', 'DomainId', 'title']
    SRC_IDENTIFIER = 'ElectricFieldEnergy'
    
    def __init__(self, jcm_dict, i_src=0):
        PP_DensityIntegration.__init__(self,jcm_dict,
                                       i_src=i_src,
                                       quantity='ElectricFieldEnergy')
    
class PP_FluxIntegration(JCM_Post_Process):
    """Holds the results of a JCM-FluxIntegration post process for the source
    with index `i_src`, as performed in this project. """
    STD_KEYS = ['ElectromagneticFieldEnergyFlux', 'DomainIdFirst','DomainIdSecond', 'title']
    SRC_IDENTIFIER = 'ElectromagneticFieldEnergyFlux'

    def __init__(self, jcm_dict, i_src=0):
        JCM_Post_Process.__init__(self, jcm_dict, i_src=i_src)

    def set_values(self):
        # Extract the info from the dict
        self.Flux = self.jcm_dict['ElectromagneticFieldEnergyFlux'][self.i_src]
        self.title = self.jcm_dict['title']
        #self.DomainId = np.zeros([len(self.jcm_dict['DomainIdFirst']),2],dtype=int)
        self.DomainIdFirst = self.jcm_dict['DomainIdFirst']
        self.DomainIdSecond = self.jcm_dict['DomainIdSecond']

    def __repr__(self):
        return self.title+'(i_src={})'.format(self.i_src)


class PP_GridStatistics3D(JCM_Post_Process):
    """Holds the results of a JCM-GridStatistics post process"""
    STD_KEYS = [        
        "DomainId","Volume",
        "SmallestElementVolume","LargestElementVolume",
        "AverageElementVolume","MedianElementVolume","SmallestEdgeLength",
        "LargestEdgeLength","AverageEdgeLength","MedianEdgeLength",
        "SmallestInnerAngle","LargestInnerAngle","AverageInnerAngle",
        "MedianInnerAngle","SmallestEdgeRatio","LargestEdgeRatio",
        "AverageEdgeRatio","MedianEdgeRatio","SmallestQ1",
        "LargestQ1","AverageQ1","MedianQ1",
        "SmallestQ2","LargestQ2","AverageQ2",
        "MedianQ2","SmallestQ3","LargestQ3",
        "AverageQ3","MedianQ3","SmallestQ5",
        "LargestQ5","AverageQ5","MedianQ5",
        "SmallestAreaRatio","LargestAreaRatio","AverageAreaRatio",
        "MedianAreaRatio","SmallestDihedralAngle","LargestDihedralAngle",
        "AverageDihedralAngle","MedianDihedralAngle"]
    SRC_IDENTIFIER = None
    
    def set_values(self):
        self.Volume = self.jcm_dict['Volume']
        self.title = self.jcm_dict['title']
        self.DomainId = self.jcm_dict['DomainId']

class PP_GridStatistics2D(JCM_Post_Process):
    """Holds the results of a JCM-GridStatistics post process"""
    STD_KEYS = [        
        "DomainId","Area",
        "SmallestElementArea","LargestElementArea",
        "AverageElementArea","MedianElementArea","SmallestEdgeLength",
        "LargestEdgeLength","AverageEdgeLength","MedianEdgeLength",
        "SmallestInnerAngle","LargestInnerAngle","AverageInnerAngle",
        "MedianInnerAngle","SmallestEdgeRatio","LargestEdgeRatio",
        "AverageEdgeRatio","MedianEdgeRatio","SmallestQ1",
        "LargestQ1","AverageQ1","MedianQ1",
        "SmallestQ2","LargestQ2","AverageQ2",
        "MedianQ2","SmallestQ3","LargestQ3",
        "AverageQ3","MedianQ3","SmallestQ5",
        "LargestQ5","AverageQ5","MedianQ5"]
    SRC_IDENTIFIER = None
    
    def set_values(self):
        self.Volume = self.jcm_dict['Area']
        self.title = self.jcm_dict['title']
        self.DomainId = self.jcm_dict['DomainId']

        
    
class PP_VolumeIntegration(JCM_Post_Process):
    """Holds the results of a JCM-DensityIntegration which is used to
    calculate the volumes of the different domain IDs."""

    STD_KEYS = ['VolumeIntegral', 'DomainId', 'title']
    SRC_IDENTIFIER = None
    
    def set_values(self):
        # Extract the info from the dict
        self.VolumeIntegral = self.jcm_dict['VolumeIntegral'][0].real
        self.title = self.jcm_dict['title']
        self.DomainId = self.jcm_dict['DomainId']

    def __repr__(self):
        return self.title+'(domain_ids={})'.format(self.DomainId)
    '''
    def __getitem__(self, index):
        return self.DomainId[index], self.VolumeIntegral[index]
    '''

class PP_ExportField(JCM_Post_Process):

    def __init__(self, jcm_dict,Quantity, i_src=0):
        jcm_dict['title'] = Quantity
        JCM_Post_Process.__init__(self, jcm_dict, i_src=i_src)
        self.STD_KEYS = ['field','DomainId','title']
        self.SRC_IDENTIFIER = 'field'
        self.Quantity = Quantity
    
    def set_values(self):
        # Extract the info from the dict
        self.field = self.jcm_dict['field'][self.i_src]
        self.grid = self.jcm_dict['grid']
        self.X = self.jcm_dict['X']        
        self.Y = self.jcm_dict['Y']
        self.Z = self.jcm_dict['Z']

    def __repr__(self):
        return self.title+'(domain_ids={})'.format(self.DomainId)

    def writeToFile(self,keys):
        folder = keys['field_storage_folder']
        subfolder = "pol{}_phi{}_theta{}".format(keys['polarization'],
                                                       keys['phi'],
                                                       keys['theta'])
        fullfolder = os.path.join(folder,subfolder)
        if not os.path.isdir(fullfolder):
            os.makedirs(fullfolder)        

        filename = "{}_{}.txt".format(self.Quantity,'real_x')
        fullfile = os.path.join(fullfolder,filename)
        np.savetxt(fullfile,np.squeeze(np.real(self.field[:,:,0])))

        filename = "{}_{}.txt".format(self.Quantity,'real_y')
        fullfile = os.path.join(fullfolder,filename)
        np.savetxt(fullfile,np.squeeze(np.real(self.field[:,:,1])))

        filename = "{}_{}.txt".format(self.Quantity,'real_z')
        fullfile = os.path.join(fullfolder,filename)
        np.savetxt(fullfile,np.squeeze(np.real(self.field[:,:,2])))
        
        filename = "{}_{}.txt".format(self.Quantity,'imag_x')
        fullfile = os.path.join(fullfolder,filename)
        np.savetxt(fullfile,np.squeeze(np.imag(self.field[:,:,0])))

        filename = "{}_{}.txt".format(self.Quantity,'imag_y')
        fullfile = os.path.join(fullfolder,filename)
        np.savetxt(fullfile,np.squeeze(np.imag(self.field[:,:,1])))

        filename = "{}_{}.txt".format(self.Quantity,'imag_z')
        fullfile = os.path.join(fullfolder,filename)
        np.savetxt(fullfile,np.squeeze(np.imag(self.field[:,:,1])))        

        filename = "{}_{}.txt".format(self.Quantity,'X')
        fullfile = os.path.join(fullfolder,filename)
        np.savetxt(fullfile,np.squeeze(self.X))

        filename = "{}_{}.txt".format(self.Quantity,'Y')
        fullfile = os.path.join(fullfolder,filename)
        np.savetxt(fullfile,np.squeeze(self.Y))

        filename = "{}_{}.txt".format(self.Quantity,'Z')
        fullfile = os.path.join(fullfolder,filename)
        np.savetxt(fullfile,np.squeeze(self.Z))
        
def iterate_sources_for_pp(pp, class_):
    """Returns a list of `class_`-instances from post process data
    of JCMsuite for each source."""
    if class_.SRC_IDENTIFIER is None:
        return class_(pp)
    else:
        n_sources = pp[class_.SRC_IDENTIFIER].keys()
        return [class_(pp, i) for i in n_sources]

def plane_wave_energy_in_volume(volume, refractive_index):
    """Returns the energy normalization factor from the case of a plane wave.
    """
    return refractive_index**2*constants.epsilon_0*volume/4.0
    #Z0 = np.sqrt( constants.mu_0 / constants.epsilon_0 )
    #print(volume,refractive_index,Z0)
    #poynting_constant = 0.5*refractive_index/(Z0)
    #return poynting_constant*volume

def plane_wave_flux_in_area(area, refractive_index):
    """Returns the energy normalization factor from the case of a plane wave.
    """
    Z0 = np.sqrt( constants.mu_0 / constants.epsilon_0 )
    poynting_constant = 0.5*refractive_index/(Z0)
    angle_factor = 1.0
    return poynting_constant*area*angle_factor

def getDomainArea(keys):
    if keys['Dimensionality'] == "2D":
       return keys['pitch']*keys['uol']
    elif keys['Dimensionality'] == "3D":
       if keys['domain_shape'] == 'Square':
           return (keys['pitch']*keys['uol'])**2
       elif keys['domain_shape'] == 'Parallelogram':
           return (keys['pitch']*keys['uol'])**2*np.sin(np.radians(60))
       elif keys['domain_shape'] == 'Hexagon':
           return 0.5*np.sqrt(3)*(keys['pitch']*keys['uol'])**2           
    return 0.0

def writeParameters(keys,results):
    wvl = keys['vacuum_wavelength']
    theta_in = np.deg2rad(keys['theta'])
    uol = keys['uol'] # unit of length for geometry data
    p = uol*keys['pitch']

def writeRefractiveIndices(keys,results,nk_data):
    # Save the refactive index data, real and imag parts marked
    # with '_n' and '_k'
    for domain in keys['Domains'].keys():
        nname = 'nk_{0}'.format(domain)
        #results[nname] =  nk_data[domain]
        results[nname+'_n'] = np.real( nk_data[domain])
        results[nname+'_k'] = np.imag( nk_data[domain])

def grabPP(pps,PP_Template,pp_dict,names):
    # For a list of post processes dictionaries, cycle through
    # to find post processes that can be formatted into the
    # given PP_Template and add the result to pp_dict with
    # key given by the ith entry in names
    for i in range(len(pps)):
        if type(pps[i]) == list:
            pps[i] = pps[i][0]
        try :
            pp = iterate_sources_for_pp(pps[i], PP_Template)
            if names[0] not in pp_dict.keys():
                pp_dict[names[0]] = pp
            else:
                pp_dict[names[1]] = pp
        except (KeyError, ValueError) as e:
            pass

    assert names[0] in pp_dict

        
def RTfromFT(pps,keys,results,nk_data):
    # Assume first PP found is for reflection
    RT = {}
    if keys['incidence'] == 'FromAbove':
        n_in = np.real(nk_data['superspace'])
        n_out = np.real(nk_data['subspace'])
        pp_order = ['R','T']
    elif keys['incidence'] == 'FromBelow':
        n_in = np.real(nk_data['subspace'])
        n_out = np.real(nk_data['superspace'])
        pp_order = ['T','R']
    grabPP(pps,PP_FourierTransform,RT,pp_order)
    # We should have found multiple sources1
    num_srcs = len(RT['R'])
    sources =list(range(num_srcs))
    theta_in = keys['theta']
    for i in sources:
        # Reflection and transmission is calculated by the get_refl_trans
        # of the PP_FourierTransform class
        refl = RT['R'][i].get_reflection(theta_in)
        trans = RT['T'][i].get_transmission(theta_in,
                                            n_subspace=n_out,
                                            n_superspace=n_in)
        #index = np.where(fdis[i].DomainIdSecond==keys['Domains']['subspace'])
        #trans = np.real(fdis[i].Flux[index])[0]/p_in
        # Save the results
        results['R_{0}'.format(i+1)] = refl
        results['T_{0}'.format(i+1)] = trans
    
def RTfromFlux(pps,keys,results,nk_data):
    RT = {}
    grabPP(pps,PP_FluxIntegration,RT,['Flux'])
    num_srcs = len(RT['Flux'])
    sources =list(range(num_srcs))

    area_in = getDomainArea(keys)
    if keys['incidence'] == 'FromAbove':
        domain_name_in = 'superspace'
        domain_name_out = 'subspace'
    elif keys['incidence'] == 'FromBelow':
        domain_name_in = 'subspace'
        domain_name_out = 'superspace'
    p_in = plane_wave_flux_in_area(area_in, np.real(nk_data[domain_name_in]))
    for i in sources:
        pp = RT['Flux'][i]
        index_up = np.where( pp.DomainIdSecond == keys['Domains'][domain_name_in])
        index_down = np.where( pp.DomainIdSecond == keys['Domains'][domain_name_out])
        refl = np.real(pp.Flux[ index_up][0])/p_in
        trans = np.real(pp.Flux[ index_down][0])/p_in                                                  
        results['R_Flux_{0}'.format(i+1)] = refl
        results['T_Flux_{0}'.format(i+1)] = trans

def absorption(pps,keys,results,nk_data):
    Abs = {}
    grabPP(pps,PP_Absorption,Abs,['Quantity'])
    num_srcs = len(Abs['Quantity'])
    sources =list(range(num_srcs))
    area_in = getDomainArea(keys)
    if keys['incidence'] == 'FromAbove':
        domain_name_in = 'superspace'    
    elif keys['incidence'] == 'FromBelow':
        domain_name_in = 'subspace'    
    p_in = plane_wave_flux_in_area(area_in, np.real(nk_data[domain_name_in]))
    for i in sources:
        pp = Abs['Quantity'][i]
        for domain in keys['Domains']:
            index = np.where( pp.DomainId == keys['Domains'][domain])
            if len(index[0]) == 0:
                results['Abs_{0}_{1}'.format(domain,i+1)] = 0.0
                continue
            absorption = np.real(pp.Quantity[index][0])/p_in
            results['Abs_{0}_{1}'.format(domain,i+1)] = absorption

def domainVolumes(pps,keys,results,nk_data):
    Volumes = {}
    if keys['useGridStatistics']:
        if keys['Dimensionality'] == "2D":
            grabPP(pps,PP_GridStatistics2D,Volumes,['Volume'])
        else:
            grabPP(pps,PP_GridStatistics3D,Volumes,['Volume'])        
    else:
        grabPP(pps,PP_VolumeIntegration,Volumes,['Volume'])

    pp = Volumes['Volume']

    if keys['useGridStatistics']:
        volumes = pp.Volume
    else:
        volumes = pp.VolumeIntegral

    for domain in keys['Domains']:
        index = np.where( pp.DomainId == keys['Domains'][domain])
        if len(index[0]) == 0:
            continue
            
        if keys['Dimensionality'] == '2D':
            volume = volumes[index][0]
        else:
            volume = volumes[index][0]
        keys['Volume_{0}'.format(domain)] = volume

            
def eFieldEnhancement(pps,keys,results,nk_data):
    eFieldEnergy = {}
    grabPP(pps,PP_ElectricFieldEnhancement,eFieldEnergy,['Quantity'])
    num_srcs = len(eFieldEnergy['Quantity'])
    sources =list(range(num_srcs))
    p_in = 1.0    
    for i in sources:
        pp = eFieldEnergy['Quantity'][i]
        for domain in keys['Domains']:
            index = np.where( pp.DomainId == keys['Domains'][domain])
            if len(index[0]) == 0:
                continue
            vol = keys['Volume_{}'.format(domain)]
            p_in = plane_wave_energy_in_volume(vol,np.real(nk_data[domain]))
            efieldEnergy = np.real(pp.Quantity[index][0])/p_in
            results['EFieldEnergyEnhancement_{0}_{1}'.format(domain,i+1)] = efieldEnergy

def energyConservation(pps,keys,results,nk_data):
    i_src = 0
    conservation_keys = ['R','T']
    conservation_keysFlux = ['R_Flux','T_Flux']

    for domain in keys['Domains']:
        conservation_keys.append('Abs_{}'.format(domain))
        conservation_keysFlux.append('Abs_{}'.format(domain))        
    keysets = [conservation_keys,conservation_keysFlux]
    conservation_name = ["Conservation","ConservationFlux"]
    while True:
        conservation = [0.0,0.0]
        nMissing = [0,0]
        for keysetnum in range(2):
            keyset = keysets[keysetnum]
            #print(results.keys())
            for ik,key in enumerate(keyset):
                #print("{}_{}".format(key,i_src+1))
                try:
                    #if keysetnum == 1:
                    #    if ik == 0 or ik == 1:
                    #        key += "_Flux"
                    conservation[keysetnum] += results["{}_{}".format(key,i_src+1)]
                except KeyError:
                    #print("{}_{} is missing".format(key,i_src+1))
                    nMissing[keysetnum] += 1
                    pass
            if nMissing[keysetnum] == 0:                
                results["{}_{}".format(conservation_name[keysetnum],
                                       i_src+1)] = conservation[keysetnum]

        if np.all(np.array(nMissing) >= len(conservation_keys)):
            break
        i_src += 1

        
        

            
def processing_function(pps, keys):
    if keys['projectType'] == "scattering":
        return processing_function_scattering(pps,keys)
    elif keys['projectType'] == "smatrix":
        return processing_function_smatrix(pps,keys)
    else:
        raise KeyError("keys[projectType] {} not a valid key".format(keys['[projectType']))

def processing_function_scattering(pps, keys):    
    """returns the a dictionary with the results from jcm
       post processes relevant to scattering problems"""
    results = {}

    # Use key defaults for keys which are not provided
    default_keys = {'min_mesh_angle' : 20.,
                    'refine_all_circle' : 2,
                    'uol' : 1.e-9,
                    'pore_angle' : 0.,
                    'info_level' : 10,
                    'storage_format' : 'Binary',
                    'fem_degree_min' : 1,
                    'n_refinement_steps' : 0}
    for dkey in default_keys:
        if not dkey in keys:
            keys[dkey] = default_keys[dkey]
            
    # Refractive indices
    wvl = keys['vacuum_wavelength']
    nk_data = {}    
    for domain in keys['Domains'].keys():
        nk_data[domain] = keys['mat_'+domain].get_nk_data(wvl)


    if "writeParameters" in keys['PostProcesses']:
        #print("writeParameters")
        writeParameters(keys,results)
    
    if "writeRefractiveIndices" in keys['PostProcesses']:
        #print("writeRefractiveIndices")        
        writeRefractiveIndices(keys,results,nk_data)

    if "RTfromFT" in keys['PostProcesses']:
        #print("RTfromFT")
        RTfromFT(pps,keys,results,nk_data)
        pass

    if "RTfromFlux" in keys['PostProcesses']:
        #print("RTfromFT")
        RTfromFlux(pps,keys,results,nk_data)
        pass

    if "Absorption" in keys['PostProcesses']:
        #print("RTfromFT")
        absorption(pps,keys,results,nk_data)
        pass

    if "DomainVolumes" in keys['PostProcesses']:
        domainVolumes(pps,keys,results,nk_data)
        for key in keys.keys():
            if key[:7] == 'Volume_':
                results[key] = keys[key]
        for key in keys.keys():
            if key[:7] == 'Volume_':
                area_in = getDomainArea(keys)
                results['Quasi_Thickness_'+key[7:]] = np.round(1e9*keys[key]
                                                               /area_in,
                                                               decimals=1)
                
    
    if "ElectricFieldEnhancement" in keys['PostProcesses']:
        domainVolumes(pps,keys,results,nk_data)
        eFieldEnhancement(pps,keys,results,nk_data)

    if "EnergyConservation" in keys['PostProcesses']:
        energyConservation(pps,keys,results,nk_data)
        
    return results

if __name__ == '__main__':
    pass
