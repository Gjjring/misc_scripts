"""
Helper script for starting JCMsuite simulations with the pypmj python package
"""

import os, sys
import pypmj as jpy
from pypmj.internals import _config
import numpy as np
import pickle
import pdb
import GenKeys
import argparse

def getParser():
    parser = argparse.ArgumentParser(description='Run simulations for dataset and keys  combination')
    parser.add_argument('DataSetName',metavar='DataSet',type=str,help ='Name of the current dataset')
    parser.add_argument('--test',action="store_true",help='test the first simulation in the simuset and print the output')
    parser.add_argument('--geo_only',action="store_true",help='test only the meshing')
    parser.add_argument('--convergence',default=None,help='run a convergence test')
    parser.add_argument('--N',default=8,help='number of threads per process')
    parser.add_argument('--M',default=8,help='multiplicity of threads')
    parser.add_argument('--resource',default='htc027',help='default resource to use')
    parser.add_argument('--project',default='fromLocalFile',help='name of the pypmj project, defulat takes value from a local file called projectname.txt')
    parser.add_argument('--NDemonJobs',default=32,help='number of jobs handed to the daemon')
    parser.add_argument('--WDirMode',default='delete',help='whether to [keep] or [delete] the simulation files')
    parser.add_argument('--Logs',default='delete',help='whether to [keep] or [delete] the log files of simulations')
    parser.add_argument('--ProcessCCosts',default=True,help='whether to hand the computational costs computed via JCMwave to the processing_function')
    parser.add_argument('--ParameterCombination',default='product',help='whether to create simulation set using product of all array variables [product] or each array variable represents a value in a global list [list]')
    parser.add_argument('--SingleSimulation',default=-1,help="choose to run a single simulation from the set by giving the simulation number. Default value is -1, meaning all simulations will be run")
    return parser

def RemoveSingletons(keys):
    subdicts = ['parameters','geometry']
    for subdict in subdicts:
        for item in keys[subdict].items():
            if type(item[1]) is np.ndarray:
                if item[1].size == 1:
                    keys[subdict][item[0]] = item[1][0]
    return keys


def StartScript(args):
    if not args.NDemonJobs == 'all':
        args.NDemonJobs = int(args.NDemonJobs)
    if args.ProcessCCosts == "True":
        args.ProcessCCosts = True
    elif args.ProcessCCosts == 'False':
        args.ProcessCCosts = False
    assert( args.ParameterCombination == 'product' or args.ParameterCombination == 'list')
    print(args)
    #print("args.ProcessCCosts:{}".format(args.ProcessCCosts))
    N = int(args.N)
    M = int(args.M)
    if args.geo_only:
        assert args.test==True ,"option geo_only requires option test"
    if args.project == 'fromLocalFile':
        with open('projectname.txt','r') as f:
            projectDir = f.readline().rstrip('\n')
    else:
        #projectDir = os.path.join("/home/numerik/bzfmanle/Simulations/pypmj/projects",args.project)
        projectDir = args.project
    #projectDir = 'scattering/nanopillar_array_blochfamilies'
    #projectDir = 'scattering/photonic_crystals/slabs/hexagonal/half_spaces_BlochFamilies/'
    project = jpy.JCMProject(projectDir)

    keys = GenKeys.getKeys(args.DataSetName)
    keys['constants']['dataset_name'] = os.path.splitext(args.DataSetName)[0]
    #keys['constants']['spectra_storage_folder'] += args.DataSetName
    #keys['constants']['field_storage_folder'] += args.DataSetName
    if 'jones_matrix_storage_folder' in keys['constants']:
        keys['constants']['jones_matrix_storage_folder'] += args.DataSetName
        if not os.path.isdir(keys['constants']['jones_matrix_storage_folder']):
            os.makedirs(keys['constants']['jones_matrix_storage_folder'])

    if 'pml_log_folder' in keys['constants']:
        keys['constants']['pml_log_folder'] += args.DataSetName
        if not os.path.isdir(keys['constants']['pml_log_folder']):
            os.makedirs(keys['constants']['pml_log_folder'])

            
    #if not os.path.isdir(keys['constants']['spectra_storage_folder']):
    #    os.makedirs(keys['constants']['spectra_storage_folder'])

    if args.ParameterCombination == 'list':
        keys = RemoveSingletons(keys)

    simple_keys = jpy.utils.convert_keys(keys)
    #if simple_keys['vacuum_wavelength'] > 1e-5:
    #    raise ValueError('vacuum wavelength larger than 10000nm, is this correct?')

    from JCMProject_Utils import processing_function
    if args.test:
        import jcmwave as jcm
        fullprojectDir = _config.get('Data','projects') +'/'+ projectDir+'/'
        simple_keys['sim_number'] = 0
        if args.geo_only:
            jcm.jcmt2jcm(fullprojectDir+'layout.jcmt',simple_keys)
            jcm.geo(fullprojectDir,simple_keys)
            jcm.view(fullprojectDir+'/grid.jcm')
        else:
            simple_keys['fem_degree_max'] = simple_keys['fem_degree_min']
            results = jcm.solve(fullprojectDir+'/project.jcmpt',simple_keys,mode='solve')
            #print(results)
            if args.ProcessCCosts:
                df = processing_function(results[0:],simple_keys)
            else:
                df = processing_function(results[1:],simple_keys)
            for item in df.items():
                print("{0}:{1}".format(item[0],item[1]))

    elif args.convergence is not None:
        import copy
        keys_test = copy.deepcopy(keys)
        try:
            keys_test['parameters'][args.convergence] = keys_test['parameters'][args.convergence][0:-1]
        except KeyError:
            keys_test['geometry'][args.convergence] = keys_test['geometry'][args.convergence][0:-1]
        keys_ref = copy.deepcopy(keys)
        try:
            keys_ref['parameters'][args.convergence] = keys_ref['parameters'][args.convergence][-1:]
        except KeyError:
            keys_ref['geometry'][args.convergence] = keys_ref['geometry'][args.convergence][-1:]
        ctest = jpy.ConvergenceTest(project,keys_test,keys_ref,duplicate_path_levels=0,storage_folder='FBH_LEDs/'+args.DataSetName)
        ctest.use_only_resources(args.resource)
        ctest.resource_manager.resources.set_m_n_for_all(1,32)
        ctest.make_simulation_schedule()
        ctest.run(N=1,processing_func=processing_function,pass_ccosts_to_processing_func=True)
        test = ['R_1','T_1','I_0','I_1','I_2','I_3','I_4','I_5','I_6','I_7','I_8','I_9']
        ctest.analyze_convergence_results(test,data_ref=df)
        ctest.write_analyzed_data_to_file(file_path='results/'+args.DataSetName+'_Analysis')
    else:
        if args.Logs == 'keep':
            keepLogs = True
        elif args.Logs == 'delete':
            keepLogs =False

        if "storage_base" in keys['constants']:
            storage_base = keys['constants']['storage_base']
        else:
            storage_base = "from_config"
        print("args.resource: {}".format(args.resource))
        if ("transitional_storage_base" in keys['constants'] and
            args.resource.startswith("z1")):
            trans_storage_base = keys['constants']['transitional_storage_base']
            print("setting trans storage base")
        else:
            trans_storage_base = None

            
            
        simuset = jpy.SimulationSet(project, keys,
                                    storage_folder=os.path.join(keys['constants']['storage_folder'],args.DataSetName),
                                    transitional_storage_base=trans_storage_base,
                                    storage_base=storage_base,
                                    combination_mode=args.ParameterCombination,
                                    store_logs=keepLogs,
                                    check_version_match=False)
        #combination_mode='list'
        simuset.make_simulation_schedule()
        resources = args.resource.split(",")
        print("resources: {}".format(resources))
        simuset.use_only_resources(resources)
        if N >0:
            simuset.resource_manager.resources.set_m_n_for_all(M,N)
        import time
        start = time.time()
        if args.SingleSimulation == -1:
            simuset.run(N=args.NDemonJobs,
                    processing_func = processing_function,
                    pass_ccosts_to_processing_func=args.ProcessCCosts,
                    wdir_mode=args.WDirMode,
                    auto_rerun_failed=1)
            stop =  time.time()
            print("Time for all simulations: {}".format(stop-start))
        else:
            simuset.solve_single_simulation(int(args.SingleSimulation),
                                            processing_func=processing_function,
                                            wdir_mode=args.WDirMode)
        if simuset.get_store_data() is not None:
            simuset.write_store_data_to_file(file_path='results/'+args.DataSetName+'.csv')
            #simuset.write_store_data_to_file(mode='Excel',file_path='results/'+args.DataSetName+'.xls')
        simuset.close_store()


if __name__ == "__main__":

    args = getParser().parse_args()
    StartScript(args)
