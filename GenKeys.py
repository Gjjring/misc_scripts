import numpy as np
import argparse
import importlib
import json
import shutil
import pdb
import os
def getKeys(Name):
    keyset = 0
    allkeysread = False
    nkeys_files = -1
    for file_ in os.listdir('keys'):
        (fileName,ext) = os.path.splitext(file_)

        if ext == '.py' and len(fileName.split('_',1))>1:

            nkeys_files += 1

            if fileName.split('_',1)[1] == Name:
                try:
                    keysModule = importlib.import_module('keys.{0}'.format(fileName))
                    print("importing keys.{0}".format(fileName))
                    return keysModule.getKeys();
                except:
                    raise IOError('Could not open keys file')
    print('[WARNING] Could not find keys with name '+Name+', generating new keys')
    try:
        keysModule = importlib.import_module('keys.keys')
        shutil.copyfile('keys/keys.py','keys/{:03d}_{}.py'.format(nkeys_files+1,Name))        
        return keysModule.getKeys();
    except:
        raise IOError('Could not generate new keys file')





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate keys object and save to disk')
    parser.add_argument('Name',metavar='Name',type=str,nargs='+',help ='Name of keys file in keys directory which should be saved as an object')
    args = parser.parse_args()

    """

    dir = os.path.dirname(Name)
    if dir not in sys.path:
        sys.path.
    """

    try:
        keysModule = importlib.import_module('keys.{}'.format(args.Name[0]))
        keys = keysModule.getKeys();
    except:
        print('Could not import keys from file {}'.format(args.Name))
