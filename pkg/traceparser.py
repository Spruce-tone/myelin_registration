import gzip, os, re
import xml.etree.ElementTree as ET
from collections import Counter
from utils import CustomLogger
from typing import List, Set, Dict, Tuple

logger = CustomLogger().info_logger

def extractTrace(tracedir: str, tracefile: str):
    '''
    Input Args:
        tracedir: str
            directory path containing .traces and image files
        tracefile: str
            name of .traces file
    
    Return:
        root: ET.Element (xml.etree.ElementTree.Element)
            .traces file data
    '''
    
    with gzip.open(os.path.join(tracedir, tracefile), 'r') as fileopen:
        tree = ET.parse(fileopen)
        root = tree.getroot()
    
    paths = {'trace' : []}
    path = {'x' : [], 'y' : [], 'z' : []}

    for child in root:
        if child.tag == 'path':
            for point in child:
                path['x'].append(point.attrib['xd'])
                path['y'].append(point.attrib['yd'])
                path['z'].append(point.attrib['zd'])
            path['trace_name'] = child.attrib['name']

        paths['trace'].append(path)
    paths['fname'] = tracefile

    return paths

def dir_exist_check(path: str):
    '''
    Input Args:
        path: str
            directory path to check. if no path, make directory in the path
    '''
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f'Make directory : {path}')
    else:
        logger.info(f'{path} exist')
    
def fname_extension_check(path: str, fname_extension: str) -> List:
    '''
    Input Args:
        path: str
            directory path containing files to check file name extension
        fname_extension: str
            filename extension for filtering
    '''
    files = [file for file in os.listdir(path) if file.endswith(fname_extension)]
    return files
     
def unique_counter(fnames: list, isregistrated: bool=False) -> Dict:
    '''
    Input Args:
        fnames: list
            list of file names for unique id check
        isregistrated: bool (default = False)
            whether the registration was applied or not.
            if True, the counting is performed at the registration directory
    
    Return:
        Counter(unique_names)
    '''
    logger.info(f'start unique id count')
    name_parser = re.compile('#(?P<unique_idx>(?P<id>\w+) d(?P<day>\d+))[ ]?(?P<description>.*)[.](?P<fname_extension>.+$)')
    
    unique_names = []
    for fname in fnames:
        if name_parser.search(fname) != None:
            logger.info(f'Check {fname}')
            
            if isregistrated:
                unique_names.append(name_parser.search(fname).group('unique_idx'))
            else:
                unique_names.append(name_parser.search(fname).group('id'))
    
    unique_name_counter = Counter(unique_names) 
    return unique_name_counter
    
def search_files(raw_path: str, regi_path: str):
    '''
    Input Args:
        raw_path: str
            directory path containing raw .traces and image files
        regi_path: str
            directory path containing registrated .traces and image files
    '''
    dir_exist_check(raw_path) # check directory existence
    logger.info('Directory existence check complete')
    
    raw_img = fname_extension_check(raw_path, '.tif')
    regi_img = fname_extension_check(regi_path, '.tif')

    unique_counter
    raw_imgs = []
    raw_traces = []
    