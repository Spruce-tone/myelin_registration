import gzip, os, re
import xml.etree.ElementTree as ET
from collections import Counter
from pkg.utils import CustomLogger
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
        logger.info(f'{path} already exist')
    
def fname_extension_check(path: str, fname_extension: str) -> List:
    '''
    Input Args:
        path: str
            directory path containing files to check file name extension
        fname_extension: str
            filename extension for filtering
    '''

    logger.debug(f'raw data path : {path}')
    logger.debug(f'raw data file list : {os.listdir(path)}')
    files = [file for file in os.listdir(path) if file.endswith(fname_extension)]
    logger.debug(f'fname extensiton filtered file list : {files}')
    return files

def define_name_parser():
    '''
    Return:
        name_parser: regular expression format
            compile regular expression format
    '''
    name_parser = re.compile('(?P<fname>#(?P<unique_idx>(?P<id>\w+) d(?P<day>\d+))[ ]?(?P<description>.*)[.](?P<fname_extension>.+$))')
    return name_parser
    
def unique_counter(fnames: list, save_name: bool=True) -> Dict:
    '''
    Input Args:
        fnames: list
            list of file names for unique id check
        save_name: bool (default = True)
            if True, save the file name 
   
    Return:
        unique_names
            count unique image id and days
    '''
    logger.info(f'start unique id count')
    name_parser = define_name_parser()
    unique_names = {}

    logger.debug(f'file list {fnames}')
    for fname in fnames:
        logger.debug(f'Check {fname}')

        if name_parser.search(fname) != None:
            logger.debug(f'Check {fname}')
            parsed_name = name_parser.search(fname)
            
            img_id = parsed_name.group('id')

            if img_id not in unique_names.keys():
                unique_names[img_id] = {'day' : [], 'count' : 0, 'fname' : []}

            if parsed_name.group('day') not in unique_names[img_id]['day']:
                unique_names[img_id]['day'].append(int(parsed_name.group('day')))
                unique_names[img_id]['count'] += 1
                if save_name:
                    unique_names[img_id]['fname'].append(parsed_name.group('fname'))
                
            else:
                duplicate_id = parsed_name.group('unique_idx')
                logger.debug(f'Duplicate day {duplicate_id}')
                raise Exception(f'Duplicate day {duplicate_id}')
    return unique_names


def search_regi_files(raw_path: str='./data', regi_path: str='./registration') -> List[Tuple]:
    '''
    Input Args:
        raw_path: str (default = './data')
            directory path containing raw .traces and image files
        regi_path: str (default = './registration')
            directory path containing registrated .traces and image files

    Return:
        file_path_regi: List[Tuple]
            path for files to be preformed registration.
            example) [(raw_file_1.tif, raw_file_1.traces), (raw_file_2.tif, raw_file_2.traces), (raw_file_3.tif, raw_file_3.traces)...]
    '''
    dir_exist_check(raw_path) # check raw data directory existence
    dir_exist_check(regi_path) # check registration directory existence

    logger.info('Directory existence check complete')
    
    raw_list = fname_extension_check(raw_path, '.traces')
    raw_list = unique_counter(raw_list, save_name=True)
    regi_list = fname_extension_check(regi_path, '.traces')
    regi_list = unique_counter(regi_list, save_name=False)

    # extract file list that is not yet preformed registration
    logger.debug('start extract file list for registration')
    file_path_regi = {}
    for raw_id in raw_list.keys(): 
        if raw_list[raw_id]['count'] < 2:
            logger.debug(f'More than 2 files are needed for registration. {raw_id} is excluded')
            continue

        if raw_id in regi_list.keys():
            if set(raw_list[raw_id]['day']) == set(regi_list[raw_id]['day']):
                logger.debug(f'Already registration is preformed. {raw_id} is excluded')
                continue
            
        file_path_regi[raw_id] = {}
        sorted_fnames = [fname.strip('.traces') for fname in sorted(raw_list[raw_id]['fname'], key=day_sorting)]
        file_path_regi[raw_id]['ref'] = [(os.path.join(raw_path, fname+'.tif'), os.path.join(raw_path, fname+'.traces')) for fname in [sorted_fnames[0]]]
        file_path_regi[raw_id]['registration'] = [(os.path.join(raw_path, fname+'.tif'), os.path.join(raw_path, fname+'.traces')) for fname in sorted_fnames[1:]]

        logger.debug(f'{raw_list[raw_id]["fname"]} is added to the registration list')

    return file_path_regi

def day_sorting(fname: str):
    '''
    Input Args:
        fname: str 
            filne for sorting
    '''
    parser = define_name_parser()
    parsed_name = int(parser.search(fname).group('day')) # sorting list in ascending order based on the day
    return parsed_name


def start_registration(file_path_regi: List[Tuple], regi_path: str='./registration'):
    '''
    Input Args:
        file_path_regi: List[Tuple]
            path for files to be preformed registration.
            example) [(raw_file_1.tif, raw_file_1.traces), (raw_file_2.tif, raw_file_2.traces), (raw_file_3.tif, raw_file_3.traces)...]
        regi_path: str (default = './registration')
            directory path containing registrated .traces and image files
    '''

    pass

if __name__=='__main__':
    file_path_regi = search_regi_files(raw_path='./data', regi_path='./registration')
