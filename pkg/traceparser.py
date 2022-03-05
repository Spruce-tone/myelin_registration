import gzip, os, re, pickle
from inspect import trace
import xml.etree.ElementTree as ET
from collections import Counter
from typing import List, Set, Dict, Tuple
from matplotlib import docstring
from skimage import io
import numpy as np

from pkg.utils import CustomLogger

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
    
    paths = {}
    
    for child in root:
        if child.tag == 'path':
            path = {'x' : [], 'y' : [], 'z' : []}
            for point in child:
                path['x'].append(int(point.attrib['x'])) # pixel index
                path['y'].append(int(point.attrib['y']))
                path['z'].append(int(point.attrib['z']))
            paths[child.attrib['name']] = path 

    # paths['fname'] = tracefile
    return paths

def dir_exist_check(path: str, get_path: bool=False):
    '''
    Input Args:
        path: str
            directory path to check. if no path, make directory in the path
        get_path: bool (default=False)
            if true, return the path
    '''
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f'Make directory : {path}')
    else:
        logger.info(f'{path} already exist')
    
    if get_path:
        return path
    
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

def day_sorting(fname: str):
    '''
    Input Args:
        fname: str 
            filne for sorting
    '''
    parser = define_name_parser()
    parsed_name = int(parser.search(fname).group('day')) # sorting list in ascending order based on the day
    return parsed_name

def search_regi_files(raw_path: str='./data', regi_path: str='./registration') -> Dict:
    '''
    Input Args:
        raw_path: str (default = './data')
            directory path containing raw .traces and image files
        regi_path: str (default = './registration')
            directory path containing registrated .traces and image files

    Return:
        file_path_regi: Dict
            path for files to be preformed registration.
            The first image is the reference
            example) {'#raw_file_id' :[(raw_file_1.tif, raw_file_1.traces), (raw_file_2.tif, raw_file_2.traces), ...]}
    '''
    dir_exist_check(raw_path) # check raw data directory existence
    dir_exist_check(regi_path) # check registration directory existence

    logger.info('Directory existence check complete')
    
    raw_list = fname_extension_check(raw_path, '.traces')
    raw_list = unique_counter(raw_list, save_name=True)
    regi_list = fname_extension_check(regi_path, '.traces')
    regi_list = unique_counter(regi_list, save_name=False)

    logger.debug(f'Raw uniqe count : {raw_list}')
    logger.debug(f'Regi uniqe count : {regi_list}')

    # extract file list that is not yet preformed registration
    logger.debug('start extract file list for registration')
    file_path_regi = {}
    for raw_id in raw_list.keys(): 
        if raw_list[raw_id]['count'] < 2:
            logger.debug(f'More than 2 files are needed for registration. {raw_id} is excluded')
            continue

        if raw_id in regi_list.keys(): ##################################### crop metadata로 확인하는것으로 바꾸기
            if set(raw_list[raw_id]['day']) == set(regi_list[raw_id]['day']):
                logger.debug(f'Already registration is preformed. {raw_id} is excluded')
                continue

        file_path_regi[raw_id] = []
        sorted_fnames = [fname.strip('.traces') for fname in sorted(raw_list[raw_id]['fname'], key=day_sorting)]
        file_path_regi[raw_id] = [(fname+'.tif', fname+'.traces') for fname in sorted_fnames]

        logger.debug(f'{raw_list[raw_id]["fname"]} is added to the registration list')

    return file_path_regi

def path_to_array(path: Dict) -> np.ndarray:
    '''
    Input Args:
        path: Dict
            x, y, z 3D coordinates of single segment
            example) {'x' : [1, 2, 3], 'y' : [3, 2, 1], 'z' : [4, 5, 6]}

    Return:
        path_array: np.ndarray [shape : (3, length of x (or y, z))]
            numpy array of x, y, z 3D coordinates
    '''
    path_array = np.array([path['x'], path['y'], path['z']])
    return path_array

def edge_indexing(size_single_axis: int, boundary_pixel_idx: Tuple[int], margin_pixel: int=10) -> Tuple[int]:
    '''
    Input Args:
        size_single_axis: int
            size of specified axis
        boundary_pixel_idx: Tuple[int]
            boundary pixel index of specified axis
            (lower bound idx, upper bound idx)
        margin_pixel: int (default=10)
            the No. of pixel for margin
    Return:
        (lowerbnd_idx, uppperbnd_idx): Tuple[int]
            lower and upper bound index of image for cropping
    '''   
    lowerbnd_idx, uppperbnd_idx = boundary_pixel_idx
    # lower bound
    # negative index
    if lowerbnd_idx - margin_pixel < 0: 
        lowerbnd_idx = 0 # lower bound is zero
    # positive index including zero
    else: 
        lowerbnd_idx = lowerbnd_idx - margin_pixel # make margin at lower bound

    # upper bound
    # (marging + upper bound idx) is larger than img edge index
    if uppperbnd_idx + margin_pixel > size_single_axis - 1: # size_single_axis - 1 is edge index of image 
        uppperbnd_idx = size_single_axis - 1 # upper bound idx is edge index of image
    else:
        uppperbnd_idx = uppperbnd_idx + margin_pixel
    return (lowerbnd_idx, uppperbnd_idx)


def crop_img(img: np.ndarray, coords_idx: Tuple[int], margin_pixel: int=10) -> Tuple[np.ndarray, Tuple[int]]:
    '''
    Input Args:
        img: np.ndarray [z, y, x]
            image to be cropped
        coords_idx: Tuple[int]
            boundary index of myelin segment
            (xmin, xmax, ymin, ymax, zmin, zmax)
        margin_pixel: int (default=10)
            the No. of pixel for margin

    Return:
        cropped_img: np.ndarry
            cropped image with several marging
    '''
    lenz, leny, lenx,  = img.shape
    xmin_idx, xmax_idx, ymin_idx, ymax_idx, zmin_idx, zmax_idx = coords_idx
    logger.debug(f'segment boundary idx (xmin_idx, xmax_idx, ymin_idx, ymax_idx, zmin_idx, zmax_idx) = {coords_idx}')

    xmin_idx, xmax_idx = edge_indexing(lenx, (xmin_idx, xmax_idx), margin_pixel) # re indexing with marging
    ymin_idx, ymax_idx = edge_indexing(leny, (ymin_idx, ymax_idx), margin_pixel) # re indexing with marging
    zmin_idx, zmax_idx = edge_indexing(lenz, (zmin_idx, zmax_idx), margin_pixel) # re indexing with marging
    logger.debug(f'segment boundary re-indexing with margin (xmin_idx, xmax_idx, ymin_idx, ymax_idx, zmin_idx, zmax_idx) = {(xmin_idx, xmax_idx, ymin_idx, ymax_idx, zmin_idx, zmax_idx)}')

    cropped_img = img[zmin_idx:zmax_idx+1, ymin_idx:ymax_idx+1, xmin_idx:xmax_idx+1]
    logger.debug(f'cropped image shape : {cropped_img.shape}')
    return cropped_img, (xmin_idx, xmax_idx, ymin_idx, ymax_idx, zmin_idx, zmax_idx)

def save_pickle(data, path: str, fname: str):
    '''
    Save data as pickle

    Parameters
    ----------
    data : 
        data to save
    path : str
        save path
    fname : str
        file name
    '''    
    with open(os.path.join(path, fname), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(path: str, fname: str):
    '''
    load pickle data from file

    Parameters
    ----------
    path : str
        load path
    fname : str
        file name

    Returns
    -------
    _type_
        loaded pickle
    '''
    with open(os.path.join(path, fname), 'rb') as f:
        return pickle.load(f)

def segment_crop(file_path_regi: Dict[str, List[Tuple]], raw_path: str='./data', crop_path: str='crop_data', margin_pixel: int=10):
    '''
    Input Args:
        file_path_regi: Dict[str : List[Tuple]]
            path for files to be preformed registration.
            example) {'#raw_file_id' :[(raw_file_1.tif, raw_file_1.traces), (raw_file_2.tif, raw_file_2.traces), ...]}
        raw_path: str (default = './data')
            directory path containing raw .traces and image files
        crop_path: str (default = './crop_data')
            directory path containing cropped image and re-indexed segment data
        margin_pixel: int (default=10)
            the No. of pixel for margin
    '''
    logger.debug('start segment_registration')
    name_parser = define_name_parser()
    
    for img_id in file_path_regi.keys(): # iterate file id
        logger.debug(f'img id : {img_id}')

        crop_dir = dir_exist_check(os.path.join(crop_path, f'#{img_id}'), get_path=True) # make directory to save cropped image corresponds to file id
        seg_container = {} # container to save the segment for cropped image
        metadata = {'day' : [], 'segment_name' : []} # metadata for the list of segment and day

        for img_name, tracefile in file_path_regi[img_id]: # fnames (.img, .trace) for each day
            img = io.imread(os.path.join(raw_path, img_name)) # load image, [z, y, x]


            paths = extractTrace(raw_path, tracefile) # load trace file
            logger.debug(f'[ID : {img_id}] | img shape : {img.shape} | image and trace fname : {img_name}, {tracefile}')
            

            # parsing day
            seg_day = name_parser.search(img_name).group('day')
            seg_container[seg_day] = []

            # add day information 
            metadata['day'].append(seg_day)

            # segment_crop
            for idx, segment_name in enumerate(paths.keys()): # load each segment
                logger.debug(f'[ID : {img_id}][fname : {tracefile}]  segment_name : [idx : {idx}] {segment_name}')

                # container to save segment metadata {segment_name : segment[np.ndarray, shape=(3, length)]}
                seg_meta = {}

                seg = path_to_array(paths[segment_name])

                # boundary index of each segment
                rect_idx = (seg[0, :].min(), seg[0, :].max(), # xmin, xmax 
                            seg[1, :].min(), seg[1, :].max(), # ymin, ymax
                            seg[2, :].min(), seg[2, :].max()) # zmin, zmax

                # image crop and index for cropped image
                cropped_img, new_bnd_idx = crop_img(img, rect_idx, margin_pixel)

                # save cropped image
                io.imsave(os.path.join(crop_dir, f'd{seg_day} {segment_name}.tif'), cropped_img)
                
                # re-indexing the segment for cropped image 
                xmin_crop_idx, _, ymin_crop_idx, _, zmin_crop_idx, _ = new_bnd_idx # take min index
                seg[0, :] = seg[0, :] - xmin_crop_idx
                seg[1, :] = seg[1, :] - ymin_crop_idx
                seg[2, :] = seg[2, :] - zmin_crop_idx

                # save segment data for cropped image
                seg_meta[segment_name] = seg # segment name and path, ex) {'1-1' : path0}
                seg_container[seg_day].append(seg_meta) # save segment for each day {'0' : [{'1-1' : path0}, {'1-2' : path1},...], '1' : []}

                # metadata for segment list
                metadata['segment_name'].append(segment_name)

                # metadata for cropped image size
                if metadata.get(segment_name, None) == None:
                    metadata[segment_name] = [np.array(cropped_img.shape)]
                else:
                    metadata[segment_name].append(np.array(cropped_img.shape))

    # count the segment name
    segment_counter = Counter(metadata['segment_name'])
    metadata['segment_name'] = [segment for segment in segment_counter.keys() if segment_counter[segment] > 1] # save the segment name if count is larger than 1
    
    # save the common largest image size for registration
    for segment in metadata.keys():
        if segment not in ['day', 'segment_name']: # exclude day and segment name keyword
            metadata[segment] = compare_size(metadata[segment], ismax=True) # calculate the common largest image size

    save_pickle(metadata, crop_dir, f'#{img_id} meta.pickle') # save metadata for day and segment name
    save_pickle(seg_container, crop_dir, f'#{img_id}.pickle') # save segment data in pickle

def compare_size(shape_list: List[np.ndarray], ismax: bool=True) -> np.ndarray:
    '''
    compare the size of images and return the maximum value for each dimension

    Parameters
    ----------
    shape_list : List[np.ndarray]
        list contain the size of images
    ismax : bool, optional
        default = True
        if True, return maximum size of each axis
        if Flase, return minimum size of each axis

    Returns
    -------
    np.ndarray
        the largest image size for registration 
    '''
    shape_list = np.array(shape_list)
    if ismax: 
        return shape_list.max(axis=0)
    else:
        return shape_list.min(axis=0)

def pad_image(img: np.ndarray, regi_size: np.ndarray) -> np.ndarray:
    '''
    padding to match size of images for registration

    Parameters
    ----------
    img : np.ndarray
        image for padding
    regi_size : np.ndarray
        image size after padding

    Returns
    -------
    padded_image : np.ndarray
        padded image
    '''
    img_size = np.array(img.shape)

    half_size_diff = (regi_size - img_size)/2

    padding = np.array([[int(np.floor(i)), int(np.ceil(i))] for i in half_size_diff])
    logger.debug(f'padding shape : {padding}')
    padded_image = np.pad(img, padding, 'constant', constant_values=np.zeros(padding.shape))
    logger.debug(f'padding size compare | before {img.shape} | after {padded_image.shape}')
    return padded_image


def regi_cropped_img(crop_path: str='./crop_data'):
    for img_id in os.listdir(crop_path): # get cropped image list
        segment_meta = load_pickle(os.path.join(crop_path, img_id), f'{img_id} meta.pickle')
        
        for seg in segment_meta['segment_name']:
            common_img_size = segment_meta[seg]
            img_stack = []

            for day in segment_meta['day']:
                img = io.imread(os.path.join(crop_path, img_id, f'd{day} {seg}.tif'))
                logger.debug(f'[{seg} segment] day {day} | img shape : {img.shape} | common image size : {common_img_size}')

                # padding image
                padded_img = pad_image(img, common_img_size)
                logger.debug(f'padded image size : {padded_img.shape}')
                img_stack.append(padded_img)
                
if __name__=='__main__':
    file_path_regi = search_regi_files(raw_path='./data', regi_path='./registration')
