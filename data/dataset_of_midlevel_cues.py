from   collections import namedtuple, Counter, defaultdict
from   dataclasses import dataclass, field
from   joblib import Parallel, delayed
import logging
import multiprocessing as mp
import os
import pickle
import json
from   PIL import Image, ImageFile
import pandas as pd
import random
import re
from   time import perf_counter 
from   tqdm import tqdm
import torch
import torch.utils.data as data
import torch.nn.functional as F
from   torchvision import transforms
from   typing import Optional, List, Callable, Union, Dict, Any
import warnings

from .taskonomy_dataset import parse_filename, LabelFile, View
from .transforms import default_loader, get_transform, LocalContrastNormalization
from .task_configs import task_parameters, SINGLE_IMAGE_TASKS
from .segment_instance import HYPERSIM_LABEL_TRANSFORM, REPLICA_LABEL_TRANSFORM, COMBINED_CLASS_LABELS

ImageFile.LOAD_TRUNCATED_IMAGES = True # TODO Test this

MAX_VIEWS = 45

RGB_MEAN = torch.Tensor([0.55312, 0.52514, 0.49313]).reshape(3,1,1)
RGB_STD =  torch.Tensor([0.20555, 0.21775, 0.24044]).reshape(3,1,1)

REPLICA_BUILDINGS = [
    'frl_apartment_5', 'office_2', 'room_2', 'office_4', 'frl_apartment_0', 'frl_apartment_4',
    'office_1', 'frl_apartment_3', 'office_0', 'apartment_2', 'room_0', 'apartment_1', 
    'frl_apartment_1', 'office_3', 'frl_apartment_2', 'apartment_0', 'hotel_0', 'room_1']

N_OUTPUTS = {
    'segment_semantic': len(COMBINED_CLASS_LABELS)-1, 'depth_zbuffer':1, 
    'normal':3, 'edge_occlusion':1, 'edge_texture':1, 'keypoints3d':1, 'principal_curvature':3}

                    
class MidLevelCuesDataset(data.Dataset):
    '''
        This expects that the data is structured
        
            /path/to/data/
                rgb/
                    modelk/
                        point_i_view_j.png
                        ...                        
                depth_euclidean/
                ... (other tasks)
                
        If one would like to use pretrained representations, then they can be added into the directory as:
            /path/to/data/
                rgb_encoding/
                    modelk/
                        point_i_view_j.npy
                ...
        
        Basically, any other folder name will work as long as it is named the same way.
    '''
    @dataclass
    class Options():
        '''
            data_path: Path to data
            tasks: Which tasks to load. Any subfolder will work as long as data is named accordingly
            buildings: Which models to include. See `splits.taskonomy` (can also be a string, e.g. 'fullplus-val')
            transform: one transform per task.
            
            Note: This assumes that all images are present in all (used) subfolders
        '''

        tasks: List[str] = field(default_factory=lambda: ['rgb'])
        buildings: List[str] = field(default_factory=lambda: ['apartment_2'])
        data_path: str = 'dataset'
        transform: Optional[Union[Dict[str, Callable], str]] = "DEFAULT"  # List[Transform], None, "DEFAULT"
        image_size: Optional[int] = None
        num_positive: Union[int, str] = 1 # Either int or 'all'
        normalize_rgb: bool = False
        randomize_views: bool = True

    def load_datasets(self, options):
        # Load saved image locations if they exist, otherwise create and save them
        self.urls = defaultdict(list)
        self.size = 0

        dataset_urls = {
            task: make_dataset(
                self.data_path, task, self.buildings)
                for task in options.tasks}

        dataset_urls, dataset_size  = self._remove_unmatched_images(dataset_urls)

        for task, urls in dataset_urls.items():
            self.urls[task] += urls
        self.size += dataset_size

        
    def __init__(self, options: Options):
        start_time = perf_counter()
        
        if isinstance(options.tasks, str):
            options.tasks = [options.tasks]
            options.transform = {options.tasks: options.transform}        
        
        self.data_path = options.data_path        
        self.image_size = options.image_size
        self.tasks = options.tasks
        self.num_positive = MAX_VIEWS if options.num_positive == 'all' else options.num_positive
        self.normalize_rgb = options.normalize_rgb
        self.randomize_views = options.randomize_views

        self.buildings = options.buildings

        self.load_datasets(options)

        self.transform = options.transform
        if isinstance(self.transform, str):
            if self.transform == 'DEFAULT':
                self.transform = {task: get_transform(task, self.image_size) for task in self.tasks}
            else:
                raise ValueError('Dataset option transform must be a Dict[str, Callable], None, or "DEFAULT"')
                
        if self.normalize_rgb and 'rgb' in self.transform:
            self.transform['rgb'] = transforms.Compose(
                self.transform['rgb'].transforms +
                [transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)]

            )

        # Blur augmentation
        # if 'rgb' in self.transform:
        #     self.transform['rgb'] = transforms.Compose(
        #         self.transform['rgb'].transforms +
        #         [transforms.GaussianBlur(9, sigma=(0.1, 2.0))]
        #     )
        #     print('Blurred RGB (kernel size = 9)')

    
        # Saving some lists and dictionaries for fast lookup
        self.tbpv_dict = {} # Save task -> building -> point -> view dict
        self.url_dict = {}  # Save (task, building, point, view) -> URL dict
        self.bpv_count = {} # Dictionary to check if all (building, point, view) tuples have all tasks
        
        for task in self.tasks:
            self.tbpv_dict[task] = {}
            for url in self.urls[task]:

                building = url.split('/')[-3]

                file_name = url.split('/')[-1].split('_')
  
                point, view = file_name[1], file_name[3]

                # Populate url_dict
                self.url_dict[(task, building, point, view)] = url

                # Populate tbpv_dict
                if building not in self.tbpv_dict[task]:
                    self.tbpv_dict[task][building] = {}
                if point not in self.tbpv_dict[task][building]:
                    self.tbpv_dict[task][building][point] = []
                self.tbpv_dict[task][building][point].append(view)

                # Populate bpv_count
                if (building, point, view) not in self.bpv_count:
                    self.bpv_count[(building, point, view)] = 1
                else:
                    self.bpv_count[(building, point, view)] += 1


        # Remove entries that don't have all tasks and create list of all (building, point, view) tuples that contain all tasks
        self.bpv_list = [bpv_tuple for bpv_tuple, count in self.bpv_count.items() if count == len(self.tasks)]

        self.views = {}    # Build dictionary that contains all the views from a certain (building, point) tuple
        self.bpv_dict = {} # Save building -> point -> view dict
        for building, point, view in self.bpv_list:
            # Populate views
            if (building, point) not in self.views:
                self.views[(building, point)] = []
            self.views[(building, point)].append(view)

            # Populate bpv_dict
            if building not in self.bpv_dict:
                self.bpv_dict[building] = {}
            if point not in self.bpv_dict[building]:
                self.bpv_dict[building][point] = []
            self.bpv_dict[building][point].append(view)

        
        random.shuffle(self.bpv_list)
        
        end_time = perf_counter()
        self.num_points = len(self.views)
        self.num_images = len(self.bpv_list)
        self.num_buildings = len(self.bpv_dict)
        
        logger = logging.getLogger(__name__)
        logger.warning("Loaded {} images in {:0.2f} seconds".format(self.num_images, end_time - start_time))
        logger.warning("\t ({} buildings) ({} points) ({} images) for domains {}".format(self.num_buildings, self.num_points, self.num_images, self.tasks))


    def __len__(self):
        return len(self.bpv_list)

    def __getitem__(self, index):
        
        result = {}
        
        # Anchor building / point / view
        building, point, view = self.bpv_list[index]
        
        positive_views = [view]
        positive_samples = {}
        
        for task in self.tasks:
            task_samples = []
            for v in positive_views:
                path = self.url_dict[(task, building, point, v)]
                res = default_loader(path)

                if self.transform is not None and self.transform[task] is not None:
                    res = self.transform[task](res)

                # transforms for converting replica and hypersim labels to combined labels
                if task == 'segment_semantic' or task == 'semantic':
                    res2 = res.clone()
                    labels = torch.unique(res)
                    for old_label in labels:
                        if old_label == -1 or old_label == 255: continue
                        res[res2 == old_label] = REPLICA_LABEL_TRANSFORM[old_label]

                task_samples.append(res)

            task_samples = torch.stack(task_samples) if self.num_positive > 1 else task_samples[0]

            positive_samples[task] = task_samples
        
        positive_samples['point'] = point
        positive_samples['building'] = building
        result = positive_samples
        
        
        return result
            
            

    def randomize_order(self, seed=0):
        random.seed(0)
        random.shuffle(self.bpv_list)
    
    def task_config(self, task):
        return task_parameters[task]

    def _remove_unmatched_images(self, dataset_urls) -> (Dict[str, List[str]], int):
        '''
            Filters out point/view/building triplets that are not present for all tasks
            
            Returns:
                filtered_urls: Filtered Dict
                max_length: max([len(urls) for _, urls in filtered_urls.items()])
        '''
        n_images_task = [(len(obs), task) for task, obs in dataset_urls.items()]
        max_images = max(n_images_task)[0]
        if max(n_images_task)[0] == min(n_images_task)[0]:
            return dataset_urls, max_images
        else:
            print("Each task must have the same number of images. However, the max != min ({} != {}). Number of images per task is: \n\t{}".format(
                max(n_images_task)[0], min(n_images_task)[0], "\n\t".join([str(t) for t in n_images_task])))
            # Get views for each task
            def _parse_fpath_for_view( path ):
                url = path
 
                building = url.split('/')[-3]
                
                # building = os.path.basename(os.path.dirname(path))
                file_name = os.path.basename(path) 
                lf = parse_filename( file_name )
                return View(view=lf.view, point=lf.point, building=building)

            self.task_to_view = {}
            for task, paths in dataset_urls.items():
                self.task_to_view[task] = [_parse_fpath_for_view( path ) for path in paths]
    
            # Compute intersection
            intersection = None
            for task, uuids in self.task_to_view.items():
                if intersection is None:
                    intersection = set(uuids)
                else:
                    intersection = intersection.intersection(uuids)
            # Keep intersection
            print('Keeping intersection: ({} images/task)...'.format(len(intersection)))
            new_urls = {}
            for task, paths in dataset_urls.items():
                new_urls[task] = [path for path in paths if _parse_fpath_for_view( path ) in intersection]
            return new_urls, len(intersection)
        raise NotImplementedError('Reached the end of this function. You should not be seeing this!')


def make_dataset(dir, task, folders=None):
    if task == 'segment_semantic': task = 'semantic'
    #  folders are building names. 
    images = []
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        assert "bad directory"
    
    if folders is None:
        folders = os.listdir(dir)

    for folder in folders:
        folder_path = os.path.join(dir, folder, task)
        for fname in sorted(os.listdir(folder_path)):
            path = os.path.join(folder_path, fname)
            images.append(path)

    return images



def make_empty_like(data_dict):
    if not isinstance(data_dict, dict):
        if isinstance(data_dict, torch.Tensor):
            return torch.zeros_like(data_dict)
        elif isinstance(data_dict, list):
            return [make_empty_like(d) for d in data_dict]
        else:
            return type(data_dict)()
        raise NotImplementedError

    result = {}
    for k, v in data_dict.items():
        result[k] = make_empty_like(v)
    return result

