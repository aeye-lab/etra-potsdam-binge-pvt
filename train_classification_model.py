from __future__ import annotations

import argparse
import sys

import os
import numpy as np
import polars as pl
import joblib
import pymovements as pm
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn import metrics
from joblib import Parallel, delayed


import preprocessing.feature_extraction as feature_extraction
import config.config as config

def get_feature_matrix_parallel(dataset,
                    sampling_rate,
                    blink_threshold,
                    blink_window_size,
                    blink_min_duration,
                    blink_velocity_threshold,
                    feature_aggregations,
                    detection_method,
                    label_grouping,
                    instance_grouping,
                    splitting_criterion,
                    max_len=None,
                    return_feature_names=False,
                    use_eye_closure_features=False,
                    use_pupil_features=False,
                    n_jobs=40,
                    flag_use_eye_state_label = False,
                    ):
                        
    event_name_dict = config.event_name_dict
    event_name_code_dict = config.event_name_code_dict
    detection_method_default_event = config.detection_method_default_event    
                        
    num_add = 1000
    group_names = []
    splitting_names = []
    iter_counter = 0
    
    dataset_list = []
    print('  ##  Create instances  ##')
    for i in tqdm(np.arange(len(dataset.gaze))):
        cur_gaze_df = dataset.gaze[i]
        try:
            cur_gaze_df.unnest()
        except:
            pass
        cur_gaze_df = cur_gaze_df.frame
        if 'position_xl' in cur_gaze_df.columns:
            cur_gaze_df =  cur_gaze_df.rename({'position_xl':'position_x',
                                'position_yl':'position_y',
                                'velocity_xl':'velocity_x',
                                'velocity_yl':'velocity_y',
                               })
        cur_event_df = dataset.events[i].frame
    
        # add events to gaze df
        # initialize event_type as None
        event_type = np.array([event_name_code_dict[detection_method_default_event[detection_method]] for _ in range(cur_gaze_df.shape[0])], dtype=np.int32)
        for event_id in range(cur_event_df.shape[0]):
            cur_event = cur_event_df[event_id]
            cur_onset_time = cur_event_df[event_id]['onset'][0]
            cur_offset_time = cur_event_df[event_id]['offset'][0]
            if 'index' in cur_gaze_df.columns:
                cur_onset_id = cur_gaze_df.filter(pl.col('time') == cur_onset_time)['index'][0]
                cur_offset_id = cur_gaze_df.filter(pl.col('time') == cur_offset_time)['index'][0]
            else:
                cur_onset_id = cur_gaze_df.with_row_index().filter(pl.col('time').cast(int) == cur_onset_time)['index'][0]
                cur_offset_id = cur_gaze_df.with_row_index().filter(pl.col('time').cast(int) == cur_offset_time)['index'][0]
            event_type[cur_onset_id:cur_offset_id] = event_name_code_dict[event_name_dict[cur_event_df[event_id]['name'][0]]]
    
        if 'postion_x' in cur_gaze_df.columns:
            pos_x = np.array(cur_gaze_df['position_x'].is_null())
        else:
            pos_x = np.zeros([cur_gaze_df.shape[0],])
        if 'velocity_x' in cur_gaze_df.columns:
            vel_x = np.array(cur_gaze_df['velocity_x'].is_null())
        else:
            vel_x = np.zeros([cur_gaze_df.shape[0],])
        if 'pixel_x' in cur_gaze_df.columns:
            pix_x = np.array(cur_gaze_df['pixel_x'].is_null())
        else:
            pix_x = np.zeros([cur_gaze_df.shape[0],])
        null_ids = np.logical_or(pos_x,
                             np.logical_or(vel_x,
                            np.array(pix_x)))
        non_ids = np.where(null_ids)[0]     
        event_type[non_ids] = -1
        cur_gaze_df = cur_gaze_df.with_columns(pl.Series(name="event_type", values=event_type))
        
        for name, data in cur_gaze_df.group_by(instance_grouping):
            if max_len is not None:
                if data.shape[0] > max_len:
                    data = data[0:max_len,:]
            dataset_list.append(data)
            label_tuple = []
            for jj in range(len(label_grouping)):
                label_tuple.append(str(data[label_grouping[jj]][0]))
            label_tuple = '_'.join(label_tuple)
            group_names.append(label_tuple)
            splitting_names.append(data[splitting_criterion][0])
            continue
    
    # extract features in parallel
    print('  ##  Extract features  ##')
    from joblib import Parallel, delayed
    n_jobs = n_jobs    
    
    use_ids = []
    num_per_fold = n_jobs
    num_folds = int(np.ceil(len(dataset_list) / n_jobs))
    for i in range(num_folds):
        use_ids.append(np.arange(i*num_per_fold,np.min([(i+1)*num_per_fold,len(dataset_list)]),1))
    
    # check for empty lists in use_ids
    t_use_ids = []
    for i in range(len(use_ids)):
        if len(use_ids[i]) > 0:
            t_use_ids.append(use_ids[i])
    use_ids = t_use_ids
    num_parallel = n_jobs
    start_id = 0
    for p_run in tqdm(np.arange(len(use_ids))):
        X_features_list = Parallel(n_jobs=num_parallel,verbose=0)(delayed(feature_extraction.compute_features)(dataset_list[i],
                                                                    sampling_rate,
                                                                    blink_threshold,
                                                                    blink_window_size,
                                                                    blink_min_duration,
                                                                    blink_velocity_threshold,
                                                                    feature_aggregations,
                                                                    use_eye_closure_features,
                                                                    use_pupil_features,
                                                                    flag_use_eye_state_label,
                                                                    ) for i in use_ids[p_run])
        
        res, feature_names = zip(*X_features_list)
        if p_run == 0:
            X_features = np.zeros([len(dataset_list),res[0].shape[0]])
        
        for i in range(len(res)):
            cur_data = res[i]
            X_features[start_id] = cur_data
            start_id += 1
    
    feature_matrix = X_features
    
    combined_feature_names = []
    for i in range(len(feature_names)):
        if len(feature_names[i]) > len(combined_feature_names):
            combined_feature_names = feature_names[i]
    feature_matrix[np.isnan(feature_matrix)] = 0.0
    if return_feature_names:
        return feature_matrix, group_names, splitting_names, combined_feature_names
    else:
        return feature_matrix, group_names, splitting_names
