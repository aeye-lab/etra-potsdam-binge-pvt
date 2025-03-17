from dataclasses import dataclass
from dataclasses import field
from typing import Any
import argparse

import config.config as config

import sys
sys.path.insert(0,'/mnt/mlshare/prasse/aeye_git/pymovements/src/')   #TODO: change before release
import pymovements as pm


import os
import numpy as np
import polars as pl
import joblib
import sys


from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn import metrics
from sklearn.metrics import mean_squared_error

import preprocessing.feature_extraction as feature_extraction
import train_classification_model as train_classification_model
import config.config as config


def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()    
    parser.add_argument(
        '--save-dir', type=str,
        default='data/',
    )
    parser.add_argument(
        '--label-file', type=str,
        default='label.csv',
    )
    parser.add_argument(
        '--sampling-rate', type=int, default=1000)
    parser.add_argument(
        '--minimum-duration', type=int,
        default=100,
    )
    parser.add_argument(
        '--dispersion-threshold', type=float,
        default=1.0,
    )
    parser.add_argument(
        '--velocity-threshold', type=float,
        default=20.0,
    )
    parser.add_argument(
        '--flag-redo', type=int,
        default=1,
    )
    parser.add_argument(
        '--data-source', type=str,
        default='eyelink',
    )
    parser.add_argument('--detection-method', type=str, default='idt')
    parser.add_argument('--window-size', type=int, default=10000)
    parser.add_argument('--stride', type=int, default=5000)
    args = parser.parse_args()
    return args
    

def evaluate_model(args):
    save_dir = args.save_dir
    label_file = args.label_file
    detection_method = args.detection_method    
    result_prefix = detection_method
    flag_redo = args.flag_redo
    window_size = args.window_size
    data_source = args.data_source
    stride = args.stride    
    
    if flag_redo == 1:
        flag_redo = True
    else:
        flag_redo = False   
    
    if data_source == 'eyelink':
        use_eye_closure_features = False
        use_pupil_features = True        
    elif data_source == 'pupilcore':
        use_eye_closure_features = True
        use_pupil_features = True
    
    # detection method params
    minimum_duration = args.minimum_duration
    dispersion_threshold = args.dispersion_threshold
    velocity_threshold = args.velocity_threshold
    
    if detection_method == 'ivt':
        detection_params = {'minimum_duration': minimum_duration,
                            'velocity_threshold': velocity_threshold,
                        }
    elif detection_method == 'idt':
        detection_params = {'minimum_duration': minimum_duration,
                            'dispersion_threshold': dispersion_threshold,
                        }
    elif detection_method == 'microsaccades':
        detection_params = {'minimum_duration': minimum_duration,
                        }
    
    detection_param_string = detection_method + '_'
    for key in detection_params:
        detection_param_string += str(key) + '_' + str(detection_params[key]) + '_'
    detection_param_string = detection_param_string[0:len(detection_param_string)-1]
    save_path = save_dir + detection_param_string + '_' + data_source +\
                '_window_' + str(window_size) + '_stride_' +\
                str(stride) + '.joblib' 
                
    
    if not flag_redo and os.path.exists(save_path):
        return None
    
    feature_aggregations = config.feature_aggregations
    blink_threshold = config.blink_threshold
    blink_window_size = config.blink_window_size
    blink_min_duration = config.blink_min_duration
    blink_velocity_threshold = config.blink_velocity_threshold
    flag_use_eye_state_label = config.flag_use_eye_state_label
    
    param_grid = config.param_grid
    grid_search_verbosity = config.grid_search_verbosity
    tracker_loss_threshold = config.tracker_loss_threshold
    minimum_trial_id = config.minimum_trial_id
    
    #################################################
    #
    # Load data
    #
    #################################################
    
    if data_source == 'eyelink':
        dataset = pm.Dataset('PotsdamBingeRemotePVT', path='data/PotsdamBingeRemotePVT')
    elif data_source == 'pupilcore':
        dataset = pm.Dataset('PotsdamBingeWearablePVT', path='data/PotsdamBingeWearablePVT')

    try:
        dataset.load()
    except:
        dataset.download()
        dataset.load()

    
    
    filepath = list(dataset.fileinfo['gaze']['filepath'])
    subject_ids = []
    session_ids = []
    filename_list = []
    condition = []
    trial_ids = []
    for filename in filepath:
        filename_clean = filename.split('/')[-1]
        filename_clean = filename_clean.split('/')[-1].replace('.csv','')
        subject_ids.append(int(filename_clean.split('_')[0]))
        session_ids.append(int(filename_clean.split('_')[1]))
        trial_ids.append(int(filename_clean.split('_')[3]))
        filename_list.append(filename_clean + '.csv')
        condition.append(filename_clean.split('_')[2])
    
    #################################################
    #
    # Filter and chunk data
    #
    #################################################
    
        
    use_gazes = []
    use_subject_ids = []
    use_session_ids = []
    use_trial_ids   = []
    use_condition = []
    use_filenames = []
    use_middletimes = []
    bac = []
    for i in tqdm(np.arange(len(dataset.gaze))):
        if  data_source == 'pupilcore':
            dataset.gaze[i].frame = dataset.gaze[i].frame.with_columns(pl.Series(name="eye_closure", values=dataset.gaze[i].frame['pupil_confidence_interpolated']))
            dataset.gaze[i].frame = dataset.gaze[i].frame.with_columns(pl.Series(name="pupil_left", values=dataset.gaze[i].frame['pupil_size_pupilcore_interpolated']))
        if  data_source == 'eyelink':
            dataset.gaze[i].frame = dataset.gaze[i].frame.with_columns(pl.Series(name="pupil_left", values=dataset.gaze[i].frame['pupil_size_eyelink']))
        cur_gaze = dataset.gaze[i].frame
        cur_condition = condition[i]
        cur_filename = list(dataset.fileinfo['gaze']['filepath'])[i]
        # filter data
        cur_gaze = cur_gaze.filter(pl.col('time') > 0)
        num_chunks = int(np.floor((cur_gaze.shape[0] - window_size) / stride))
        for j in range(num_chunks):
            cur_subset = cur_gaze[(j*stride):(j*stride) + window_size,:]
            cur_bac = 0.0
            use_instance = True
            middle_id = int(np.round(cur_subset.shape[0] / 2))
            if cur_condition == 'a':
                try:                
                    time_middle_prev = np.array(cur_subset['time_to_prev_bac'])[middle_id]
                    time_middle_next = np.array(cur_subset['time_to_next_bac'])[middle_id]
                    sum_times = time_middle_prev + time_middle_next
                    prev_bac = np.array(cur_subset['prev_bac'])[0]
                    next_bac = np.array(cur_subset['next_bac'])[0]
                    
                    middle_bac = (1 - (time_middle_prev / sum_times)) * prev_bac
                    middle_bac += (1 - (time_middle_next / sum_times)) * next_bac
                    cur_bac = middle_bac
                except:
                    use_instance = False
            cur_middle_time = np.array(cur_subset['time'])[middle_id]
            if use_instance:
                use_gazes.append(pm.gaze.gaze_dataframe.GazeDataFrame(data=cur_subset,
                                                                      experiment = dataset.gaze[0].experiment,
                                                                      trial_columns = ['trial_id'],
                                                                     ))
                use_subject_ids.append(subject_ids[i])
                use_session_ids.append(session_ids[i])
                use_trial_ids.append(trial_ids[i])
                use_condition.append(condition[i])
                use_filenames.append(cur_filename)
                bac.append(cur_bac)
                use_middletimes.append(cur_middle_time)
            
    use_subject_ids = np.array(use_subject_ids)
    use_session_ids = np.array(use_session_ids)
    use_condition = np.array(use_condition)
    use_filenames = np.array(use_filenames)
    use_trial_ids = np.array(use_trial_ids)
    bac = np.array(bac)
    use_middletimes = np.array(use_middletimes)
    
    dataset.gaze = use_gazes
    dataset.fileinfo['gaze'] = pl.DataFrame({'subject_id': use_subject_ids,
                                             'filepath': use_filenames,
                                             'session_id': use_session_ids})
    subject_ids = use_subject_ids
    session_ids = use_session_ids
    condition = use_condition
    trial_ids = use_trial_ids
    middle_times = use_middletimes
    
    
    sampling_rate = dataset.definition.experiment.sampling_rate

    # transform pixel coordinates to degrees of visual angle
    dataset.pix2deg()

    # transform positional data to velocity data
    dataset.pos2vel()

    # detect events
    dataset.detect(detection_method)
    

    label_grouping = None
    instance_grouping = ['trial_id','block_id']
    splitting_criterion = None
    sampling_rate = dataset.definition.experiment.sampling_rate
    label_grouping = instance_grouping
    splitting_criterion = ['subject_id']
    
    # create features
    
    feature_matrix, group_names, splitting_names, combined_feature_names = train_classification_model.get_feature_matrix_parallel(dataset,
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
                            return_feature_names=True,
                            use_eye_closure_features=use_eye_closure_features,
                            use_pupil_features=use_pupil_features,
                            n_jobs=40,
                            flag_use_eye_state_label=flag_use_eye_state_label,
                            )
    
    
    # calculate tracker loss
    tracker_loss = []
    for file_id, (gaze, fileinfo_row) in tqdm(
                    enumerate(zip(dataset.gaze, dataset.fileinfo['gaze'].to_dicts())),
                    disable=False,
            ):
        tracker_loss.append(np.sum(np.isnan(np.array(gaze.frame['pixel_x']))) / len(gaze.frame))
    

    y = bac
    subjects = subject_ids
         
    joblib.dump({'y':y,
             'subjects':subjects,
             'feature_matrix':feature_matrix,
             'condition':condition,
             'feature_names': combined_feature_names,
             'bac':bac,
             'tracker_loss':tracker_loss,
             'session_ids':session_ids,
             'trial_ids':trial_ids,
             'middle_times':middle_times,
            },
            save_path,
            compress=3, protocol=2)   
    
def main() -> int:
    args = get_argument_parser()
    evaluate_model(args)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
