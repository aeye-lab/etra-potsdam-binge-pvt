# we allow 3 event_types: 'Fixation', 'Saccade', or 'None'
event_name_dict = {
                    'fixation': 'Fixation',
                    'saccade' : 'Saccade',
                  }


event_name_code_dict = {
                'Fixation': 0,
                'Saccade': 1,
                None: -1,
                }

detection_method_default_event = {
                    'ivt': 'Saccade',
                    'idt': 'Saccade',
                    'microsaccades': 'Fixation',
                                 }
                                 
feature_aggregations = ['mean', 'std', 'median', 'skew', 'kurtosis']

blink_threshold=0.6
blink_window_size=100
blink_min_duration=10
blink_velocity_threshold=0.1
tracker_loss_threshold=0.1
minimum_trial_id = 3
flag_use_eye_state_label = False


# learning params
param_grid={
        'n_estimators': [100,250,1000],
        'max_features': ['log2','sqrt',None],
        'max_depth': [32, 64, 128, None],
        #'min_samples_split': [
        #                    #0.05,0.01,
        #                    #0.1,0.2,
        #    0.05, 0.01, 0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        #'criterion': ['squared_error'],#, 'absolute_error'],
        'n_jobs': [-1],
    }

grid_search_verbosity = 0#10
n_outer_splits = 5
n_inner_splits = 2