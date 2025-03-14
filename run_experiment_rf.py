from dataclasses import dataclass
from dataclasses import field
from typing import Any
import argparse

import config.config as config

import sys
# to be changed after release!!!
sys.path.append('/mnt/mlshare/prasse/aeye_git/pymovements/src/')
from pymovements.gaze.experiment import Experiment
from pymovements.dataset.dataset_definition import DatasetDefinition
from pymovements.dataset.dataset_library import register_dataset
from pymovements.dataset.dataset_paths import DatasetPaths

import os
import numpy as np
import polars as pl
import joblib
import pymovements as pm
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn import metrics
from sklearn.metrics import mean_squared_error
#import shap
sys.path.append('/mnt/mlshare/prasse/aeye_git/eye-movement-preprocessing/')
import preprocessing.feature_extraction as feature_extraction
import train_classification_model as train_classification_model
import config.config as config




def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save-dir', type=str,
        default='results/',
    )    
    parser.add_argument(
        '--flag-redo', type=int,
        default=0,
    )
    parser.add_argument(
        '--bac-threshold', type=float,
        default=0.0,
    )
    parser.add_argument(
        '--input-file', type=str,
        default='data/'
    )
    parser.add_argument('--min-sacc-count', type=int, default=10)
    parser.add_argument('--group-variable', type=str, default='subject') #trial, random, subject
    parser.add_argument('--feature-set', type=str, default='all') # 'all', 'pupil', 'closure','onlyvel','onlypupil','onlyclosure'
    parser.add_argument('--max-folds', type=int, default=-1) # -1 -> standard, 0 -> loo
    parser.add_argument('--save-scores', type=int, default=0) # 0 -> no, 1 -> yes
    args = parser.parse_args()
    return args

def evaluate(args):    
    param_grid=config.param_grid
    tracker_loss_threshold = config.tracker_loss_threshold
    grid_search_verbosity=config.grid_search_verbosity
    load_path = args.input_file
    result_dir = args.save_dir
    bac_threshold = args.bac_threshold
    min_sacc_count = args.min_sacc_count
    group_variable = args.group_variable
    feature_set = args.feature_set
    max_folds = args.max_folds
    save_scores = args.save_scores
    
    if save_scores == 1:
        save_scores = True
        appendix = '_save_scores'
    else:
        save_scores = False
        appendix = ''
    
    save_path = load_path.split('/')[-1].replace('.joblib','_bac_' + str(bac_threshold) +\
                            '_splitting_' + str(group_variable) +\
                            '_feature_' + str(feature_set) + '_rf' + appendix + '.csv')
    if max_folds != -1:
        save_path = save_path.replace('.csv','_maxfolds_' + str(max_folds) + '.csv')
        if max_folds != 0:
            config.n_outer_splits = max_folds
            config.n_inner_splits = max_folds
    
    save_path = result_dir + save_path
    
    print('from: ' + str(load_path) + ' -> ' + str(save_path))
    flag_redo = False
    if flag_redo == 1:
        flag_redo = True
    if os.path.exists(save_path) and not flag_redo:
        print('skipping... already exists')
        return None
    data = joblib.load(load_path)
    #print(data.keys())
    #print(np.unique(data['events'], return_counts=True))
    #print(allo)
    y = data['y']
    subjects = data['subjects']
    feature_matrix = data['feature_matrix']
    condition = data['condition']
    feature_names = data['feature_names']
    tracker_loss = np.array(data['tracker_loss'])
    session_ids = np.array(data['session_ids'])
    trial_ids = np.array(data['trial_ids'])
    middle_times = np.array(data['middle_times'])

    # filter by tracker loss
    use_ids = np.where(np.array(tracker_loss) <= tracker_loss_threshold)[0]
    subjects = subjects[use_ids]
    feature_matrix = feature_matrix[use_ids]
    condition = condition[use_ids]
    tracker_loss = tracker_loss[use_ids]
    y = y[use_ids]
    session_ids = session_ids[use_ids]
    trial_ids = trial_ids[use_ids]
    middle_times = middle_times[use_ids]
    
    
    # get session and trial in session ids to split on
    session_trial_names = []
    subject_trial_session_dict = dict()
    for i in range(len(subjects)):
        cur_s = subjects[i]
        cur_t = trial_ids[i]
        cur_ses = session_ids[i]
        cur_time = middle_times[i]
        cur_str = str(cur_s) + '_' + str(cur_t) + '_' + str(cur_ses)
        session_trial_names.append(cur_str)
        if cur_str not in subject_trial_session_dict:
            subject_trial_session_dict[cur_str] = []
        subject_trial_session_dict[cur_str].append(cur_time)


    subject_trial_time_dict = dict()
    for elem in subject_trial_session_dict:
        cur_list = np.array(subject_trial_session_dict[elem])
        sort_list = cur_list[np.argsort(cur_list)]#[::-1]]
        for s_i in range(len(sort_list)):
            s_elem = sort_list[s_i]
            cur_str = elem + '_' + str(s_elem)
            subject_trial_time_dict[cur_str] = s_i
            
    session_in_trial_names = []
    for i in range(len(subjects)):
        cur_s = subjects[i]
        cur_t = trial_ids[i]
        cur_ses = session_ids[i]
        cur_time = middle_times[i]
        cur_str = str(cur_s) + '_' + str(cur_t) + '_' + str(cur_ses) + '_' + str(cur_time)
        session_in_trial_names.append(subject_trial_time_dict[cur_str])

    session_trial_names = np.array(session_trial_names)
    session_in_trial_names = np.array(session_in_trial_names)
    
    
    # filter by saccade count
    use_ids = np.where(feature_matrix[:,np.where(np.array(data['feature_names']) == 'count_saccades')[0]] >= min_sacc_count)[0]
    subjects = subjects[use_ids]
    feature_matrix = feature_matrix[use_ids]
    condition = condition[use_ids]
    tracker_loss = tracker_loss[use_ids]
    y = y[use_ids]
    session_ids = session_ids[use_ids]
    trial_ids = trial_ids[use_ids]
    middle_times = middle_times[use_ids]
    session_trial_names = session_trial_names[use_ids]
    session_in_trial_names = session_in_trial_names[use_ids]
    
    # create label
    pos_ids = np.where(y > bac_threshold)[0]
    neg_ids = np.where(y == 0)[0]
    use_ids = np.concatenate([pos_ids,neg_ids])
    y_bac = y[use_ids]
    y = np.concatenate([np.ones([len(pos_ids),]),
                        np.zeros([len(neg_ids),]),
                        ])
    subjects = subjects[use_ids]
    feature_matrix = feature_matrix[use_ids]
    condition = condition[use_ids]
    tracker_loss = tracker_loss[use_ids]
    session_ids = session_ids[use_ids]
    trial_ids = trial_ids[use_ids]
    middle_times = middle_times[use_ids]
    session_trial_names = session_trial_names[use_ids]
    session_in_trial_names = session_in_trial_names[use_ids]
    
    
    # shuffle data
    rand_ids = np.random.permutation(np.arange(len(trial_ids)))
    y = y[rand_ids]
    y_bac = y_bac[rand_ids]
    subjects = subjects[rand_ids]
    feature_matrix = feature_matrix[rand_ids]
    condition = condition[rand_ids]
    tracker_loss = tracker_loss[rand_ids]
    session_ids = session_ids[rand_ids]
    trial_ids = trial_ids[rand_ids]
    middle_times = middle_times[rand_ids]
    session_trial_names = session_trial_names[rand_ids]
    session_in_trial_names = session_in_trial_names[rand_ids]


    # select features
    if feature_set == 'all':
        feature_ids = np.arange(feature_matrix.shape[1])
    elif feature_set == 'pupil':
        feature_ids = []
        for i in range(len(feature_names)):
            if 'pupil' not in feature_names[i]:
                feature_ids.append(i)
        feature_ids = np.array(feature_ids)
    elif feature_set == 'closure':
        feature_ids = []
        for i in range(len(feature_names)):
            if 'eye_closure' not in feature_names[i]:
                feature_ids.append(i)
        feature_ids = np.array(feature_ids)
    elif feature_set == 'onlyvel':
        feature_ids = []
        for i in range(len(feature_names)):
            if 'eye_closure' not in feature_names[i]:
                if 'pupil' not in feature_names[i]:
                    feature_ids.append(i)
        feature_ids = np.array(feature_ids)
    elif feature_set == 'onlypupil':
        feature_ids = []
        for i in range(len(feature_names)):
            if 'pupil' in feature_names[i]:
                feature_ids.append(i)
        feature_ids = np.array(feature_ids)
    elif feature_set == 'onlyclosure':
        feature_ids = []
        for i in range(len(feature_names)):
            if 'pupil' in feature_names[i]:
                feature_ids.append(i)
        feature_ids = np.array(feature_ids)
    feature_matrix = feature_matrix[:,feature_ids]


    # select variable to split on
    if group_variable == 'trial':
        group_values = trial_ids
        outer_group_kfold = GroupKFold(n_splits=len(np.unique(trial_ids)))
        inner_group_kfold= KFold(n_splits=config.n_inner_splits)
    elif group_variable == 'subject':
        group_values = subjects
        if max_folds == 0:
            config.n_outer_splits = len(np.unique(group_values))
        outer_group_kfold = GroupKFold(n_splits=config.n_outer_splits)
        inner_group_kfold=GroupKFold(n_splits=config.n_inner_splits)
    elif group_variable == 'random':
        outer_group_kfold = KFold(n_splits=config.n_outer_splits)
        inner_group_kfold=KFold(n_splits=config.n_inner_splits)
        group_values = subjects
    elif group_variable == 'trial-time':
        group_values = session_in_trial_names
        max_groups = len(np.unique(group_values))
        use_splits = np.min([config.n_outer_splits,
                            max_groups])
        outer_group_kfold = GroupKFold(n_splits=use_splits)
        inner_group_kfold= KFold(n_splits=config.n_inner_splits)
    
    
    print(' === Evaluating model ===')
    print(' === Number of subjects: ' + str(len(np.unique(subjects))) + ' === ')
    aucs_instance = []
    aucs_subject = []
    fpr_instance = []
    fpr_subject = []
    tpr_instance = []
    tpr_subject = []
    fold_dicts = []
    for i, (train_index, test_index) in enumerate(outer_group_kfold.split(feature_matrix, y, group_values)):
        X_train = feature_matrix[train_index]
        y_train = y[train_index]
        y_train_bac = y_bac[train_index]
        subjects_train = subjects[train_index]
        session_train = session_ids[train_index]
        trial_train = trial_ids[train_index]
        X_test = feature_matrix[test_index]
        y_test = y[test_index]
        y_test_bac = y_bac[test_index]
        subjects_test = subjects[test_index]
        session_test = session_ids[test_index]
        trial_test = trial_ids[test_index]
        
        X_train[np.isnan(X_train)] = 0
        X_test[np.isnan(X_test)]   = 0
        
        print(' === Fold ' + str(i+1) + ' of ' + str(outer_group_kfold.n_splits) + ' ===')
        # rf
        rf = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=param_grid, verbose=grid_search_verbosity,
            cv = inner_group_kfold,
        )
        rf.fit(X_train, y_train, groups = subjects_train)

        # dummy classifier
        dr = DummyClassifier()
        dr.fit(X_train,y_train)

        best_parameters = rf.best_params_
        predictions_rf = rf.predict(X_test)
        pred_proba = rf.predict_proba(X_test)
        predictions_dr = dr.predict(X_test)

        fpr, tpr, _ = metrics.roc_curve(y_test, pred_proba[:,1], pos_label=1)
        fpr_instance.append(fpr)
        tpr_instance.append(tpr)
        auc = metrics.auc(fpr, tpr)
        aucs_instance.append(auc)

        combined_str = []
        for j in range(len(subjects_test)):
            combined_str.append(str(subjects_test[j]) + '_' + str(session_test[j]) + '_' + str(trial_test[j]))
        combined_str = np.array(combined_str)
        unique_s_ids = list(np.unique(combined_str))
        test_subject_scores = []
        test_subject_label = []
        for j in range(len(unique_s_ids)):
            cur_id = unique_s_ids[j]
            cur_ids = np.where(combined_str == cur_id)[0]
            cur_ys = y_test[cur_ids][0]
            cur_score = np.mean(pred_proba[:,1][cur_ids])
            test_subject_scores.append(cur_score)
            test_subject_label.append(cur_ys)
        fpr, tpr, _ = metrics.roc_curve(test_subject_label, test_subject_scores, pos_label=1)
        fpr_subject.append(fpr)
        tpr_subject.append(tpr)
        auc = metrics.auc(fpr, tpr)
        aucs_subject.append(auc)
        print('AUC in fold ' + str(i+1) + ': ' + str(auc))
        
        fold_dicts.append({'y_test':y_test,
                            'y_test_bac':y_test_bac,
                            'subjects_test':subjects_test,
                            'session_test':session_test,
                            'trial_test':trial_test,
                            'predictions':pred_proba[:,1],
                            'X_test':X_test,})

    #dr_df = pl.DataFrame({'fold':np.arange(len(dummy_mses)),
    #                  'auc': dummy_mses})
    #dr_df.write_csv(save_path.replace('.csv','_dr.csv'))
    rf_df = pl.DataFrame({'fold':np.arange(len(aucs_instance)),
                      'aucs_instance': aucs_instance,
                      'aucs_subject':aucs_subject,
                      })     
    rf_df.write_csv(save_path)
    if save_scores:
        joblib.dump({'fpr_instance':fpr_instance,
                    'tpr_instance':tpr_instance,
                    'fpr_subject':fpr_subject,
                    'tpr_subject':tpr_subject,
                    'fold_dicts':fold_dicts,
                    }, save_path.replace('.csv','.joblib'), compress=3, protocol=2)
    else:
        joblib.dump({'fpr_instance':fpr_instance,
                    'tpr_instance':tpr_instance,
                    'fpr_subject':fpr_subject,
                    'tpr_subject':tpr_subject,
                    }, save_path.replace('.csv','.joblib'), compress=3, protocol=2)
    
    
    
def main() -> int:
    args = get_argument_parser()
    evaluate(args)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())