# Detection of Alcohol Inebriation from Eye Movements using Remote and Wearable Eye Trackers
This repository contains the data and the code for the paper: 'Detection of Alcohol Inebriation from Eye Movements using Remote and Wearable Eye Trackers'
## Pipeline to reproduce the results from the paper
* run ```create_data_rf.sh``` if you want to create the training data (create features)
    * the data was already created and can be found in ```data/```
* run ```run_rf.sh``` to run all the models (this trains the random forest models)
 
### Plotting
* run ```plot_results_eyelink.ipynb``` to plot and show the results for the remote eye-tracker
* run ```plot_results_pupilcore.ipynb``` to plot and show the results for the wearable eye-tracker
* run ```plot_results.ipynb``` to plot and show the results for the results for Figure 5


# Data Collection
We collected eye movement and eye-closure sequences of 57 participants. Data was collected using two different eye-trackers. We used the Pupil Core eye-tracking glasses (Pupil Labs) to track the left eye and the EyeLink 1000 Plus eye tracker (SR Research) to track the right eye. The data can be found here: https://osf.io/qf7e6/

## Format of data
For each subject and experimental session we provide a file ```[subjectID]_[sessionID]_[condition]_[trial_id]_[block_id].csv``` containing raw eye tracking data collected with the Pupil Core glasses and the EyeLink 1000 Plus eye tracker (resampled to 1,000 Hz).
* ```[condition]``` is a character decoding the type of condition during the recording
  * ```a```: alcohol recording
  * ```b``` and ```e```: baseline recording
* ```[subjectID]_[sessionID]_[condition]_[trial_id]_[block_id].csv```

| column  | type | description|
| ------------- | ------------- | ------------- |
| trial_id      | float       | Trial-id for current recording   |
| block_id   | float        | Block-id for current recording      |
|x_pix_eyelink | float |x-pixel coordinates using eyelink remote eye-tracker (EyeLink 1000 Plus)|
|y_pix_eyelink | float | y-pixel coordinates using eyelink remote eye-tracker (EyeLink 1000 Plus)|
|eyelink_timestamp | int |Timestamp or recording in ms (EyeLink 1000 Plus)|
|x_pix_pupilcore_interpolated | float | x-pixel coordinates using pupil-core eye-tracker upsampled to 1,000 Hz (Pupil Core eye-tracking glasses) |
|y_pix_pupilcore_interpolated | float | y-pixel coordinates using pupil-core eye-tracker upsampled to 1,000 Hz (Pupil Core eye-tracking glasses)|
|pupil_size_eyelink | float | Pupil-size of pupil using eyelink remote eye-tracker (Pupil Core eye-tracking glasses)|
|target_distance | float | Distance to eyelink remote eye-tracker (screen) in mm (EyeLink 1000 Plus)|
|pupil_size_pupilcore_interpolated | float | Pupil-size of pupil pupil-core eye-tracker upsampled to 1,000 Hz (Pupil Core eye-tracking glasses)|
|pupil_confidence_interpolated | float | Pupil detection confidence of pupil pupil-core eye-tracker upsampled to 1,000 Hz (Pupil Core eye-tracking glasses)|
| time_to_prev_bac| float | Elapsed time from previous BAC testing in ms |
|time_to_next_bac | float | Remaining time for next BAC testing in ms |
|prev_bac | float | Previous BAC concentration |
|next_bac | float | Next BAC concentration |
