for bac_threshold in 0 #0.01 0.02 0.03
do
	for input in data/idt_minimum_duration_100_dispersion_threshold_1.0_eyelink_window_60000_stride_60000.joblib data/idt_minimum_duration_100_dispersion_threshold_1.0_pupilcore_window_60000_stride_60000.joblib #`find data/ -name "*.joblib" -type f`;
	do
		for group_variable in trial-time trial random subject
		do
			for feature_set in all pupil closure onlyvel onlypupil onlyclosure
			do
				echo "python run_experiment_rf.py --bac-threshold ${bac_threshold} --input-file ${input} --group-variable ${group_variable} --feature-set ${feature_set} --save-scores 1"
				python run_experiment_rf.py --bac-threshold ${bac_threshold} --input-file ${input} --group-variable ${group_variable} --feature-set ${feature_set} --save-scores 1
			done
		done
	done
done
