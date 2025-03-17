for data_source in eyelink pupilcore 
do
    for window_size in 60000    
    do
    	for detection_method in idt
    	do
			python create_data_rf.py --detection-method ${detection_method} --window-size ${window_size} --stride ${window_size} --data-source ${data_source}
		done
	done
done