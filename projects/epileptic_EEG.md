Bonn EEG dataset

Epilepsy is a neurological disorder characterized by seizures. 
University of Bonn has [a dataset](http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3) of epileptic EEGs. 
The dataset has 5 sets (A to E) of data, sampled at from different types of subjects, under different conditions, and from different parts of the brain. 

Your goal is the following binary classification tasks:
1. A vs. B
2. C vs .D
3. A and B vs. C and D
4. C and D vs. E 
5. C vs.  D

In your report, please try to explain the medical sense of the 5 binary classifications above. Information of the 5 sets can be found in the Bonn paper mentioned in the link above. You can access the paper thru e-journal portal of ISU library. Do NOT pay your pocket money to buy the paper. 

Try at least the following classifier approaches: 
1. SVMs with manual features extraction
2. NNs with manual features extraction
3. NNs with raw data
3. Decision trees with manual features extraction
4. Random forests with manual features extraction

Report results using different classifier approaches. Because a classifier's performance depends on the hyperparameters, please also use grid search to find the optimal ones. For a fair comparision, please use cross validations when reporting the performance of classifiers. 

In most classifier approaches above, you need to extract features for each time series segment before feeding to a classifier. Please try the at least the following feature extraction toolboxes: 
1. PyEEG
2. Shapelet 

In your report, please report the importance and ranks features. 

You need to filter the time series as mentioned in the link above. 
