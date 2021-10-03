Step 1:  

Use 'pre_processing.py' to obtain a csv file from the original 'SCORE_csv.xlsx' file.  
Need to 'pip uninstall xlrd' in order to import excel files.  

Step 2:

Use 'embedding2_csv_average_embbedings.py' for the averaged embedding or use 'embedding3_csv_concatenate_embbedings.py' for the concatenated embedding. Both use pre-trained RoBERTa language model.  
  
Need to run under flair environment, which can be created by the commands below:  
conda create -n flair python=3.6  
conda activate flair  
pip install flair  

