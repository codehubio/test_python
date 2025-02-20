# Create the folders at the root folder (ignore if they were created)

- **files**: Store the standardized files
- **result**: Store the result of all cols
- **good_acc_files**: Store the result of which acc > 0.9
- **good_auc_files**: Store the result of which auc > 0.7

# Source file is toxcast_data.csv

## Run: python3 ./read.py

Result is written to folder **files**

## Run: python3 ./gat.py

Result is written to folder **result**

File names which acc >= 0.9 is written ito **good_file_list.txt**
