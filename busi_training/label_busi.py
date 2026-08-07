import argparse
import os
import re
import numpy as np
import pandas as pd

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='label_BUSI.py')
    parser.add_argument('--dataset_path', type=str, help='Directory where encoded images are saved', required=True)
    args = parser.parse_args()
    file_index = 0
    folder_df = pd.DataFrame(columns=['label', 'person_id', 'path'])
    for subdir, dirs, files in os.walk(args.dataset_path):
        for file in files:
            if re.search(r'\.pt$', file):
                label = re.split(r'\/', subdir)[-1]
                assert label in ['benign', 'malignant', 'normal'], 'Unexpected label: {}'.format(label)
                path = os.path.join(subdir, file)
                person_id = re.findall(r'\((\d+)\)', file)
                assert len(person_id) == 1, 'Unexpected file: {}'.format(file)
                folder_df.loc[file_index] = [label, person_id[0], path]
                file_index += 1
    # Sort df and add 'study_id' column
    folder_df['person_id'] = pd.to_numeric(folder_df['person_id'])
    folder_df = folder_df.sort_values(['label', 'person_id']).reset_index(drop=True)

    mask = folder_df.duplicated(['label', 'person_id'])
    study_id = pd.Series(np.nan, index=folder_df.index)
    study_id.loc[~mask] = np.arange(int((~mask).sum()))
    folder_df['study_id'] = study_id
    folder_df['study_id'] = folder_df['study_id'].ffill()
    # Write .csv to directory
    temp_df = folder_df[['study_id', 'label', 'path']].set_index('study_id')
    #temp_df.label = temp_df.label.apply(lambda label: [1,0,0] if label == 'normal' else [0,1,0] if label == 'benign' else [0,0,1])
    temp_df['is_malignant'] = temp_df.label.apply(lambda label: 1 if label == 'malignant' else 0)
    temp_df.to_csv(os.path.join(args.dataset_path, 'labels.csv'))
