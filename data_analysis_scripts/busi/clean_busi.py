import pandas as pd
import numpy as np
import os

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
        default='/cluster/tufts/hugheslab/datasets/BUSI/')
    args = parser.parse_args()

    BUSIDIR = args.dataset_path

    rawdf = pd.read_csv(os.path.join(BUSIDIR, 'dataset_comment_list.csv'),
        delimiter=';')
    # Goal: remove any objections, such as
    # - benign/malig conflicting label
    # - axilla images

    df = rawdf.query("Objection != 'axilla'").copy()
    df = df.query("Objection != 'needle'").copy()
    kdf = df.query("Objection != 'multiclass'").copy()
    mdf = df.query("Objection == 'multiclass'").copy()

    hasb = mdf.Filename.str.contains('benign')
    hasm = mdf.Filename.str.contains('malignant')
    kmdf = mdf.loc[~np.logical_and(hasb, hasm)]

    kdf = pd.concat([kdf, kmdf])

    adf = pd.read_csv(os.path.join(BUSIDIR, 'labels.csv'))
    adf['basename'] = [a.split(os.path.sep)[-1].split('.')[0] for a in adf['path'].values]

    pid = np.zeros(adf.shape[0], dtype=np.int32)
    dq_reason = ['' for a in range(adf.shape[0])]
    for rowid, basename in enumerate(adf['basename'].values):
        match_mask = kdf['Filename'].str.contains(basename, regex=False)
        try:
            assert match_mask.sum() == 1
            pid[rowid] = match_mask.argmax()
        except Exception:
            pid[rowid] = -999
            ii = rawdf['Filename'].str.contains(basename, regex=False).argmax()
            reason = rawdf['Objection'].values[ii]
            if reason == 'multiclass':
                reason = 'class label conflict'
            dq_reason[rowid] = 'disqualified: %s' % reason
    adf['patient_id'] = pid
    adf['dq_reason'] = dq_reason

    adf.to_csv(
        os.path.join(BUSIDIR, 'all_labels_with_patient_id.csv'),
        columns=['patient_id', 'study_id', 'is_malignant', 'label', 'basename', 'path', 'dq_reason'],
        index=False,
        header=True)

    qdf = adf.query("patient_id >= 0").copy()
    dqs = np.asarray([len(s) for s in qdf['dq_reason'].values])
    assert np.max(dqs) == 0 # verify all elements qualify
    qdf.to_csv(
        os.path.join(BUSIDIR, 'clean_labels_with_patient_id.csv'),
        columns=['patient_id', 'study_id', 'is_malignant', 'label', 'basename', 'path'],
        index=False,
        header=True)

    print("saved clean CSV file with %d patients and %d images" % (
        qdf['patient_id'].unique().size,
        qdf.shape[0]))

