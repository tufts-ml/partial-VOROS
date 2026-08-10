import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_dataset_dir(dataset_path=None):
    repo_root = Path(__file__).resolve().parents[1]
    candidates = []

    if dataset_path is not None:
        candidate = Path(dataset_path).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        candidates.append(candidate)

    candidates.extend([
        repo_root / 'busi_data',
        repo_root / 'busi_training',
        repo_root / 'busi_training' / 'busi_embeddings',
        repo_root,
    ])

    for candidate in candidates:
        if candidate.exists():
            if (candidate / 'dataset_comment_list.csv').exists() and (candidate / 'labels.csv').exists():
                return candidate.resolve()

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (repo_root / 'busi_data').resolve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=None)
    args = parser.parse_args()

    BUSIDIR = resolve_dataset_dir(args.dataset_path)

    rawdf = pd.read_csv(BUSIDIR / 'dataset_comment_list.csv', delimiter=';')
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

    adf = pd.read_csv(BUSIDIR / 'labels.csv')
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
        BUSIDIR / 'all_labels_with_patient_id.csv',
        columns=['patient_id', 'study_id', 'is_malignant', 'label', 'basename', 'path', 'dq_reason'],
        index=False,
        header=True)

    qdf = adf.query("patient_id >= 0").copy()
    dqs = np.asarray([len(s) for s in qdf['dq_reason'].values])
    assert np.max(dqs) == 0 # verify all elements qualify
    qdf.to_csv(
        BUSIDIR / 'clean_labels_with_patient_id.csv',
        columns=['patient_id', 'study_id', 'is_malignant', 'label', 'basename', 'path'],
        index=False,
        header=True)

    print("saved clean CSV file with %d patients and %d images" % (
        qdf['patient_id'].unique().size,
        qdf.shape[0]))

