## Steps

```
$ BUSIPATH=/cluster/tufts/hugheslab/datasets/BUSI/kaggle_version

# STEP 1: Extract 768-dim ViT features from raw data
# output stored in $BUSIPATH/ViT_embeddings/
$ python -W encode_busi.py --dataset_path $BUSIPATH

# STEP 2: Build CSV of labels for all images in dataset
# output stored in $BUSIPATH/all_labels.csv
$ python -W label_busi.py --dataset_path $BUSIPATH

# STEP 3: Discard images following guidance from Pawlowska et al '23
# output storedin $BUSIPATH/clean_labels_with_patient_id.csv
$ python -W clean_busi.py --dataset_path $BUSIPATH

# STEP 4: Run LogisticRegression on embeddings (PCA optional via --reduced_dim flag)
# Splits respect the patient_id labels from STEP 3
# output written to stdout
$ python -W try_clf.py --random_state 101
```


## Origin

Preprocessing code for BUSI originally from C. Ratigan's work

<https://github.com/ChristopherRatigan/VOROS/tree/main>

Uses data documented in the README at

/cluster/tufts/hugheslab/datasets/BUSI/
