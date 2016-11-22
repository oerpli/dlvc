
### Group 1 ###

Score: 92/100

* `ImageVectorizer` and `FeatureVectorDataset` use hard-coded values that won't work with other datasets [-4]
* `test_*_dataset.py` scripts don't save samples [-2]
* `HDF5FeatureVectorDataset` should not hard-code paths (that what `fpath` is here for) [-2]
* Good report. Remark: Testing set does not prevent overfitting, it just alows us to identify overfitting.

