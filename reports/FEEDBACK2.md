
* Batch generator transforms samples only once. While this improves efficiency, it does not work with random transformations for data augmentation. Please change this behavior for assignment 3.
* `PerChannelDivisionImageTransformation` and `DivisionTransformation`s `from_dataset_stddev()` do not apply `tform` [-1]
* `test_sample_transformations.py` produces incorrect output with chained per-channel transformations (different mean and max value) [-1]
* You are tuning the learning rate decay hyperparameter, not weight decay (see Keras docs for details) [-2]
* Bonus task implemented [+5 bonus points]
* Report: good explanations and very nice visualizations.
* Good work!
* **Score: 100/100**

