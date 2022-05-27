# Files in this folder:

1. <u>ComputeFBeta</u>

   It is used for compute f1 score between two json files.



# How to run your code
## Part A
```python
# Face detection on validation data
python FaceDetector.py --input_path validation_folder/images --output ./results_val.json

# Validation


# Face detection on test data
python FaceDetector.py --input_path test_folder/images --output ./results.json
```

## Part B
```python
python FaceCluster.py --input_path faceCluster_5 --num_cluster 5
```

