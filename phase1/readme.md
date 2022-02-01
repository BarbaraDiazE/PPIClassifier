### In order to execute the pipeline. Follow the next instructions.

## Train models
#### Train Desition Tree
`python driver_code/train_DT.py`

#### Train Random Forest
`python driver_code/train_RF.py`

#### Train logistic regression
`python driver_code/train_LRG.py`

#### Train support vector machine algorithms
`python driver_code/train_SVM.py`

## Generate id and storage customization information

#### Generate numerical ID (PK) for trained models
`python driver_code/generate_numerical_id.py`

#### Get information related to trained models in a json file
`python driver_code/information_models.py`

## Metrics

#### Plot models' metrics scatter plot
`python driver_code/metrics/plot_metrics.py`

#### Filter models with best performance
`python driver_code/metrics/model_filtering.py`

#### Plot headmap of models with bests performance to make easier models selection
`python driver_code/metrics/headmap_plot.py`

#### Get Confusion Matrix
`python driver_code/metrics/get_confusin_matrix.py`

## Validations

#### run validations for each algorithm
`python driver_code/validations/validation_dt.py`\
`python driver_code/validations/validation_rf.py`\
`python driver_code/validations/validation_lrg.py`\
`python driver_code/validations/validation_svm.py`

## Ensemble
#### Train
`python driver_code/ensemble/ensemble.py`

#### Validate
`python driver_code/ensemble/validation_ensemble.py`