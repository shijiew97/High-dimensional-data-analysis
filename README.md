# STAT-718-Deep-Learning-Project

### Abstract 
In this project, we apply penalized likelihood method and feature importance variable selection method to deal with overfitting problem in high dimension dataset. Thereafter,we fit fifteen model on the train data by cross validation to select several prospective model
candidates. Then, by means of boosting and stacking method, we find out boosted SVM and 2 layer stacked model with final layer to be LDA. Our final model is a voting ensemble of LDA, boosted SVM and stacked model, which has an accuracy of 0.782 on the test dataset.

### Main code
`Model.py` is the `Python` code for machine learning project.
