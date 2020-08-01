# Wine-quality-prediction
This project was done for a class competition in the PhD level course Advanced Statistical Learning at HEC Montreal.

In this project, we aim to predict the wine quality score (Y) given wine characteristics (Xs). Y is an ordinal variable where 1 refers to inferior, 2 to average, 3 to superior wine quality. We aim to maximize the accuracy.

When predicting an ordinal variable, we can try leveraging information from the categories order which may allow us getting better accuracy compared to treating the same variable as categorical. 
In this project, both approaches are compared. 

The data set for training contains only 2000 observations, and randomly splitting it into into a training and a test set 
did not work well. Depending on a random seed used to split the data, the test accuracy varied sometimes 10-15 percentage points for the same model type. The ranking of models by performance also changed from split to split. To get more reliable accuracy estimates, I use 10-fold cross-validation (CV) repeated 5 times. This, however, increased model training time substantially.

Several models were fit with the hyperparameter optimization using **caret::train** function. In parentheses, the name of the package, the function used and the method corresponding 
to that function in caret::train are shown:

*	Ordinal regression with elastic net penalty (ordinalNet::ordinalNet, method = 'ordinalNet'), including models with proportional (OrdReg.POM) or non-proportional odds (OrdReg.nonPOM) 
*	SVMs (kernlab::ksvm, methods = {'svmLinear', 'svmRadial', 'svmPoly'}) with linear, radial and polynomial kernels 
*	Decision tree on ordinal responses (rpartScore::rpartScore, method = 'rpartScore') 
*	Random forests (ranger::ranger, method = 'ranger')
*	Ordinal forests (ordinalForest::ordfor, not supported by caret::train, so CV was run manually) 
*	eXtreme Gradient Boosting (xgboost::xgboost, method = {'xgbTree', 'xgbLinear'}) using a tree as a base learner and also a model with a linear booster
*	AdaBag (adabag::bagging, method = 'AdaBag') and AdaBoost (adabag::boosting, method = 'AdaBoost.M1') 
*	Naive Bayes (naivebayes::naive_bayes, method = 'naive_bayes')

Random Forest performed the best in this task and was one of the fastest. The accuracy on the test set (without labels) was 0.6729 and the result ranked 8th out of 107 students.
The best result was 0.6781 also using caret & ranger. 

The same code is given in .R and .Rmd format.
