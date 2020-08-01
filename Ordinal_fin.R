knitr::opts_chunk$set(echo = TRUE)

# install.packages("corrplot")
# install.packages("e1071")
# install.packages("ranger")
# install.packages("caret")
# install.packages("MLmetrics")
# install.packages("ordinalForest")
# install.packages("naivebayes")
# install.packages("kernlab")
# install.packages("ordinalNet")
# install.packages("doParallel")
# install.packages("rpartScore")
# install.packages("xgboost")

library(ggplot2)
library(dplyr)
# library(e1071)
library(corrplot)
library(ranger)
library(caret)
library("MLmetrics")
library("ordinalForest")
library(naivebayes)
library("kernlab")
library("ordinalNet")
library("doParallel")
library("rpartScore")
library("xgboost")

datrain <- read.csv("~/HEC/!Advanced statistical learning - 80619A/homework_H2020/datrain.txt", sep="")

summary(datrain)
n = names(datrain)
n = n[-12]

# check the distribution of Y - imbalanced class problem
summary(as.factor(datrain$y))/2000
barplot(summary(as.factor(datrain$y))/2000)

# look at the distributions of Xs
for (i in n) {
  print(i)
  hist(datrain[[i]],main=paste(i))
}

# look at the distributions of Xs
for (i in n) {
  print(i)
  hist(log(datrain[[i]]),main=paste(i))
}

# look at boxplots of Y vs other variables
for (i in n) {
  boxplot(datrain[[i]]~datrain$y,main=paste(i,"vs Y"))
}

# look at boxplots of Y vs other variables and density plots of Xs vs Y
for (i in n) {
  plot =  ggplot(datrain,aes(x=datrain[,i],col=factor(y)))+
    geom_density()+xlab(i)
  print(plot)
}

correl = cor(datrain)
# cor of y with oher vars
as.matrix(correl[12,])
corrplot(correl)

options(digits=3)
correl 

train1 = datrain
train1$y.fact = as.factor(datrain$y)
# train1$y.fact = as.ordered(train$y)
levels(train1$y.fact) = c("inferior","average","superior")
table(train1$y.fact)/2000
train1 = train1[-12]
names(train1)

# 5 repeats of 10-fold CV
set.seed(1234)
myFolds <- createMultiFolds(datrain$y, k = 10, times=5)

# creating train control
myControl <- trainControl(
  summaryFunction = multiClassSummary,
  classProbs = TRUE, 
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds,
  allowParallel = TRUE
)

nbGrid = data.frame(
  laplace = 0,
  usekernel = TRUE,
  adjust = c(0.1,0.2,0.3,0.4,0.5,1,1.5,2)
)

nb1 <- train(
  y.fact ~ .,
  data = train1,
  tuneGrid = nbGrid,
  method = "naive_bayes",
  trControl = myControl
)

nb1

plot(nb1)

RFgrid <- data.frame(
  mtry = rep(c(2,3,4,5,7,11),each=2),#c(2:11)
  splitrule = c("gini","extratrees"),
  min.node.size = 5
)

max.cl.sh = max(prop.table(table(train$y)))
w = as.numeric(max.cl.sh/prop.table(table(train$y))) # correct!!!
w

set.seed(2222)
rf1 <- train(
  y.fact ~ .,
  data = train1,
  tuneGrid = RFgrid,
  method = "ranger",
  class.weights=w,
  trControl = myControl
)

rf1

plot(rf1)

set.seed(2222)
rf2 <- train(
  y.fact ~ .,
  data = train1,
  tuneGrid = RFgrid,
  method = "ranger",
  trControl = myControl
)

rf2

plot(rf2)

rf2$pred %>%
  filter(mtry==2, splitrule=="extratrees") %>%
  summarize(mean.accuracy = mean(pred==obs))

rf2.p = rf2$pred%>%
  filter(mtry==2, splitrule=="extratrees") %>%
  select(pred,obs)

table(rf2.p$obs, rf2.p$pred)

# Cart.ord.grid2 <- data.frame(
#   cp = rep(c(0,0.005,0.01),each=4),
#   split = rep(c("abs","quad"),6),
#   prune = rep(rep(c("mr", "mc"),each=2),3)
#   )

Cart.ord.grid2 <- data.frame(
  cp = 0.005,
  split = "abs",
  prune = "mc"
  )

CART.ord2 <- train(
  y.fact ~ .,
  data = train1,
  tuneGrid = Cart.ord.grid2,
  method = "rpartScore",
  trControl = myControl
)
# Fitting cp = 0.005, split = abs, prune = mc on full training set

CART.ord2

plot(CART.ord2)

# XGB4.6.8.10.grid <- data.frame(
#   nrounds = rep(seq(50,750,50),12), # 15 values of nrounds
#   max_depth = rep(rep(c(4,6,8,10),each=15),3), 
#   eta = rep(c(0.05,0.1,0.3),each=60), 
#   gamma = 0, 
#   colsample_bytree = 1, 
#  min_child_weight = 1,
#  subsample = 1
#   )

XGB4.6.8.10.grid <- data.frame(
  nrounds = 250, 
  max_depth = 8, 
  eta = 0.05, 
  gamma = 0, 
  colsample_bytree = 1, 
 min_child_weight = 1,
 subsample = 1
  )

set.seed(567)
XGB4.6.8.10 <- train(
  y.fact ~ .,
  data = train1,
  tuneGrid = XGB4.6.8.10.grid,
  method = "xgbTree",
  trControl = myControl
)
# The final values used for the model were nrounds = 250, max_depth = 8, eta = 0.05, gamma = 0, colsample_bytree =
# 1, min_child_weight = 1 and subsample = 1.

XGB4.6.8.10 

plot(XGB4.6.8.10)

set.seed(567)
XGB.lin <- train(
  y.fact ~ .,
  data = train1,
  tuneGrid = data.frame(nrounds = 50, lambda = 1e-04, alpha = 0, eta = 0.3), # should be commented for CV.
  method = "xgbLinear",
  trControl = myControl
)
# The final values used for the model were nrounds = 50, lambda = 1e-04, alpha = 0 and eta = 0.3.

XGB.lin

plot(XGB.lin)

# cls = makeCluster(4) 
# registerDoParallel(cls)

# Adabag.grid1 <- data.frame(
#   maxdepth=rep(c(3,6,9,10,11,15),3), mfinal=rep(c(50,100,150),each=6)
#   )

# set.seed(567)
# AdaBag1 <- train(
#   y.fact ~ .,
#   tuneGrid = Adabag.grid1,
#   data = train1,
#   method = "AdaBag",
#   trControl = myControl
# )
# # The final values used for the model were mfinal = 150 and maxdepth = 11 on sample n=1500.

# AdaBag1

# plot(AdaBag1)

# Adabag.grid2 <- data.frame(
#   maxdepth=rep(c(12,14,16,18),3), mfinal=rep(c(50,100,150),each=4)
#   )

# set.seed(567)
# AdaBag2 <- train(
#   y.fact ~ .,
#   tuneGrid = Adabag.grid2,
#   data = train1,
#   method = "AdaBag",
#   trControl = myControl
# )
# # The final values used for the model were mfinal = 150 and maxdepth = 14 on sample n=1500.

# AdaBag2

# plot(AdaBag2)

Adabag.grid3 <- data.frame(
  maxdepth=15, mfinal=rep(c(50,100,150))
  )

set.seed(567)
AdaBag3 <- train(
  y.fact ~ .,
  tuneGrid = Adabag.grid3,
  data = train1,
  method = "AdaBag",
  trControl = myControl
)

AdaBag3

plot(AdaBag3)

# Adaboost.grid1 <- data.frame(
#   maxdepth=rep(c(1,2,3,6),3), 
#   mfinal=rep(c(50,100,150),each=4),
#   coeflearn="Freund"
#   )

# Runs about 2.5h, so commented.
# set.seed(567)
# AdaBoost1 <- train(
#   y.fact ~ .,
#   data = train1,
#   tuneGrid = Adaboost.grid1, 
#   method = "AdaBoost.M1",
#   trControl = myControl
# )
# Fitting mfinal = 150, maxdepth = 6, coeflearn = Freund on full training set

# AdaBoost1

# plot(AdaBoost1)

Adaboost.grid2 <- data.frame(
  maxdepth=rep(c(10),15), 
  mfinal=seq(20,160,10),
  coeflearn="Freund"
  )

# still runs 30-40 mins
set.seed(567)
AdaBoost2 <- train(
  y.fact ~ .,
  data = train1,
  tuneGrid = Adaboost.grid2,
  method = "AdaBoost.M1",
  trControl = myControl
)
# Fitting mfinal = 130, maxdepth = 10, coeflearn = Freund on full training set

AdaBoost2

plot(AdaBoost2)

svc.grid1 <- data.frame(
  # C=16
  C=2^(seq(-2,5,1))
  )

# Runs fast
svc1 <- train(
  y.fact ~ .,
  data = train1,
  tuneGrid = svc.grid1, 
  method = "svmLinear",
  # preProc = c("center", "scale"),
  preProcess = "pca",
  trControl = myControl
)
# The final value used for the model was C = 0.5 with PCA.
# The final value used for the model was C = 0.25 with center/scale.
# Valid. accuracy a bit higher with pca.
# maximum number of iterations reached almost for every fold/run, so results are unsure.

svc1

plot(svc1)

# Runs fast
svc2 <- train(
  y.fact ~ .,
  data = train1,
  tuneLength = 10,
  method = "svmRadialCost",
  preProc = c("center", "scale"),
  trControl = myControl
)
# The final value used for the model was C = 16.

svc2

plot(svc2)

svc3 <- train(
  y.fact ~ .,
  data = train1,
  tuneGrid= data.frame(degree = 3, scale = 0.01, C = 0.25), # to comment for CV.
  method = "svmPoly",
  preProc = c("center", "scale"),
  trControl = myControl
)
# Fitting degree = 3, scale = 0.01, C = 0.25 on full training set

svc3

plot(svc3)

train[myFolds[[50]],]
nrow(train[myFolds[[50]],])

train[-myFolds[[49]],]

# OF5.acc = NULL
# OF5.conf.mat = list()
# OF5.rec=data.frame(NULL)
# OF5.prec=data.frame(NULL)
# 
# 
# for (i in 1:length(myFolds)) {
#   set.seed(2222)
#   OF.prop = ordfor(depvar="y.fact", train1[myFolds[[i]],], nsets = 1000, ntreeperdiv = 100,
#                    ntreefinal = 5000, perffunction = "proportional", nbest = 10,
#                    naive = FALSE, num.threads = 8, npermtrial = 500,
#                    permperdefault = FALSE, mtry = 5, min.node.size = 5,
#                    replace = TRUE, keep.inbag = FALSE)
#   # predict on the new data, collect only class predictions with $pred
#   yhat = predict(OF.prop,newdata = train1[-myFolds[[i]],])$ypred
# 
#   yfact = train1[-myFolds[[i]],"y.fact"]
#   
#   # using custom function
#   cm = conf.acc(yfact,yhat) # cm is a list
#   OF5.conf.mat = c(OF5.conf.mat,cm[1])
# 
#   OF5.acc = c(OF5.acc,cm[[2]])
#   print(paste(i,"accuracy:",cm[[2]]))
#   
#   OF5.rec = rbind(OF5.rec,cm[[3]])
#   
#   OF5.prec = rbind(OF5.prec,cm[[4]])
# }

# mean(OF5.acc)
# median(OF5.acc)
# boxplot(OF5.acc)

ORDgrid1 <- data.frame(
  alpha = 0.3,
  # alpha = rep(seq(0.1,0.9,0.2),each=2),
  link = "logit",
  # link = rep(c("logit","probit"),5),
  criteria = "aic"
  )

ord.en1 <- train(
  y.fact ~ .,
  data = train1,
  # tuneLength = 10,
  tuneGrid=ORDgrid1,
  method = "ordinalNet",
  trControl = myControl
)
# Fitting alpha = 0.3, criteria = aic, link = logit on full training set (on n=1350) 

ord.en1

ord.en1.2 <- train(
  y.fact ~ .,
  data = train1,
  # tuneLength = 10,
  tuneGrid=ORDgrid1,
  method = "ordinalNet",
  preProc = c("center", "scale"),
  trControl = myControl
)
# Fitting alpha = 0.3, criteria = aic, link = logit on full training set (on n=1350) 

ord.en1.2

ord.en1.3 <- train(
  y.fact ~ .,
  data = train1,
  # tuneLength = 10,
  tuneGrid=ORDgrid1,
  method = "ordinalNet",
  preProc = "pca",
  trControl = myControl
)
# Fitting alpha = 0.3, criteria = aic, link = logit on full training set (on n=1350) 

ord.en1.3

ORDgrid2 <- data.frame(
  alpha = 0.5,
  # alpha = rep(seq(0.1,0.9,0.2),each=2),
  link = "logit",
  # link = rep(c("logit","probit"),5),
  criteria = "aic"
  )

# allow preprocessing and nonparallelTerms
ord.en2 <- train(
  y.fact ~ .,
  data = train1,
  tuneGrid = ORDgrid2,
  method = "ordinalNet",
  # preProc = c("center", "scale"),
  preProcess = "pca",
  trControl = myControl,
  nonparallelTerms = TRUE
)
# The final values used for the model were alpha = 0.5, criteria = aic and link = logit.

ord.en2

# allow preprocessing and nonparallelTerms
ord.en2.2 <- train(
  y.fact ~ .,
  data = train1,
  tuneGrid = ORDgrid2,
  method = "ordinalNet",
  # preProc = c("center", "scale"),
  trControl = myControl,
  nonparallelTerms = TRUE
)
# The final values used for the model were alpha = 0.5, criteria = aic and link = logit.

ord.en2.2

# allow preprocessing and nonparallelTerms
ord.en2.3 <- train(
  y.fact ~ .,
  data = train1,
  tuneGrid = ORDgrid2,
  method = "ordinalNet",
  preProc = c("center", "scale"),
  trControl = myControl,
  nonparallelTerms = TRUE
)
# The final values used for the model were alpha = 0.5, criteria = aic and link = logit.

ord.en2.3

models <- resamples(list(NB = nb1, RF.classW = rf1, RF = rf2, CART.ord = CART.ord2,XGB.tree=XGB4.6.8.10,XGB.lin=XGB.lin, AdaBag = AdaBag3,AdaBoost = AdaBoost2,svmLinear = svc1,svmRadial = svc2,svmPoly = svc3,OrdReg.POM = ord.en1.3,OrdReg.nonPOM = ord.en2)) #
summary(models)

bwplot(models, metric = "Accuracy")

models2 <- resamples(list(RF.classW = rf1, RF = rf2,XGB.tree=XGB4.6.8.10,XGB.lin=XGB.lin, AdaBag = AdaBag3,AdaBoost = AdaBoost2)) #

differences = diff(models2, test = t.test, confLevel = 0.95, adjustment = "bonferroni")
# pdf("diff.pdf")
dotplot(differences)
# dev.off(dev.cur())

# useless command. extracts predictions for training data, getting almost 100% accuracy everywhere.

pred = extractPrediction(list(RF.classW = rf1, RF = rf2,XGB.tree=XGB4.6.8.10,AdaBag = AdaBag3,AdaBoost = AdaBoost2))
# also extractProb(list(models))

pred %>%
  group_by(object) %>%
  summarize(mean.accuracy = mean(pred==obs))

rf2$pred %>%
  filter(mtry==2, splitrule=="extratrees") %>%
  summarize(mean.accuracy = mean(pred==obs))

rf2.p = rf2$pred%>%
  filter(mtry==2, splitrule=="extratrees") %>%
  select(pred,obs)

table(rf2.p$obs, rf2.p$pred)

head(rf2.p)


rf1$pred %>%
  filter(mtry==3, splitrule=="extratrees") %>%
  summarize(mean.accuracy = mean(pred==obs))

rf1.p = rf1$pred%>%
  filter(mtry==3, splitrule=="extratrees") %>%
  select(pred,obs)

table(rf1.p$obs, rf1.p$pred)

head(rf1.p)

AdaBag3$pred %>%
  filter(mfinal==150) %>%
  summarize(mean.accuracy = mean(pred==obs))

AdaBag3.p = AdaBag3$pred%>%
  filter(mfinal==150) %>%
  select(pred,obs)

table(AdaBag3.p$obs, AdaBag3.p$pred)
head(AdaBag3.p)


AdaBoost2$pred %>%
  filter(maxdepth==10, mfinal==130) %>%
  summarize(mean.accuracy = mean(pred==obs))

AdaBoost2.p = AdaBoost2$pred %>%
  filter(maxdepth==10, mfinal==130) %>%
  select(pred,obs)

table(AdaBoost2.p$obs, AdaBoost2.p$pred)

XGB4.6.8.10$pred %>%
  filter(nrounds == 250, max_depth == 8, eta == 0.05) %>%
  summarize(mean.accuracy = mean(pred==obs))

XGB4.6.8.10.p = XGB4.6.8.10$pred%>%
  filter(nrounds == 250, max_depth == 8, eta == 0.05) %>%
  select(pred,obs)

table(XGB4.6.8.10.p$obs, XGB4.6.8.10.p$pred)
head(XGB4.6.8.10.p)

top5_pred = cbind(rf2.p$pred,rf1.p$pred,AdaBoost2.p$pred,XGB4.6.8.10.p$pred,AdaBag3.p$pred,rf2.p$obs)

maj_vote = NULL
for (i in 1:nrow(top5_pred)){
  maj_vote = c(maj_vote, names(which.max(table(top5_pred[i,1:5]))))
}

# accuracy of majority voting prediction 
mean(maj_vote==top5_pred[,6])


# predictions of maj voting vs the best model coinside by 94.32%
mean(maj_vote==top5_pred[,1])

# predictions of two RFs coinside by 95.39%
mean(top5_pred[,1]==top5_pred[,2])

# confusion matrix for the best single model
table(rf2.p$obs, rf2.p$pred)

# confusion matrix for the majority voting with top-5 models

table(rf2.p$obs, maj_vote)

test.set <- read.csv("~/HEC/!Advanced statistical learning - 80619A/homework_H2020/dateststudent.txt", sep="")
nrow(test.set)

rf1.small = rf1$pred %>%
  filter(mtry == 3, splitrule == "extratrees")

table(rf1.small$obs,rf1.small$pred)/10000 # all 2000 cases predicted 5 times due to repeat CV
prop.table(table(rf1.small$pred)) 

rf2.small = rf2$pred %>%
  filter(mtry == 2, splitrule == "extratrees")

table(rf2.small$obs,rf2.small$pred)/10000  
prop.table(table(rf2.small$pred)) 


prediction = predict(rf2, newdata = test.set) #non-weighted
prediction1 = predict(rf1, newdata = test.set) #weighted
prediction2 = predict(AdaBoost2, newdata = test.set) #weighted
prediction3 = predict(XGB4.6.8.10, newdata = test.set) #weighted
prediction4 = predict(AdaBag3, newdata = test.set) #weighted

top5_pred_test = cbind(prediction,prediction1,prediction2,prediction3,prediction4)

maj_vote_test = NULL
for (i in 1:nrow(top5_pred_test)){
  maj_vote_test = c(maj_vote_test, names(which.max(table(top5_pred_test[i,]))))
}

# predictions of maj voting vs the best model coinside by 94%
mean(maj_vote_test==top5_pred_test[,1])

head(prediction)
table(prediction)
class(prediction)

pred_numeric = as.numeric(prediction)
table(pred_numeric)
prop.table(table(pred_numeric)) # prop looks reasonably close to that on the valid. set. 
