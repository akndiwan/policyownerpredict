## ----setup, include=FALSE------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ------------------------------------------------------------------------
library(caret)
library(glmnet)
library(pROC)
library(e1071)
library(randomForest)
library(gbm)
library(glmnet)
library(arules)
library(arulesViz)
library(fastDummies)
library(rpart)
library(dplyr)
library(class)

fun <- function(x){ 
  a <- mean(x) 
  b <- sd(x)
  if(b == 0){
    (x-a)
  }
  else{
  (x - a)/(b) 
  }
}



## ------------------------------------------------------------------------
caravan_train <- read.csv("CaravanTrain.csv")
caravan_test <- read.csv("CaravanTest.csv")
caravan_target <- read.csv("TestingTarget.csv")
caravan_train_over <- read.csv("CaravanTrain_Over.csv")
caravan_train_under <- read.csv("CaravanTrain_under.csv")
names(caravan_target)[1] = c("Caravan.Policy")
caravan_target$Caravan.Policy <- as.factor(caravan_target$Caravan.Policy)



## ------------------------------------------------------------------------
#Fitting logistic regression model
caravan_logistic <- glm(Caravan.Policy~.,family="binomial", data = caravan_train, control=list(maxit=100))



## ------------------------------------------------------------------------
# Do the prediction
predict_logistic <- predict(caravan_logistic, newdata = caravan_test)
predict_logistic <- as.factor(as.numeric(predict_logistic>0.5))

# Make the confusion matrix
confusionMatrix(data = predict_logistic, caravan_target$Caravan.Policy, positive = "1")


## ------------------------------------------------------------------------
# Ridge Regression
# make x and y
y <- caravan_train$Caravan.Policy
x <- caravan_train %>% select(-Caravan.Policy)%>%
  data.matrix()

# Ridge regression parameter space
lambdas <- 10^seq(3, -2, by = -.1)

# fit ridge regression model
caravan_ridge <- glmnet(x, y,family="binomial", alpha = 0, lambda = lambdas)
summary(caravan_ridge)



## ------------------------------------------------------------------------
#Tune hyper parameter of Ridge Regression
#We'll automatically find a value for lambda that is optimal by using cv.glmnet()

caravan_ridge_cv <- cv.glmnet(x, y,family="binomial", alpha = 0, lambda = lambdas)



## ------------------------------------------------------------------------
#The lowest point in the curve is the optimal lambda. the log value of lambda that best minimized the error in cross-validation. We can get this value by:
opt_lambda <- caravan_ridge_cv$lambda.min
opt_lambda



## ------------------------------------------------------------------------
fit_ridge <- caravan_ridge_cv$glmnet.fit
summary(fit_ridge)



## ------------------------------------------------------------------------
predict_ridge <- predict(fit_ridge, s = opt_lambda, newx = data.matrix(caravan_test))

# Do the prediction
predict_ridge <- as.factor(as.numeric(predict_ridge>0.5))

# Make the confusion matrix
confusionMatrix(data = predict_ridge, caravan_target$Caravan.Policy, positive = "1")



## ------------------------------------------------------------------------
# Lasso Regression
caravan_lasso <- glmnet(x, y,family="binomial", alpha = 1, lambda = lambdas)
summary(caravan_lasso)



## ------------------------------------------------------------------------
#Tune hyper parameter of Lasso Regression
caravan_lasso_cv <- cv.glmnet(x, y,family="binomial", alpha = 1, lambda = lambdas)



## ------------------------------------------------------------------------
#cv.glmnet() uses cross-validation to work out how well each model generalizes, we now visualize 
plot(caravan_lasso_cv)



## ------------------------------------------------------------------------
#The lowest point in the curve is the optimal lambda. the log value of lambda that best minimized the error in cross-validation. We can get this value by:
opt_lambda <- caravan_lasso_cv$lambda.min
opt_lambda



## ------------------------------------------------------------------------
fit_lasso <- caravan_lasso_cv$glmnet.fit
summary(fit_lasso)   



## ------------------------------------------------------------------------
predict_lasso <- predict(fit_lasso, s = opt_lambda, newx = data.matrix(caravan_test))

# Do the prediction
predict_lasso <- as.factor(as.numeric(predict_lasso>0.5))

# Make the confusion matrix
confusionMatrix(data = predict_lasso, caravan_target$Caravan.Policy, positive = "1")


## ------------------------------------------------------------------------
#KNN for full dataset

#scaling for KNN
caravan_train[,1:57] <- apply(caravan_train[,1:57], 2, fun)
set.seed(12345)
inTrain <- createDataPartition(caravan_train$Caravan.Policy, p=0.7, list=FALSE)

#training and validation split
dftrain1 <- caravan_train[inTrain,]
dfvalidation1 <- caravan_train[-inTrain,]

#creating the datasets
train_input <- as.matrix(dftrain1[,-58])
train_output <- as.vector(dftrain1[,58])
validate_input <- as.matrix(dfvalidation1[,-58])

#
kmax <- 15
ER1 <- rep(0,kmax)
ER2 <- rep(0,kmax)

#finding best k using validation set
for (i in 1:kmax){
  prediction <- knn(train_input, train_input, train_output, k=i)
  prediction2 <- knn(train_input, validate_input, train_output, k=i)
  
  CM1 <- table(prediction, dftrain1$Caravan.Policy)
  
  ER1[i] <- (CM1[1,2]+CM1[2,1])/sum(CM1)
  
  CM2 <- table(prediction2, dfvalidation1$Caravan.Policy)
  
  ER2[i] <- (CM2[1,2]+CM2[2,1])/sum(CM2)
}
z <- which.min(ER2)


#run knn on test data and form confusion matrix
test_input <- as.matrix(caravan_test[])
#
predictknn3 <- knn(train_input, test_input, train_output, k=z)
#
confusionMatrix(predictknn3, caravan_target$Caravan.Policy, positive = "1")



## ------------------------------------------------------------------------
#KNN for undersampled data
caravan_train_under[,1:57] <- apply(caravan_train_under[,1:57], 2, fun)
set.seed(12345)
inTrain <- createDataPartition(caravan_train_under$Caravan.Policy, p=0.7, list=FALSE)

#scaling for KNN
dftrain1 <- data.frame(caravan_train_under[inTrain,])
dfvalidation1 <- data.frame(caravan_train_under[-inTrain,])

#training and validation split
train_input <- as.matrix(dftrain1[,-58])
train_output <- as.vector(dftrain1[,58])
validate_input <- as.matrix(dfvalidation1[,-58])

#
kmax <- 15
ER1 <- rep(0,kmax)
ER2 <- rep(0,kmax)

#finding the best k using validation set
for (i in 1:kmax){
  prediction3 <- knn(train_input, train_input, train_output, k=i)
  prediction4 <- knn(train_input, validate_input, train_output, k=i)
  
  CM1 <- table(prediction3, dftrain1$Caravan.Policy)
  
  ER1[i] <- (CM1[1,2]+CM1[2,1])/sum(CM1)
  
  CM2 <- table(prediction4, dfvalidation1$Caravan.Policy)
  
  ER2[i] <- (CM2[1,2]+CM2[2,1])/sum(CM2)
}
z <- which.min(ER2)

#run knn on test data and form confusion matrix
test_input <- as.matrix(caravan_test[])
#
predictknn3 <- knn(train_input, test_input,train_output, k=z)
#
confusionMatrix(predictknn3, caravan_target$Caravan.Policy, positive = "1")


## ------------------------------------------------------------------------
#KNN for oversampled data

#scaling for knn
caravan_train_over[,1:57] <- apply(caravan_train_over[,1:57], 2, fun)
set.seed(12345)
inTrain <- createDataPartition(caravan_train_over$Caravan.Policy, p=0.7, list=FALSE)

#training and validation split
dftrain1 <- data.frame(caravan_train_over[inTrain,])
dfvalidation1 <- data.frame(caravan_train_over[-inTrain,])

#creating training validation datasets
train_input <- as.matrix(dftrain1[,-58])
train_output <- as.vector(dftrain1[,58])
validate_input <- as.matrix(dfvalidation1[,-58])

#
kmax <- 15
ER1 <- rep(0,kmax)
ER2 <- rep(0,kmax)

#running for optimal k
for (i in 1:kmax){
  prediction <- knn(train_input, train_input,train_output, k=i)
  prediction2 <- knn(train_input, validate_input,train_output, k=i)
  
  CM1 <- table(prediction, dftrain1$Caravan.Policy)
  
  ER1[i] <- (CM1[1,2]+CM1[2,1])/sum(CM1)
  
  CM2 <- table(prediction2, dfvalidation1$Caravan.Policy)
  
  ER2[i] <- (CM2[1,2]+CM2[2,1])/sum(CM2)
}

z <- which.min(ER2)

##run knn on test data and form confusion matrix
test_input <- as.matrix(caravan_test[])
#
predictknn3 <- knn(train_input, test_input,train_output, k=z)
#
confusionMatrix(predictknn3, caravan_target$Caravan.Policy, positive = "1")




## ------------------------------------------------------------------------
#Naive Bayes
caravan_train_factor <- data.frame(sapply(caravan_train, as.factor))
caravan_test_factor <- data.frame(sapply(caravan_test, as.factor))

caravan_nb <- naiveBayes(Caravan.Policy~., data = caravan_train_factor)

predict_nb <- predict(caravan_nb, newdata = caravan_test_factor[])
confusionMatrix(data = predict_nb, caravan_target$Caravan.Policy, positive = "1")



## ------------------------------------------------------------------------
#Naive Bayes Undersampled
caravan_train_under_factor <- data.frame(sapply(caravan_train_under, as.factor))

caravan_nb_under <- naiveBayes(Caravan.Policy~., data = caravan_train_under_factor)

predict_nb_under <- predict(caravan_nb_under, newdata = caravan_test_factor[])
confusionMatrix(data = predict_nb_under, caravan_target$Caravan.Policy, positive = "1")


## ------------------------------------------------------------------------
#Naive Bayes Oversampled
caravan_train_over_factor <- data.frame(sapply(caravan_train_over, as.factor))

caravan_nb_over <- naiveBayes(Caravan.Policy~., data = caravan_train_over_factor)

predict_nb_over <- predict(caravan_nb_over, newdata = caravan_test_factor[])
confusionMatrix(data = predict_nb_over, caravan_target$Caravan.Policy, positive = "1")


## ------------------------------------------------------------------------
#Classification Trees
caravan_tree <- rpart(Caravan.Policy ~., data = caravan_train, method="class", control = rpart.control(minsplit =1,minbucket=15, cp=0))

predict_tree <- predict(caravan_tree, type="class", newdata = caravan_test)
confusionMatrix(predict_tree, caravan_target$Caravan.Policy, positive = "1")


## ------------------------------------------------------------------------
#Classification Trees undersampled
caravan_tree_under <- rpart(Caravan.Policy ~., data = caravan_train_under, method="class", control = rpart.control(minsplit =1,minbucket=15, cp=0))

predict_tree_under <- predict(caravan_tree_under, type="class", newdata = caravan_test)
confusionMatrix(predict_tree_under, caravan_target$Caravan.Policy, positive = "1")


## ------------------------------------------------------------------------
#Classification Trees oversampled
caravan_tree_over <- rpart(Caravan.Policy ~., data = caravan_train_over, method="class", control = rpart.control(minsplit =1,minbucket=15, cp=0))

predict_tree_over <- predict(caravan_tree_over, type="class", newdata = caravan_test)
confusionMatrix(predict_tree_over, caravan_target$Caravan.Policy, positive = "1")



## ------------------------------------------------------------------------
#Bagging
caravan_bag <- randomForest(as.factor(Caravan.Policy) ~., data = caravan_train, mtry = 57, importance = TRUE)

predict_bagging <- predict(caravan_bag, caravan_test)

confusionMatrix(predict_bagging, caravan_target$Caravan.Policy, positive = "1")



## ------------------------------------------------------------------------
#Bagging Undersampled
caravan_bag_under <- randomForest(as.factor(Caravan.Policy) ~., data = caravan_train_under, mtry = 57, importance = TRUE)

predict_bagging_under <- predict(caravan_bag_under, caravan_test)

confusionMatrix(predict_bagging_under, caravan_target$Caravan.Policy, positive = "1")



## ------------------------------------------------------------------------
#Bagging Oversampled
caravan_bag_over <- randomForest(as.factor(Caravan.Policy) ~., data = caravan_train_over, mtry = 57, importance = TRUE)

predict_bagging_over <- predict(caravan_bag_over, caravan_test)

confusionMatrix(predict_bagging_over, caravan_target$Caravan.Policy, positive = "1")



## ------------------------------------------------------------------------
#Random Forests
caravan_rf <- randomForest(as.factor(Caravan.Policy) ~., data = caravan_train, mtry = 7, importance = TRUE)

predict_rf <- predict(caravan_rf, caravan_test)

confusionMatrix(predict_rf, caravan_target$Caravan.Policy, positive = "1")



## ------------------------------------------------------------------------
#Random Forests unsersampled
caravan_rf_under <- randomForest(as.factor(Caravan.Policy) ~., data = caravan_train_under, mtry = 7, importance = TRUE)

predict_rf_under <- predict(caravan_rf_under, caravan_test)

confusionMatrix(predict_rf_under, caravan_target$Caravan.Policy, positive = "1")



## ------------------------------------------------------------------------
#Random Forest oversampled
caravan_rf_over <- randomForest(as.factor(Caravan.Policy) ~., data = caravan_train_over, mtry = 7, importance = TRUE)

predict_rf_over <- predict(caravan_rf_over, caravan_test)

confusionMatrix(predict_rf_over, caravan_target$Caravan.Policy, positive = "1")



## ------------------------------------------------------------------------
#Boosting
caravan_boost <- gbm(Caravan.Policy ~., data = caravan_train, distribution = "bernoulli", n.trees = 5000, interaction.depth = 10, verbose = FALSE)

p_boosting <- predict.gbm(caravan_boost, newdata = caravan_test, n.trees = 5000, type = "response")
predict_boosting <- ifelse(p_boosting > 0.5, 1, 0)
predict_boosting <- as.factor(predict_boosting)

confusionMatrix(predict_boosting, caravan_target$Caravan.Policy, positive = "1")



## ------------------------------------------------------------------------
#Boosting unsersampling
caravan_boost_under <- gbm(Caravan.Policy ~., data = caravan_train_under, distribution = "bernoulli", n.trees = 5000, interaction.depth = 10, verbose = FALSE)

p_boosting_under <- predict.gbm(caravan_boost_under, newdata = caravan_test, n.trees = 5000, type = "response")
predict_boosting_under <- ifelse(p_boosting_under > 0.5, 1, 0)
predict_boosting_under <- as.factor(predict_boosting_under)

confusionMatrix(predict_boosting_under, caravan_target$Caravan.Policy, positive = "1")



## ------------------------------------------------------------------------
#Boosting oversampling
caravan_boost_over <- gbm(Caravan.Policy ~., data = caravan_train_over, distribution = "bernoulli", n.trees = 5000, interaction.depth = 10, verbose = FALSE)

p_boosting_over <- predict.gbm(caravan_boost_over, newdata = caravan_test, n.trees = 5000, type = "response")
predict_boosting_over <- ifelse(p_boosting_over > 0.5, 1, 0)
predict_boosting_over <- as.factor(predict_boosting_over)

confusionMatrix(predict_boosting_over, caravan_target$Caravan.Policy, positive = "1")



## ------------------------------------------------------------------------
#Association Rules
column_names <- colnames(caravan_train)
dummy_column <- dummy_cols(caravan_train, select_columns = column_names)

`%ni%` <- Negate(`%in%`)
caravan_ar <- subset(dummy_column, select = names(dummy_column) %ni% column_names)
caravan_sparse <- as.matrix(caravan_ar)
caravan_transactions <- as(caravan_sparse, "transactions")

caravan_rules <- apriori(data = caravan_transactions , parameter = list( supp = 0.001 , conf = 0.7) , appearance = list(default = "lhs" , rhs = "Caravan.Policy_1") )
caravan_rules <- sort(caravan_rules,decreasing=TRUE,by="lift")

inspect(caravan_rules[1:15])
plot(caravan_rules,cex = 0.5, jitter = 0)


