
rm(list=ls());

set.seed(2104);

library(caret);
library(ggplot2);

#reading the data

estimdata <- read.csv("C:/Alex/AulasWEB/PracticalMachineLearning/R_programs/pml-training.csv");
testdata <- read.csv("C:/Alex/AulasWEB/PracticalMachineLearning/R_programs/pml-testing.csv");

estimdata <- as.data.frame(estimdata);
testdata <- as.data.frame(testdata);

dim(estimdata);
names(estimdata);

names(testdata);
dim(testdata);

table(estimdata$classe);
is.factor(estimdata$classe);

estimdata <- subset(estimdata, select=-c(X, user_name, new_window));
names(estimdata);

testdata <- subset(testdata, select=-c(X, user_name, new_window));
names(testdata);

#excluding variables with missing data 

indicator_numeric_columns <- sapply(estimdata, is.numeric);

estimdata_num <- estimdata[,indicator_numeric_columns];

names_num_columns_with_na <- names(estimdata_num[is.na(colMeans(estimdata_num))]);

estimdata_valid <- estimdata_num[,!(names(estimdata_num) %in% names_num_columns_with_na)];

testdata_num <- testdata[, indicator_numeric_columns];
testdata_valid <- testdata_num[,!(names(testdata_num) %in% names_num_columns_with_na)];

classe = estimdata$classe;
estimdata_valid = data.frame(classe, estimdata_valid);

dim(estimdata_valid); dim(testdata_valid);

#creating samples for training, testing and evaluating the final model selected (60%, 20% and 20% in each sample)

inTrain <- createDataPartition(y = estimdata_valid$classe, p=0.80, list=FALSE);

training0 <- estimdata_valid[inTrain,];
evaluating <- estimdata_valid[-inTrain,];

inTrain1 <- createDataPartition(y = training0$classe, p = 0.75, list=FALSE);

training <- training0[inTrain1,];
testing <- training0[-inTrain1,];

dim(training); dim(testing); dim(evaluating); dim(estimdata_valid);
teste_dimensions <- dim(training)[1] + dim(testing)[1] + dim(evaluating)[1] - dim(estimdata_valid)[1];
percentagem_training <- dim(training)[1] / dim(estimdata_valid)[1];
percentagem_testing <- dim(testing)[1] / dim(estimdata_valid)[1];
percentagem_evaluating <- dim(evaluating)[1] / dim(estimdata_valid)[1];

teste_dimensions
percentagem_training
percentagem_testing
percentagem_evaluating

#creating principal components for data compression

preObj <- preProcess(training[,-1], method = "pca", thresh=0.95);
preObj

training_pca <- data.frame(classe = training$classe, predict(preObj, training[,-1]));
testing_pca <- data.frame(classe = testing$classe, predict(preObj, testing[,-1]));

#linear discriminant analysis

set.seed(2104);
modFit1 <- train(classe ~ ., method="lda", data=training);
preds <- predict(modFit1, newdata=testing);
confusionMatrix(preds, testing$classe)

#linear discriminant analysis com pca

set.seed(2104);
modFit2 <- train(classe ~ ., method="lda", data=training_pca);
preds <- predict(modFit2, newdata=testing_pca);
confusionMatrix(preds, testing_pca$classe)

#classification trees

set.seed(2104);
modFit3 <- train(classe ~ ., method="rpart", data=training);
preds <- predict(modFit3, newdata=testing);
confusionMatrix(preds, testing$classe)

#classification trees com pca

set.seed(2104);
modFit4 <- train(classe ~ ., method="rpart", data=training_pca);
preds <- predict(modFit4, newdata=testing_pca);
confusionMatrix(preds, testing_pca$classe)

#boosting 

#set.seed(2104);
#modFit5 <- train(classe ~ ., method="gbm", data=training, verbose=FALSE);
#preds <- predict(modFit5, newdata=testing);
#confusionMatrix(preds, testing$classe)

#boosting com pca

set.seed(2104);
modFit6 <- train(classe ~ ., method="gbm", data=training_pca, verbose=FALSE);
preds <- predict(modFit6, newdata=testing_pca);
confusionMatrix(preds, testing_pca$classe)

#random forests com pca

#set.seed(2104);
#modFit6 <- train(classe ~ ., method="rf", data=training_pca, prox=TRUE);
#preds <- predict(modFit6, newdata=testing_pca);
#confusionMatrix(preds, testing_pca$classe)

#bagging with lda com pca

#set.seed(2104);
#modFit7 <- train(classe ~ .,  
#                method = "bag", 
#                B = 10, 
#                data = training_pca,
#                bagControl = bagControl(fit = ldaBag$fit,
#                                        predict = ldaBag$pred,
#                                        aggregate = ldaBag$aggregate),
#                tuneGrid = data.frame(vars = c((1:10)*10 , ncol(training_pca))));
#preds <- predict(modFit7, newdata=testing_pca);
#confusionMatrix(preds, testing_pca$classe)

#regularized discriminant analysis com pca

#set.seed(2104);
#modFit8 <- train(classe ~ ., method="rda", data=training_pca);
#preds <- predict(modFit8, newdata=testing_pca);
#confusionMatrix(preds, testing_pca$classe)

#-------------------------------------------------------------------------------------------------
#training for testing + training sample the best model, to evaluate with evaluation sample
#-------------------------------------------------------------------------------------------------

training_eval <- estimdata_valid[inTrain,];
percentagem_training_eval <- dim(training_eval)[1] / dim(estimdata_valid)[1];
percentagem_training_eval

preObj_eval <- preProcess(training_eval[,-1], method = "pca", thresh=0.95);
preObj_eval

training_eval_pca <- data.frame(classe = training_eval$classe, predict(preObj_eval, training_eval[,-1]));
evaluating_pca <- data.frame(classe = evaluating$classe, predict(preObj_eval, evaluating[,-1]));

#evaluating the out-of-sample performance for the chosen model

set.seed(2104);
modFit_eval <- train(classe ~ ., method="gbm", data=training_eval_pca, verbose=FALSE);
preds <- predict(modFit_eval, newdata=evaluating_pca);
confusionMatrix(preds, evaluating_pca$classe)

#------------------------------------------------------------------------------------------------
#finally, training the final model with all the data to be used with the provided testing data
#------------------------------------------------------------------------------------------------

#preObj <- preProcess(estimdata_valid[,-1], method = "pca", thresh=0.95);
#preObj

#training_final_pca <- data.frame(classe = estimdata_valid$classe, predict(preObj, estimdata_valid[,-1]));

#set.seed(2104);
#modFit_final <- train(classe ~ ., method="gbm", data=training_final_pca, verbose=FALSE);

prediction_final_pca <- predict(preObj_eval, testdata_valid);
preds <- predict(modFit_eval, newdata=prediction_final_pca);

prediction_submitted_data <- predict(modFit_eval, newdata=prediction_final_pca);



