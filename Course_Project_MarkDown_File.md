Course: Practical Machine Learning
Prediction Study Design - Alexandre Xavier Ywata de Carvalho
========================================================

1. Data Preparation for the Prediction Study Design

To predict the outcome variable, I first imported the training data and the testing data. I extracted some summary statistics and noticed that, although the table is quite complex, in terms of possible predictive variables, some of these variables had several rows with missing data. I then decided to keep and use only the variables with no missing data for all rows. All the operations of columns extraction was done for both training and testing data.

2. Selecting Training, Testing and Evaluation Samples

I then focused only on the training data set. For this dataset, I separated the data into three subsets. The first dataset was used for training only; the second dataset was used for testing. The third one was used for validation of the final model. The training dataset had 60% of the observations; the testing and the validation datasets had each one of them 20% the observations in the original imported training table. Therefore, I had the mix 60%, 20%, 20% for the size of the training, testing and validation datasets. I reinforce that I am using now only the predicting variables with no missing data.

3. Choosing the Model

For the training sub dataset (60% of the original training imported dataset), I initially applied a principal component analysis to extract components explaining up to 95% of the total linear variation on the valid predictors (predictors with no missing data). Specifying the rotation for the principal components was done using exclusively the training sub dataset. I then applied the same rotation to the testing sub dataset (20% of the original training imported dataset). Using the training sub dataset for training and the testing sub dataset for evaluating the out-of-sample prediction error, I then tried several different models. When possible, I tried models with principal components compression or without it. The models tried were:

    1.  Linear discriminant analysis
    2.	Linear discriminant analysis to the principal components 
    3.	Classification trees
    4.	Classification trees to the principal components
    5.	Boosting 
    6.	Boosting to the principal components
    7.	Random forests to the principal components
    8.	Bagging with LDA for the principal components
    9.	Regularized discriminant analysis

Models 5, 7, 8 and 9 were not possible to train in the end, due to computer memory issues. I emphasize that each model was trained using only the training sub dataset (60% of the original training imported dataset). I used the testing sub dataset (20% of the original training imported dataset) to calculated out-of-sample prediction errors, for each model, and I found out that the best model, in terms of out-of-sample prediction, was the boosting applied to the principal components. 

4. Refining the Model

After selecting the best model, using 60% of the original sample for training, and 20% of the original sample for testing the out-of-sample performance for each tried model, I re-estimated the best model (boosting with PCA) to the 80% sub dataset, consisting of the training sub dataset plus the testing sub dataset. For this new trained model, I evaluated the out-of-sample performance using the validation sub dataset (the other 20% of the original sample, left out of the new training). The performance of the model in the validation sub dataset was very good: I obtained an accuracy of 83.43%. I was very careful in applying the principal components rotation, based only on the data used for training the model.

5. Using the Final Model

Finally, the last step would be to use the selected model (boosting with PCA) and trained it one more time, now using the whole training original data. In that way, I would be able to use all information in the provided training database. However, when trying to train this new model (with 100% of the original training data), I had again computer memory issues. I then decided to use the previous model, estimated with 80% of the original provided training sample. 

I then applied the best available trained model to the provided testing dataset. Once again, I was very careful in applying the principal components rotation, based only on the data effectively used for training the model. The final predicted categories were:

      Observation	Predicted Classe
      1		        B
      2		        A
      3		        C
      4		        A
      5		        A
      6		        E
      7		        E
      8		        D
      9		        A
      10		      A
      11		      A
      12		      C
      13		      B
      14		      A
      15		      E
      16		      E
      17		      A
      18		      B
      19		      A
      20		      B


```r
rm(list=ls());

set.seed(2104);

library(caret);
```

```
## Warning: package 'caret' was built under R version 3.0.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.0.3
```

```r
library(ggplot2);

#reading the data

estimdata <- read.csv("C:/Alex/AulasWEB/PracticalMachineLearning/R_programs/pml-training.csv");
testdata <- read.csv("C:/Alex/AulasWEB/PracticalMachineLearning/R_programs/pml-testing.csv");

estimdata <- as.data.frame(estimdata);
testdata <- as.data.frame(testdata);

dim(estimdata);
```

```
## [1] 19622   160
```

```r
# names(estimdata);

# names(testdata);
dim(testdata);
```

```
## [1]  20 160
```

```r
table(estimdata$classe);
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

```r
is.factor(estimdata$classe);
```

```
## [1] TRUE
```

```r
estimdata <- subset(estimdata, select=-c(X, user_name, new_window));
names(estimdata);
```

```
##   [1] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
##   [3] "cvtd_timestamp"           "num_window"              
##   [5] "roll_belt"                "pitch_belt"              
##   [7] "yaw_belt"                 "total_accel_belt"        
##   [9] "kurtosis_roll_belt"       "kurtosis_picth_belt"     
##  [11] "kurtosis_yaw_belt"        "skewness_roll_belt"      
##  [13] "skewness_roll_belt.1"     "skewness_yaw_belt"       
##  [15] "max_roll_belt"            "max_picth_belt"          
##  [17] "max_yaw_belt"             "min_roll_belt"           
##  [19] "min_pitch_belt"           "min_yaw_belt"            
##  [21] "amplitude_roll_belt"      "amplitude_pitch_belt"    
##  [23] "amplitude_yaw_belt"       "var_total_accel_belt"    
##  [25] "avg_roll_belt"            "stddev_roll_belt"        
##  [27] "var_roll_belt"            "avg_pitch_belt"          
##  [29] "stddev_pitch_belt"        "var_pitch_belt"          
##  [31] "avg_yaw_belt"             "stddev_yaw_belt"         
##  [33] "var_yaw_belt"             "gyros_belt_x"            
##  [35] "gyros_belt_y"             "gyros_belt_z"            
##  [37] "accel_belt_x"             "accel_belt_y"            
##  [39] "accel_belt_z"             "magnet_belt_x"           
##  [41] "magnet_belt_y"            "magnet_belt_z"           
##  [43] "roll_arm"                 "pitch_arm"               
##  [45] "yaw_arm"                  "total_accel_arm"         
##  [47] "var_accel_arm"            "avg_roll_arm"            
##  [49] "stddev_roll_arm"          "var_roll_arm"            
##  [51] "avg_pitch_arm"            "stddev_pitch_arm"        
##  [53] "var_pitch_arm"            "avg_yaw_arm"             
##  [55] "stddev_yaw_arm"           "var_yaw_arm"             
##  [57] "gyros_arm_x"              "gyros_arm_y"             
##  [59] "gyros_arm_z"              "accel_arm_x"             
##  [61] "accel_arm_y"              "accel_arm_z"             
##  [63] "magnet_arm_x"             "magnet_arm_y"            
##  [65] "magnet_arm_z"             "kurtosis_roll_arm"       
##  [67] "kurtosis_picth_arm"       "kurtosis_yaw_arm"        
##  [69] "skewness_roll_arm"        "skewness_pitch_arm"      
##  [71] "skewness_yaw_arm"         "max_roll_arm"            
##  [73] "max_picth_arm"            "max_yaw_arm"             
##  [75] "min_roll_arm"             "min_pitch_arm"           
##  [77] "min_yaw_arm"              "amplitude_roll_arm"      
##  [79] "amplitude_pitch_arm"      "amplitude_yaw_arm"       
##  [81] "roll_dumbbell"            "pitch_dumbbell"          
##  [83] "yaw_dumbbell"             "kurtosis_roll_dumbbell"  
##  [85] "kurtosis_picth_dumbbell"  "kurtosis_yaw_dumbbell"   
##  [87] "skewness_roll_dumbbell"   "skewness_pitch_dumbbell" 
##  [89] "skewness_yaw_dumbbell"    "max_roll_dumbbell"       
##  [91] "max_picth_dumbbell"       "max_yaw_dumbbell"        
##  [93] "min_roll_dumbbell"        "min_pitch_dumbbell"      
##  [95] "min_yaw_dumbbell"         "amplitude_roll_dumbbell" 
##  [97] "amplitude_pitch_dumbbell" "amplitude_yaw_dumbbell"  
##  [99] "total_accel_dumbbell"     "var_accel_dumbbell"      
## [101] "avg_roll_dumbbell"        "stddev_roll_dumbbell"    
## [103] "var_roll_dumbbell"        "avg_pitch_dumbbell"      
## [105] "stddev_pitch_dumbbell"    "var_pitch_dumbbell"      
## [107] "avg_yaw_dumbbell"         "stddev_yaw_dumbbell"     
## [109] "var_yaw_dumbbell"         "gyros_dumbbell_x"        
## [111] "gyros_dumbbell_y"         "gyros_dumbbell_z"        
## [113] "accel_dumbbell_x"         "accel_dumbbell_y"        
## [115] "accel_dumbbell_z"         "magnet_dumbbell_x"       
## [117] "magnet_dumbbell_y"        "magnet_dumbbell_z"       
## [119] "roll_forearm"             "pitch_forearm"           
## [121] "yaw_forearm"              "kurtosis_roll_forearm"   
## [123] "kurtosis_picth_forearm"   "kurtosis_yaw_forearm"    
## [125] "skewness_roll_forearm"    "skewness_pitch_forearm"  
## [127] "skewness_yaw_forearm"     "max_roll_forearm"        
## [129] "max_picth_forearm"        "max_yaw_forearm"         
## [131] "min_roll_forearm"         "min_pitch_forearm"       
## [133] "min_yaw_forearm"          "amplitude_roll_forearm"  
## [135] "amplitude_pitch_forearm"  "amplitude_yaw_forearm"   
## [137] "total_accel_forearm"      "var_accel_forearm"       
## [139] "avg_roll_forearm"         "stddev_roll_forearm"     
## [141] "var_roll_forearm"         "avg_pitch_forearm"       
## [143] "stddev_pitch_forearm"     "var_pitch_forearm"       
## [145] "avg_yaw_forearm"          "stddev_yaw_forearm"      
## [147] "var_yaw_forearm"          "gyros_forearm_x"         
## [149] "gyros_forearm_y"          "gyros_forearm_z"         
## [151] "accel_forearm_x"          "accel_forearm_y"         
## [153] "accel_forearm_z"          "magnet_forearm_x"        
## [155] "magnet_forearm_y"         "magnet_forearm_z"        
## [157] "classe"
```

```r
testdata <- subset(testdata, select=-c(X, user_name, new_window));
names(testdata);
```

```
##   [1] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
##   [3] "cvtd_timestamp"           "num_window"              
##   [5] "roll_belt"                "pitch_belt"              
##   [7] "yaw_belt"                 "total_accel_belt"        
##   [9] "kurtosis_roll_belt"       "kurtosis_picth_belt"     
##  [11] "kurtosis_yaw_belt"        "skewness_roll_belt"      
##  [13] "skewness_roll_belt.1"     "skewness_yaw_belt"       
##  [15] "max_roll_belt"            "max_picth_belt"          
##  [17] "max_yaw_belt"             "min_roll_belt"           
##  [19] "min_pitch_belt"           "min_yaw_belt"            
##  [21] "amplitude_roll_belt"      "amplitude_pitch_belt"    
##  [23] "amplitude_yaw_belt"       "var_total_accel_belt"    
##  [25] "avg_roll_belt"            "stddev_roll_belt"        
##  [27] "var_roll_belt"            "avg_pitch_belt"          
##  [29] "stddev_pitch_belt"        "var_pitch_belt"          
##  [31] "avg_yaw_belt"             "stddev_yaw_belt"         
##  [33] "var_yaw_belt"             "gyros_belt_x"            
##  [35] "gyros_belt_y"             "gyros_belt_z"            
##  [37] "accel_belt_x"             "accel_belt_y"            
##  [39] "accel_belt_z"             "magnet_belt_x"           
##  [41] "magnet_belt_y"            "magnet_belt_z"           
##  [43] "roll_arm"                 "pitch_arm"               
##  [45] "yaw_arm"                  "total_accel_arm"         
##  [47] "var_accel_arm"            "avg_roll_arm"            
##  [49] "stddev_roll_arm"          "var_roll_arm"            
##  [51] "avg_pitch_arm"            "stddev_pitch_arm"        
##  [53] "var_pitch_arm"            "avg_yaw_arm"             
##  [55] "stddev_yaw_arm"           "var_yaw_arm"             
##  [57] "gyros_arm_x"              "gyros_arm_y"             
##  [59] "gyros_arm_z"              "accel_arm_x"             
##  [61] "accel_arm_y"              "accel_arm_z"             
##  [63] "magnet_arm_x"             "magnet_arm_y"            
##  [65] "magnet_arm_z"             "kurtosis_roll_arm"       
##  [67] "kurtosis_picth_arm"       "kurtosis_yaw_arm"        
##  [69] "skewness_roll_arm"        "skewness_pitch_arm"      
##  [71] "skewness_yaw_arm"         "max_roll_arm"            
##  [73] "max_picth_arm"            "max_yaw_arm"             
##  [75] "min_roll_arm"             "min_pitch_arm"           
##  [77] "min_yaw_arm"              "amplitude_roll_arm"      
##  [79] "amplitude_pitch_arm"      "amplitude_yaw_arm"       
##  [81] "roll_dumbbell"            "pitch_dumbbell"          
##  [83] "yaw_dumbbell"             "kurtosis_roll_dumbbell"  
##  [85] "kurtosis_picth_dumbbell"  "kurtosis_yaw_dumbbell"   
##  [87] "skewness_roll_dumbbell"   "skewness_pitch_dumbbell" 
##  [89] "skewness_yaw_dumbbell"    "max_roll_dumbbell"       
##  [91] "max_picth_dumbbell"       "max_yaw_dumbbell"        
##  [93] "min_roll_dumbbell"        "min_pitch_dumbbell"      
##  [95] "min_yaw_dumbbell"         "amplitude_roll_dumbbell" 
##  [97] "amplitude_pitch_dumbbell" "amplitude_yaw_dumbbell"  
##  [99] "total_accel_dumbbell"     "var_accel_dumbbell"      
## [101] "avg_roll_dumbbell"        "stddev_roll_dumbbell"    
## [103] "var_roll_dumbbell"        "avg_pitch_dumbbell"      
## [105] "stddev_pitch_dumbbell"    "var_pitch_dumbbell"      
## [107] "avg_yaw_dumbbell"         "stddev_yaw_dumbbell"     
## [109] "var_yaw_dumbbell"         "gyros_dumbbell_x"        
## [111] "gyros_dumbbell_y"         "gyros_dumbbell_z"        
## [113] "accel_dumbbell_x"         "accel_dumbbell_y"        
## [115] "accel_dumbbell_z"         "magnet_dumbbell_x"       
## [117] "magnet_dumbbell_y"        "magnet_dumbbell_z"       
## [119] "roll_forearm"             "pitch_forearm"           
## [121] "yaw_forearm"              "kurtosis_roll_forearm"   
## [123] "kurtosis_picth_forearm"   "kurtosis_yaw_forearm"    
## [125] "skewness_roll_forearm"    "skewness_pitch_forearm"  
## [127] "skewness_yaw_forearm"     "max_roll_forearm"        
## [129] "max_picth_forearm"        "max_yaw_forearm"         
## [131] "min_roll_forearm"         "min_pitch_forearm"       
## [133] "min_yaw_forearm"          "amplitude_roll_forearm"  
## [135] "amplitude_pitch_forearm"  "amplitude_yaw_forearm"   
## [137] "total_accel_forearm"      "var_accel_forearm"       
## [139] "avg_roll_forearm"         "stddev_roll_forearm"     
## [141] "var_roll_forearm"         "avg_pitch_forearm"       
## [143] "stddev_pitch_forearm"     "var_pitch_forearm"       
## [145] "avg_yaw_forearm"          "stddev_yaw_forearm"      
## [147] "var_yaw_forearm"          "gyros_forearm_x"         
## [149] "gyros_forearm_y"          "gyros_forearm_z"         
## [151] "accel_forearm_x"          "accel_forearm_y"         
## [153] "accel_forearm_z"          "magnet_forearm_x"        
## [155] "magnet_forearm_y"         "magnet_forearm_z"        
## [157] "problem_id"
```

```r
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
```

```
## [1] 19622    56
```

```
## [1] 20 55
```

```r
#creating samples for training, testing and evaluating the final model selected (60%, 20% and 20% in each sample)

inTrain <- createDataPartition(y = estimdata_valid$classe, p=0.80, list=FALSE);

training0 <- estimdata_valid[inTrain,];
evaluating <- estimdata_valid[-inTrain,];

inTrain1 <- createDataPartition(y = training0$classe, p = 0.75, list=FALSE);

training <- training0[inTrain1,];
testing <- training0[-inTrain1,];

dim(training); dim(testing); dim(evaluating); dim(estimdata_valid);
```

```
## [1] 11776    56
```

```
## [1] 3923   56
```

```
## [1] 3923   56
```

```
## [1] 19622    56
```

```r
teste_dimensions <- dim(training)[1] + dim(testing)[1] + dim(evaluating)[1] - dim(estimdata_valid)[1];
percentagem_training <- dim(training)[1] / dim(estimdata_valid)[1];
percentagem_testing <- dim(testing)[1] / dim(estimdata_valid)[1];
percentagem_evaluating <- dim(evaluating)[1] / dim(estimdata_valid)[1];

teste_dimensions
```

```
## [1] 0
```

```r
percentagem_training
```

```
## [1] 0.6001
```

```r
percentagem_testing
```

```
## [1] 0.1999
```

```r
percentagem_evaluating
```

```
## [1] 0.1999
```

```r
#creating principal components for data compression

preObj <- preProcess(training[,-1], method = "pca", thresh=0.95);
preObj
```

```
## 
## Call:
## preProcess.default(x = training[, -1], method = "pca", thresh = 0.95)
## 
## Created from 11776 samples and 55 variables
## Pre-processing: principal component signal extraction, scaled, centered 
## 
## PCA needed 28 components to capture 95 percent of the variance
```

```r
training_pca <- data.frame(classe = training$classe, predict(preObj, training[,-1]));
testing_pca <- data.frame(classe = testing$classe, predict(preObj, testing[,-1]));

#linear discriminant analysis

set.seed(2104);
modFit1 <- train(classe ~ ., method="lda", data=training);
```

```
## Loading required package: MASS
```

```
## Warning: package 'e1071' was built under R version 3.0.3
```

```r
preds <- predict(modFit1, newdata=testing);
confusionMatrix(preds, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 946 121  73  35  46
##          B  32 494  59  29 100
##          C  54  90 442  80  50
##          D  79  30  89 478  77
##          E   5  24  21  21 448
## 
## Overall Statistics
##                                        
##                Accuracy : 0.716        
##                  95% CI : (0.701, 0.73)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.64         
##  Mcnemar's Test P-Value : <2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.848    0.651    0.646    0.743    0.621
## Specificity             0.902    0.930    0.915    0.916    0.978
## Pos Pred Value          0.775    0.692    0.617    0.635    0.863
## Neg Pred Value          0.937    0.917    0.925    0.948    0.920
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.241    0.126    0.113    0.122    0.114
## Detection Prevalence    0.311    0.182    0.183    0.192    0.132
## Balanced Accuracy       0.875    0.791    0.781    0.830    0.800
```

```r
#linear discriminant analysis com pca

set.seed(2104);
modFit2 <- train(classe ~ ., method="lda", data=training_pca);
preds <- predict(modFit2, newdata=testing_pca);
confusionMatrix(preds, testing_pca$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 736 172 180  39  79
##          B  96 318  60  72 126
##          C 105 141 355 120  91
##          D 126  89  58 340  85
##          E  53  39  31  72 340
## 
## Overall Statistics
##                                         
##                Accuracy : 0.533         
##                  95% CI : (0.517, 0.548)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.408         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.659   0.4190   0.5190   0.5288   0.4716
## Specificity             0.833   0.8881   0.8589   0.8909   0.9391
## Pos Pred Value          0.610   0.4732   0.4372   0.4871   0.6355
## Neg Pred Value          0.860   0.8643   0.8942   0.9060   0.8875
## Prevalence              0.284   0.1935   0.1744   0.1639   0.1838
## Detection Rate          0.188   0.0811   0.0905   0.0867   0.0867
## Detection Prevalence    0.307   0.1713   0.2070   0.1779   0.1364
## Balanced Accuracy       0.746   0.6535   0.6890   0.7098   0.7053
```

```r
#classification trees

set.seed(2104);
modFit3 <- train(classe ~ ., method="rpart", data=training);
```

```
## Loading required package: rpart
```

```r
preds <- predict(modFit3, newdata=testing);
confusionMatrix(preds, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1011  322  320  266   70
##          B   14  244   23  102   53
##          C   86  193  341  240  146
##          D    0    0    0    0    0
##          E    5    0    0   35  452
## 
## Overall Statistics
##                                         
##                Accuracy : 0.522         
##                  95% CI : (0.506, 0.538)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.376         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.906   0.3215   0.4985    0.000    0.627
## Specificity             0.652   0.9393   0.7947    1.000    0.988
## Pos Pred Value          0.508   0.5596   0.3390      NaN    0.919
## Neg Pred Value          0.946   0.8523   0.8824    0.836    0.922
## Prevalence              0.284   0.1935   0.1744    0.164    0.184
## Detection Rate          0.258   0.0622   0.0869    0.000    0.115
## Detection Prevalence    0.507   0.1111   0.2564    0.000    0.125
## Balanced Accuracy       0.779   0.6304   0.6466    0.500    0.807
```

```r
#classification trees com pca

set.seed(2104);
modFit4 <- train(classe ~ ., method="rpart", data=training_pca);
preds <- predict(modFit4, newdata=testing_pca);
confusionMatrix(preds, testing_pca$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 740 170 278 154 151
##          B 148 293  75  69 116
##          C 136 121 261 108  68
##          D  65 122  63 212  80
##          E  27  53   7 100 306
## 
## Overall Statistics
##                                         
##                Accuracy : 0.462         
##                  95% CI : (0.446, 0.478)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.311         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.663   0.3860   0.3816    0.330    0.424
## Specificity             0.732   0.8710   0.8663    0.899    0.942
## Pos Pred Value          0.496   0.4180   0.3761    0.391    0.621
## Neg Pred Value          0.845   0.8554   0.8690    0.873    0.879
## Prevalence              0.284   0.1935   0.1744    0.164    0.184
## Detection Rate          0.189   0.0747   0.0665    0.054    0.078
## Detection Prevalence    0.381   0.1787   0.1769    0.138    0.126
## Balanced Accuracy       0.697   0.6285   0.6239    0.615    0.683
```

```r
#boosting 

#set.seed(2104);
#modFit5 <- train(classe ~ ., method="gbm", data=training, verbose=FALSE);
#preds <- predict(modFit5, newdata=testing);
#confusionMatrix(preds, testing$classe)

#boosting com pca

set.seed(2104);
modFit6 <- train(classe ~ ., method="gbm", data=training_pca, verbose=FALSE);
```

```
## Loading required package: gbm
```

```
## Warning: package 'gbm' was built under R version 3.0.3
```

```
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
## Loading required package: plyr
```

```r
preds <- predict(modFit6, newdata=testing_pca);
confusionMatrix(preds, testing_pca$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 996  94  49  20  34
##          B  33 586  49  21  41
##          C  30  36 562  80  30
##          D  48  23  16 498  41
##          E   9  20   8  24 575
## 
## Overall Statistics
##                                         
##                Accuracy : 0.82          
##                  95% CI : (0.808, 0.832)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.772         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.892    0.772    0.822    0.774    0.798
## Specificity             0.930    0.954    0.946    0.961    0.981
## Pos Pred Value          0.835    0.803    0.762    0.796    0.904
## Neg Pred Value          0.956    0.946    0.962    0.956    0.956
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.254    0.149    0.143    0.127    0.147
## Detection Prevalence    0.304    0.186    0.188    0.160    0.162
## Balanced Accuracy       0.911    0.863    0.884    0.868    0.889
```

```r
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
```

```
## [1] 0.8001
```

```r
preObj_eval <- preProcess(training_eval[,-1], method = "pca", thresh=0.95);
preObj_eval
```

```
## 
## Call:
## preProcess.default(x = training_eval[, -1], method = "pca", thresh = 0.95)
## 
## Created from 15699 samples and 55 variables
## Pre-processing: principal component signal extraction, scaled, centered 
## 
## PCA needed 28 components to capture 95 percent of the variance
```

```r
training_eval_pca <- data.frame(classe = training_eval$classe, predict(preObj_eval, training_eval[,-1]));
evaluating_pca <- data.frame(classe = evaluating$classe, predict(preObj_eval, evaluating[,-1]));

#evaluating the out-of-sample performance for the chosen model

set.seed(2104);
modFit_eval <- train(classe ~ ., method="gbm", data=training_eval_pca, verbose=FALSE);
preds <- predict(modFit_eval, newdata=evaluating_pca);
confusionMatrix(preds, evaluating_pca$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 992  48  23  30  27
##          B  28 609  45  20  54
##          C  32  60 583  62  31
##          D  55  17  21 512  27
##          E   9  25  12  19 582
## 
## Overall Statistics
##                                         
##                Accuracy : 0.836         
##                  95% CI : (0.824, 0.847)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.792         
##  Mcnemar's Test P-Value : 2.43e-10      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.889    0.802    0.852    0.796    0.807
## Specificity             0.954    0.954    0.943    0.963    0.980
## Pos Pred Value          0.886    0.806    0.759    0.810    0.900
## Neg Pred Value          0.956    0.953    0.968    0.960    0.958
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.253    0.155    0.149    0.131    0.148
## Detection Prevalence    0.285    0.193    0.196    0.161    0.165
## Balanced Accuracy       0.922    0.878    0.898    0.880    0.893
```

```r
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
```



