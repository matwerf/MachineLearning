Title: Exercise prediction
========================================================
### Synopsis:
In this project, the goal was to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise. They performed barbell lifts correctly and incorrectly in 5 different ways described at http://groupware.les.inf.puc-rio.br/har. The training data set was divided into two groups 60% for training and 40% for cross validation, then the test data was run. It was found that the Random Forest Model provided the lowest error of the models tested on the reserved data, and resulted in the correct prediction of the 20 samples with less than a 0.01% out of sample error.

This report has 5 parts: Model Building, Cross Validation, Sample Error, Explaination and Prediction. 

The "classe" variable in the training set described the manner of exercise preformed and was one of 5 choices.


1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details. 
### Model Building:


```
## Warning: package 'caret' was built under R version 3.1.1
```

```
## Warning: package 'AppliedPredictiveModeling' was built under R version
## 3.1.1
```

```
## Warning: package 'rattle' was built under R version 3.1.1
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.3.0 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```
#Read in the file and unzip it



```r
#Divide data into test and training sets
exerdata <- read.csv("./data/trainingfile.csv")
inTrain <-createDataPartition(y=exerdata$classe, p=0.6, list = FALSE)
training <- exerdata[inTrain,]
testing <- exerdata[-inTrain,]
datatest <- read.csv("./data/testfile.csv")
```
The parameters were evaluated with the  following code, as well as a manual inspection. Those parameters that were not majority NA or empty were used for the training.

```r
##parameter evaluations

#as.data.frame( t(sapply(training, function(cl) list(means=mean(cl,na.rm=TRUE), sds=sd(cl,na.rm=TRUE), numNA=sum(is.na(cl)))) ))

#magnet_belt_x  magnet_belt_y	magnet_belt_z	roll_arm	pitch_arm	yaw_arm	total_accel_arm roll_dumbbell  pitch_dumbbell	yaw_dumbbell accel_arm_x  accel_arm_y	accel_arm_z	magnet_arm_x	magnet_arm_y	magnet_arm_z gyros_arm_x  gyros_arm_y	gyros_arm_z gyros_belt_x  gyros_belt_y	gyros_belt_z	accel_belt_x	accel_belt_y	accel_belt_z	magnet_belt_x	magnet_belt_y	magnet_belt_z	roll_arm	pitch_arm	yaw_arm	total_accel_arm X  user_name	raw_timestamp_part_1	raw_timestamp_part_2	cvtd_timestamp	new_window	num_window	roll_belt	pitch_belt	yaw_belt	total_accel_belt
```
The following methods were tested, the "rpart" method in the caret package with and without preprocessing. Then, both "lda" and "nb" methods were tried. After this, the "rf" method was tested in the train function in the caret package. The defalts for this program possibly "overfit" the data and were not needed 500 trees and 25 cv. 


```r
##tree - rpart
fit1 <- train(classe ~ roll_dumbbell  + pitch_dumbbell  + yaw_dumbbell + accel_arm_x  + accel_arm_y	+ accel_arm_z	+ magnet_arm_x	+ magnet_arm_y	+ magnet_arm_z + gyros_arm_x  + gyros_arm_y	+ gyros_arm_z + gyros_belt_x  +gyros_belt_y	+gyros_belt_z	+accel_belt_x	+accel_belt_y	+ accel_belt_z	+ magnet_belt_x	+ magnet_belt_y	+magnet_belt_z	+ roll_arm	+pitch_arm	+ yaw_arm	+ total_accel_arm	+ new_window	+num_window	+ roll_belt	+ pitch_belt	+ yaw_belt	+ total_accel_belt, data=training, method = "rpart") 
prefit1 <- predict(fit1,testing)
table(prefit1, testing$classe)
```

```
##        
## prefit1    A    B    C    D    E
##       A 1412  250  299  131   67
##       B  458  974  350  504  413
##       C  154  184  535   26   81
##       D  206  110  184  568    9
##       E    2    0    0   57  872
```

```r
fit1Acc <- sum(table(prefit1, testing$classe)/nrow(testing)*diag(5))
fit1Acc
```

```
## [1] 0.5558246
```

```r
#0.5652

##tree - rpart centered and scaled
fit2 <- train(classe ~ roll_dumbbell  + pitch_dumbbell  + yaw_dumbbell + accel_arm_x  + accel_arm_y  + accel_arm_z	+ magnet_arm_x	+ magnet_arm_y	+ magnet_arm_z + gyros_arm_x  + gyros_arm_y	+ gyros_arm_z + gyros_belt_x  +gyros_belt_y	+gyros_belt_z	+accel_belt_x	+accel_belt_y	+ accel_belt_z	+ magnet_belt_x	+ magnet_belt_y	+magnet_belt_z	+ roll_arm	+pitch_arm	+ yaw_arm	+ total_accel_arm	+ new_window	+num_window	+ roll_belt	+ pitch_belt	+ yaw_belt	+ total_accel_belt, data=training, method = "rpart", preProcess = c("center","scale")) 
prefit2 <- predict(fit2,testing)
table(prefit2, testing$classe)
```

```
##        
## prefit2    A    B    C    D    E
##       A 1412  250  299  131   67
##       B  458  974  350  504  413
##       C  154  184  535   26   81
##       D  206  110  184  568    9
##       E    2    0    0   57  872
```

```r
fit2Acc <- sum(table(prefit2, testing$classe)/nrow(testing)*diag(5))
fit2Acc
```

```
## [1] 0.5558246
```

```r
#0.5652

###lda
#fit8 <- train(classe ~ roll_dumbbell  + pitch_dumbbell  + yaw_dumbbell + accel_arm_x  + accel_arm_y  + accel_arm_z  + magnet_arm_x	+ magnet_arm_y	+ magnet_arm_z + gyros_arm_x  + gyros_arm_y	+ gyros_arm_z + gyros_belt_x  +gyros_belt_y	+gyros_belt_z	+accel_belt_x	+accel_belt_y	+ accel_belt_z	+ magnet_belt_x	+ magnet_belt_y	+magnet_belt_z	+ roll_arm	+pitch_arm	+ yaw_arm	+ total_accel_arm	+ new_window	+num_window	+ roll_belt	+ pitch_belt	+ yaw_belt	+ total_accel_belt, data=training, method = "lda", preProcess = c("center","scale")) 
#prefit8 <- predict(fit8,testing)
#table(prefit8, testing$classe)
#fit8Acc <- sum(table(prefit8, testing$classe)/nrow(testing)*diag(5))
#fit8Acc
#0.58

###nb
#fit9 <- train(classe ~ roll_dumbbell  + pitch_dumbbell  + yaw_dumbbell + accel_arm_x  + accel_arm_y  + accel_arm_z  + magnet_arm_x  + magnet_arm_y	+ magnet_arm_z + gyros_arm_x  + gyros_arm_y	+ gyros_arm_z + gyros_belt_x  +gyros_belt_y	+gyros_belt_z	+accel_belt_x	+accel_belt_y	+ accel_belt_z	+ magnet_belt_x	+ magnet_belt_y	+magnet_belt_z	+ roll_arm	+pitch_arm	+ yaw_arm	+ total_accel_arm	+ new_window	+num_window	+ roll_belt	+ pitch_belt	+ yaw_belt	+ total_accel_belt, data=training, method = "nb", preProcess = c("center","scale")) 
#prefit9 <- predict(fit9,testing)
#table(prefit9, testing$classe)
#fit9Acc <- sum(table(prefit9, testing$classe)/nrow(testing)*diag(5))
#fit9Acc

#rf
fit10 <- train(classe ~ roll_dumbbell  + pitch_dumbbell  + yaw_dumbbell + accel_arm_x  + accel_arm_y  + accel_arm_z  + magnet_arm_x  + magnet_arm_y  + magnet_arm_z + gyros_arm_x  + gyros_arm_y	+ gyros_arm_z + gyros_belt_x  +gyros_belt_y	+gyros_belt_z	+accel_belt_x	+accel_belt_y	+ accel_belt_z	+ magnet_belt_x	+ magnet_belt_y	+magnet_belt_z	+ roll_arm	+pitch_arm	+ yaw_arm	+ total_accel_arm	+ new_window	+num_window	+ roll_belt	+ pitch_belt	+ yaw_belt	+ total_accel_belt, data=training, method = "rf", preProcess = c("center","scale"), ntree=32) 
prefit10 <- predict(fit10,testing)
table(prefit10, testing$classe)
```

```
##         
## prefit10    A    B    C    D    E
##        A 2229    3    0    0    0
##        B    0 1509    6    3    1
##        C    0    6 1361    5    0
##        D    3    0    1 1277    0
##        E    0    0    0    1 1441
```

```r
fit10Acc<- sum(table(prefit10, testing$classe)/nrow(testing)*diag(5))
fit10Acc
```

```
## [1] 0.9963038
```

```r
#1

fit11 <- train(classe ~ roll_dumbbell  + pitch_dumbbell  + yaw_dumbbell + accel_arm_x  + accel_arm_y  + accel_arm_z  + magnet_arm_x  + magnet_arm_y  + magnet_arm_z + gyros_arm_x  + gyros_arm_y  + gyros_arm_z + gyros_belt_x  +gyros_belt_y	+gyros_belt_z	+accel_belt_x	+accel_belt_y	+ accel_belt_z	+ magnet_belt_x	+ magnet_belt_y	+magnet_belt_z	+ roll_arm	+pitch_arm	+ yaw_arm	+ total_accel_arm	+ new_window	+num_window	+ roll_belt	+ pitch_belt	+ yaw_belt	+ total_accel_belt, data=training, method = "rf", preProcess = c("center","scale"), ntree=16) 
prefit11 <- predict(fit10,testing)
table(prefit11, testing$classe)
```

```
##         
## prefit11    A    B    C    D    E
##        A 2230    4    0    0    0
##        B    0 1507    6    3    1
##        C    0    7 1361    5    0
##        D    2    0    1 1277    0
##        E    0    0    0    1 1441
```

```r
fit11Acc <- sum(table(prefit11, testing$classe)/nrow(testing)*diag(5))
fit11Acc
```

```
## [1] 0.9961764
```

```r
system.time({fit11 <- train(classe ~ roll_dumbbell  + pitch_dumbbell  + yaw_dumbbell + accel_arm_x  + accel_arm_y  + accel_arm_z  + magnet_arm_x  + magnet_arm_y  + magnet_arm_z + gyros_arm_x  + gyros_arm_y  + gyros_arm_z + gyros_belt_x  +gyros_belt_y    +gyros_belt_z  +accel_belt_x	+accel_belt_y	+ accel_belt_z	+ magnet_belt_x	+ magnet_belt_y	+magnet_belt_z	+ roll_arm	+pitch_arm	+ yaw_arm	+ total_accel_arm	+ new_window	+num_window	+ roll_belt	+ pitch_belt	+ yaw_belt	+ total_accel_belt, data=training, method = "rf", preProcess = c("center","scale"), ntree=16) })
```

```
##    user  system elapsed 
##  57.168   1.821  58.988
```

#The best model was then used to provide the best model for the prediction

```r
##section can be commented in to predict the date
##Applying the top prediction to datatest
#fit10test <- predict(fit10,datatest)
#pml_write_files = function(x){
#  n = length(x)
#  for(i in 1:n){
#    filename = paste0("problem_id_",i,".txt")
#    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
#  }
#}
#pml_write_files(fit10test)

#[1] B A B A A E D B A A B C B A E E A B B B
```
