library('caret'); library('rattle'); library('rpart.plot'); library('randomForest'); library('AppliedPredictiveModeling')

#download the training and test data. read into data objects
trainingURL <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
testURL <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
if (!file.exists('pml-training.csv')) download.file(trainingURL, 'pml-training.csv') else 'test data present'
if (!file.exists('pml-test.csv')) download.file(testURL, 'pml-test.csv') else 'training data present'

data <- read.csv('pml-training.csv')
colnames_data <- colnames(data)
test <- read.csv('pml-test.csv')
colnames_test <- colnames(test)

#Confirm both the training and testing data set column names are the same, (excluding the object in each)
equal <- all.equal(colnames_data[1:length(colnames_data)-1], colnames_test[1:length(colnames_test)-1])


if (equal) "Data sets match" else "Data sets don't match"
rm(trainingURL, testURL)


#Data Set Preparation.
# Count the number of non-NAs in each col.
nonNAs <- function(x) {
    as.vector(apply(x, 2, function(x) length(which(!is.na(x)))))
}
##build a vector of columns with missing data to drop
colcnts <- nonNAs(data)
drops <- c()
for (cnt in 1:length(colcnts)) {
    if (colcnts[cnt] < nrow(data)) {
        drops <- c(drops, colnames_data[cnt])
    }
}

# Drop NA data and the first 7 columns as they're unnecessary for predicting.
data <- data[,!(names(data) %in% drops)]
data <- data[,8:length(colnames(data))]

#Identify covariates that have near zero variablility for removal
nsv <- nearZeroVar(data, saveMetrics=TRUE)
nsv
nsvkeep <- nsv[nsv$nzv == F,]
nsvkeep <- rownames(nsvkeep)
data <- data[,nsvkeep]

# Divide the given training set into 4 roughly equal sets.
set.seed(9874653)
ids_small <- createDataPartition(y=data$classe, p=0.25, list=FALSE)
small_1 <- data[ids_small,]
remainder <- data[-ids_small,]
set.seed(9874653)
ids_small <- createDataPartition(y=remainder$classe, p=0.33, list=FALSE)
small_2 <- remainder[ids_small,]
remainder <- remainder[-ids_small,]
set.seed(9874653)
ids_small <- createDataPartition(y=remainder$classe, p=0.5, list=FALSE)
small_3 <- remainder[ids_small,]
small_4 <- remainder[-ids_small,]
# Divide each of these 4 sets into training (60%) and test (40%) sets.
set.seed(9874653)
inTrain <- createDataPartition(y=small_1$classe, p=0.6, list=FALSE)
training_small_1 <- small_1[inTrain,]
test_small_1 <- small_1[-inTrain,]
set.seed(9874653)
inTrain <- createDataPartition(y=small_2$classe, p=0.6, list=FALSE)
training_small_2 <- small_2[inTrain,]
test_small_2 <- small_2[-inTrain,]
set.seed(9874653)
inTrain <- createDataPartition(y=small_3$classe, p=0.6, list=FALSE)
training_small_3 <- small_3[inTrain,]
test_small_3 <- small_3[-inTrain,]
set.seed(9874653)
inTrain <- createDataPartition(y=small_4$classe, p=0.6, list=FALSE)
training_small_4 <- small_4[inTrain,]
test_small_4 <- small_4[-inTrain,]

#Classification Tree training
#Tree 1 - Default Settings
set.seed(8746)
tree <- train(training_small_1$classe ~ ., data = training_small_1, method= 'rpart')
print(tree, digits=3)
fancyRpartPlot(tree$finalModel, digits=2)

#Tree 1 - Evaluated against Test set 1
set.seed(8746)
tree_predictions <- predict(tree, newdata=test_small_1)
print(confusionMatrix(tree_predictions, test_small_1$classe), digits=3)

#Tree 2 - preprocessing only
set.seed(8746)
tree <- train(training_small_1$classe ~ ., preProcess=c('center', 'scale'), data = training_small_1, method= 'rpart')
print(tree, digits=3)

#Tree 3 - cross validation only
set.seed(8746)
tree <- train(training_small_1$classe ~ ., trControl=trainControl(method = "cv", number = 4), data = training_small_1, method= 'rpart')
print(tree, digits=3)

#Tree 4 - both Preprocessing and cross validation
set.seed(8746)
tree <- train(training_small_1$classe ~ ., preProcess=c('center', 'scale'),trControl=trainControl(method = "cv", number = 4), data = training_small_1, method= 'rpart')
print(tree, digits=3)

#Tree 4 - Evaluated against Test set 1
set.seed(8746)
tree_predictions <- predict(tree, newdata=test_small_1)
print(confusionMatrix(tree_predictions, test_small_1$classe), digits=3)

##Random Forest Training
#Forest 1.1 - Train on training set 1 with only cross validation.
set.seed(9856)
forest <- train(training_small_1$classe ~ ., method="rf", trControl=trainControl(method = "cv", number = 4), data=training_small_1)
print(forest, digits=3)

#Forest 1.1 - Evaluate against test set
predictions <- predict(forest, newdata=test_small_1)
print(confusionMatrix(predictions, test_small_1$classe), digits=4)

#Forest 1.2 - Train on training set 1 with cross validation and preprocessiing.
set.seed(9856)
forest <- train(training_small_1$classe ~ ., preProcess=c("center", "scale"), method="rf", trControl=trainControl(method = "cv", number = 4), data=training_small_1)
print(forest, digits=3)

#Forest 1.2 - Evaluate against test set
predictions <- predict(forest, newdata=test_small_1)
print(confusionMatrix(predictions, test_small_1$classe), digits=4)

#Test Forest 1.2 against provided test set
print(predict(forest, newdata = test))

#Forest 2 - Train on training set 2 with cross validation and preprocessiing.
set.seed(9856)
forest <- train(training_small_2$classe ~ ., preProcess=c("center", "scale"), method="rf", trControl=trainControl(method = "cv", number = 4), data=training_small_2)
print(forest, digits=3)

#Forest 2 - Evaluate against test set
predictions <- predict(forest, newdata=test_small_2)
print(confusionMatrix(predictions, test_small_2$classe), digits=4)

#Test Forest 2 against provided test set
print(predict(forest, newdata = test))

#Forest 3 - Train on training set 2 with cross validation and preprocessiing.
set.seed(9856)
forest <- train(training_small_3$classe ~ ., preProcess=c("center", "scale"), method="rf", trControl=trainControl(method = "cv", number = 4), data=training_small_3)
print(forest, digits=3)

#Forest 3 - Evaluate against test set
predictions <- predict(forest, newdata=test_small_3)
print(confusionMatrix(predictions, test_small_3$classe), digits=4)

#Test Forest 2 against provided test set
print(predict(forest, newdata = test))

#Forest 4 - Train on training set 2 with cross validation and preprocessiing.
set.seed(9856)
forest <- train(training_small_4$classe ~ ., preProcess=c("center", "scale"), method="rf", trControl=trainControl(method = "cv", number = 4), data=training_small_4)
print(forest, digits=3)

#Forest 4 - Evaluate against test set
predictions <- predict(forest, newdata=test_small_4)
print(confusionMatrix(predictions, test_small_4$classe), digits=4)

#Test Forest 2 against provided test set
print(predict(forest, newdata = test))


# Submission Function
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
pml_write_files_2 = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem2_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
answers <- c('B', 'A', 'B', 'A', 'A', 'E', 'D', 'B', 'A', 'A', 'B', 'C', 'B', 'A', 'E', 'E', 'A', 'B', 'B', 'B')
answers_2 <- c('B', 'A', 'A', 'A', 'A', 'E', 'D', 'B', 'A', 'A', 'B', 'C', 'B', 'A', 'E', 'E', 'A', 'B', 'B', 'B')
setwd("c:/Users/Paul/OneDrive/Coursera/Practical Machine Learning/PML Project/answers")
pml_write_files(answers)
pml_write_files_2(answers_2)
setwd("c:/Users/Paul/OneDrive/Coursera/Practical Machine Learning/PML Project")
