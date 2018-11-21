# PARAMETER GRID EXAMPLE

# we assume we have a model that has 2 parameters: C and sigma
# we want to evaluate this model with different parameter settings of C and sigma to find the best suited one
# just specify all values to be evaluated for each parameter, expand.grid constructs all permutations of them
library(caret)
myGrid <- expand.grid(C=3**(-10:10), sigma=3**(-10:10))
myGrid[c(1:3,100,200,300),]
str(myGrid) # each line = a parameter set for the classifier, with which the model is trained and evaluated

########################################################
# DATA PARTITIONING AND REGRESSION MODEL TRAINING
########################################################

library(caret) # caret ML framework
library(doMC) # parallelization
registerDoMC(3) # register 3 cores (more cores require more RAM)

# data partitioning
set.seed(12345) # make it reproducible
indexes_train <- createDataPartition(iris$Species, p = 0.75, list = F) # indexes of training samples
indexes_test <- (1:nrow(iris))[-indexes_train] # use all samples not in train as test samples

training <- iris[indexes_train,]
testing <- iris[indexes_test,]

# predict first numeric feature (target variable) from feature 2-4 - takes a moment...
modelKnn <- train(x = training[,2:4], 
                  y = training[,1], # ensure this is a numeric for regression and a factor for classification
                  preProcess = NULL, # center, scale, boxcox, pca, ...
                  method = 'knn', # model type
                  tuneGrid = expand.grid(k=1:20), # permutations of parameters to try out - use your own here!
                  metric = 'RMSE', # RMSE, Accuracy, Kappa, ROC, ...
                  trControl = trainControl(method = 'repeatedcv', # cv, repeatedcv, LOOCV, ...
                                           number = 10, # nr of CV partitions
                                           repeats = 20, # nr of partitioning repetitions
                                           returnData = F, 
                                           # classProbs = T, # enable computation of class probabilities?
                                           # summaryFunction = twoClassSummary, # use when classifying two classes 
                                           returnResamp = 'final', # return CV partition results for best model
                                           allowParallel = T))

# computed model -- provides further details with model$...
modelKnn
modelKnn$results
plot(modelKnn) # fitness landscape of this model as xy-plot: only works if there are different parametrizations
modelKnn # this final model object was trained with the best performing parameters on *all* training data (after evaluating all parameter sets)
str(modelKnn$finalModel) # ...and its parameters
# plot(modelKnn$finalModel) # we can plot *some* final models, but not the one from KNN

# A) Internally, train does random splits into CV partions, so results will be different each run
# To make train results reproducible use the "seeds" parameter in trainControl --
#   this can be used to do gallery independent partitioning with CV
#   (needs to contain n+1 seeds for n CV partitions)

# B) If you repeat your training and CV performance changes largely, you probably need more repeats

# regression error metrics 
# training data performance 
trainPredicted <- predict(modelKnn, training[,2:4]) # let model predict output variable for all training set samples
plot(trainPredicted,training[,1]) # plot predicted vs real values -- scatter represents error, straight line would mean perfect prediction
abline(0,1, col=2) # this line represents the ideal fit: with an error free model, all predictions would be on this line
sqrt(mean((trainPredicted - training[,1])**2)) # RMSE on training data = apparent error

# test data performance
testPredicted <- predict(modelKnn, testing[,2:4])
plot(testPredicted,testing[,1]) 
abline(0,1, col=2) # line representing perfect fit
sqrt(mean((testPredicted - testing[,1])**2)) # realistic error estimate from held back test set: higher than the apparent error

# try different models + parameter grids!

########################################################
# DATA PARTITIONING AND CLASSIFICATION MODEL TRAINING
########################################################

library(caret) # caret ML framework
library(doMC) # paralellization
registerDoMC(3) # register 3 cores (more cores require more RAM)

# make data partitioning reproducible
set.seed(12345) # make it reproducible
indexes_train <- createDataPartition(iris$Species, p = 0.75, list = F) # indexes of training samples
indexes_test <- (1:nrow(iris))[-indexes_train] # use all samples not in train as test samples

training <- iris[indexes_train,]
testing <- iris[indexes_test,]

# iris classification from all 4 features using training data only
modelKnn <- train(x = training[,1:4], 
                  y = training[,5], # ensure this is a numeric for regression and a factor for classification
                  preProcess = NULL, # center, scale, boxcox, pca, ...
                  method = 'knn', # model type
                  tuneGrid = expand.grid(k=1:10), # permutations of parameters to try out
                  metric = 'Kappa', # RMSE, Accuracy, Kappa, ROC, ... 
                  trControl = trainControl(method = 'repeatedcv', # none, cv, repeatedcv, LOOCV, ...
                                           number = 10, # nr of CV partitions
                                           repeats = 20, # nr of partitioning repetitions
                                           returnData = F, 
                                           classProbs = T, # enable computation of class probabilities?
                                           # summaryFunction = twoClassSummary, # use when classifying two classes 
                                           returnResamp = 'final', # return CV partition results for best model
                                           allowParallel = T))

# computed model -- provides further details with model$...
modelKnn
modelKnn$results
plot(modelKnn) # only work if there are different parametrizations
modelKnn # this final model object was trained with the best performing parameters on *all* training data (after evaluating all parameter sets)
str(modelKnn$finalModel) # ...and its parameters
# plot(modelKnn$finalModel) # we can plot *some* final models, but not the one from KNN

# Confusion matrixse
# Training confusion Matrix - this is based on the apparent error!
trainPredicted <- predict(modelKnn, newdata = training[,1:4])
trainConfMatrix <- confusionMatrix(data = trainPredicted, reference = training[,5]) # create confusion matrix over training data = apparent error
trainConfMatrix
levelplot(sweep(x = trainConfMatrix$table, STATS = colSums(trainConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))

# Test confusion matrix: realistic error estimate from held back test set, usually higher than apparent error
testPredicted <- predict(modelKnn, newdata = testing[,1:4])
testConfMatrix <- confusionMatrix(data = testPredicted, reference = testing[,5])
testConfMatrix
levelplot(sweep(x = testConfMatrix$table, STATS = colSums(testConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))

# All CV results can be used together for a realistic error estimate
# This is the one you usually want to use during development
cvConfMatrix <- confusionMatrix(modelKnn)
cvConfMatrix
levelplot(sweep(x = cvConfMatrix$table, STATS = colSums(cvConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))

########################################################
# MODEL SELECTION
########################################################

library(caret) # caret ML framework
library(doMC) # paralellization
registerDoMC(3) # register 3 cores (more cores require more RAM)

# make data partitioning reproducible
set.seed(12345) # make it reproducible
indexes_train <- createDataPartition(iris$Species, p = 0.75, list = F) # indexes of training samples
indexes_test <- (1:nrow(iris))[-indexes_train] # use all samples not in train as test samples

training <- iris[indexes_train,]
testing <- iris[indexes_test,]

# applied to all models (execpt e.g. results for one model are not stable)
trControl <- trainControl(method = 'repeatedcv', 
                          number = 10, 
                          repeats = 20, 
                          returnData = F, 
                          classProbs = T, 
                          returnResamp = 'final', 
                          allowParallel = T)

# aggregate all models in a list
models <- list()

names(getModelInfo('knn')) # KNN
getModelInfo('knn')[[2]]$parameters 
models$modelKnn <- train( x = training[,1:4], 
                          y = training[,5], 
                          preProcess = NULL, 
                          method = 'knn', 
                          tuneGrid = expand.grid(k=1:10), 
                          metric = 'Kappa', 
                          trControl = trControl)
# look at the 1D fitness landscape of this model
models$modelKnn
plot(models$modelKnn)
cvConfMatrix <- confusionMatrix(models$modelKnn)
cvConfMatrix
levelplot(sweep(x = cvConfMatrix$table, STATS = colSums(cvConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))

names(getModelInfo('nb')) # naive bayes 
getModelInfo('nb')[[1]]$parameters 
models$modelNaiveBayes <- train( x = training[,1:4], 
                                 y = training[,5], 
                                 preProcess = NULL, 
                                 method = 'nb', 
                                 tuneGrid = NULL, # autoselects parameters -- be careful, parameters are likely not optimal... 
                                 metric = 'Kappa', 
                                 trControl = trControl)
models$modelNaiveBayes
plot(models$modelNaiveBayes)
str(models$modelNaiveBayes$finalModel) # the final model
plot(models$modelNaiveBayes$finalModel) # visual representations of final model
cvConfMatrix <- confusionMatrix(models$modelNaiveBayes)
cvConfMatrix
levelplot(sweep(x = cvConfMatrix$table, STATS = colSums(cvConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))

names(getModelInfo('lda')) # linear discriminant analysis
getModelInfo('lda')[[1]]$parameters # this model does not have any hyperparameters
models$modelLda <- train(x = training[,1:4], 
                         y = training[,5], 
                         preProcess = NULL, 
                         method = 'lda', 
                         tuneGrid = NULL, 
                         metric = 'Kappa', 
                         trControl = trControl)
models$modelLda
cvConfMatrix <- confusionMatrix(models$modelLda)
cvConfMatrix
levelplot(sweep(x = cvConfMatrix$table, STATS = colSums(cvConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))

names(getModelInfo('lda2')) # linear discriminant analysis 
getModelInfo('lda2')[[1]]$parameters # ...with hyperparameters
models$modelLda2 <- train(x = training[,1:4], 
                          y = training[,5], 
                          preProcess = NULL, # this LDA models has no tuning parameters
                          method = 'lda2', 
                          tuneGrid = expand.grid(dimen=1:6), 
                          metric = 'Kappa',
                          trControl = trControl)
models$modelLda2
plot(models$modelLda2)
cvConfMatrix <- confusionMatrix(models$modelLda2)
cvConfMatrix
levelplot(sweep(x = cvConfMatrix$table, STATS = colSums(cvConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))

names(getModelInfo('svmRadial')) # radial=Gaussian SVM
getModelInfo('svmRadial')[[2]]$parameters
models$modelSvmRadial<- train( x = training[,1:4], 
                               y = training[,5], 
                               preProcess = NULL, 
                               method = 'svmRadial', 
                               tuneGrid = expand.grid(sigma=3**(-4:0), C=3**(-0:4)), # use DIFFERENT parameter ranges for your problems, e.g. try 3**(-10:10)
                               metric = 'Kappa',
                               trControl = trControl)
models$modelSvmRadial
# radial svm has 2 parameters: we can look at a 2D fitness landscape now (either default caret plot or levelplot)
plot(models$modelSvmRadial, scales=list(log=3)) 
levelplot(data = models$modelSvmRadial$results, x = Kappa ~ C * sigma, col.regions = gray(100:0/100), scales=list(log=3)) 
models$modelSvmRadial$finalModel
cvConfMatrix <- confusionMatrix(models$modelSvmRadial)
cvConfMatrix
levelplot(sweep(x = cvConfMatrix$table, STATS = colSums(cvConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))

names(getModelInfo('rpart')) # classification and regression tree (CART)
getModelInfo('rpart')[[1]]$parameters
models$modelRpart<- train( x = training[,1:4], 
                           y = training[,5], 
                           preProcess = NULL, 
                           method = 'rpart', 
                           tuneGrid = expand.grid(cp = seq(0.05, 0.95, 0.05)), 
                           metric = 'Kappa',
                           trControl = trControl)
models$modelRpart
plot(models$modelRpart)
models$modelRpart$finalModel
plot(models$modelRpart$finalModel)
text(models$modelRpart$finalModel)
cvConfMatrix <- confusionMatrix(models$modelRpart)
cvConfMatrix
levelplot(sweep(x = cvConfMatrix$table, STATS = colSums(cvConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))
# nicer CART plot
# install.packages('rattle') # requires libgtk2.0-dev library and rpart.plot package
library(rattle)
fancyRpartPlot(models$modelRpart$finalModel)

names(getModelInfo('rf')) # random forest
getModelInfo('rf')[[2]]$parameters
models$modelRandomForest <- train( x = training[,1:4], 
                                   y = training[,5], 
                                   preProcess = NULL, 
                                   method = 'rf',
                                   tuneGrid = expand.grid(mtry=1:4),
                                   metric = 'Kappa', 
                                   trControl = trControl)
models$modelRandomForest
plot(models$modelRandomForest)
models$modelRandomForest$finalModel
plot(models$modelRandomForest$finalModel)
cvConfMatrix <- confusionMatrix(models$modelRandomForest)
cvConfMatrix
levelplot(sweep(x = cvConfMatrix$table, STATS = colSums(cvConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))

# we can easily save and load all models for later usage/analysis
saveRDS(object = models, file = '03_models_demonstration.RData')
# models <- readRDS('03_models_demonstration.RData')

# compare models: uses all CV partition performances for each model
results <- resamples(models)
summary(results)
bwplot(results) # bwplot is the boxplot version of the lattice library ("box-and-wisker-plot")

# REMEMBER
# try different models + parameter sets
# you might want to ensure that the performance for each model is "stable" 
#   (= doesn't change "too much" if re-run)
# think about what the performance differences between models mean and
#   if they fit your expectation from graphical data/feature analysis
# remember to chose the least complex model with reasonable performance!

########################################################
# Specialties: Gallery Independent CV partitioning
########################################################

library(caret)
library(plyr)
library(doMC)
registerDoMC(3)

# example: predict iris sepal length from sepal width and petal length. this should also work 
#   on new species of flowers = be gallery independent accross flower spcies.
# therefore need to split data in gallery independent way
pairs(iris[,1:3], col=iris$Species, upper.panel = NULL)

# a) NOT gallery independent (accross iris classes)
# shuffle data + create train/test partitions
indexes_train <- createDataPartition(iris[,1], p = 0.5, list = F)
pairs(iris[indexes_train, 1:3], col=iris$Species[indexes_train], upper.panel = NULL)
# train
model <- train(x = iris[indexes_train,2:3], y = iris[indexes_train,1], method = 'lm', 
               trControl = trainControl(method = 'repeatedcv', number = 3, repeats = 3, allowParallel = T, 
                                        returnData = F, returnResamp = 'final', savePredictions = F, classProbs = F))
model
model$resample # pretty similar error over partitions
# train error
predictedTrain <- predict(model, iris[indexes_train,2:3])
plot(predictedTrain, x = iris[indexes_train,1]) # apparent error
abline(0, 1, col=2)
sqrt(mean((predictedTrain-iris[indexes_train,1])**2))
# test error
predictedTest <- predict(model, iris[-indexes_train,2:3])
plot(predictedTest, x = iris[-indexes_train,1]) # realistic error from held back test set
abline(0, 1, col=2)
sqrt(mean((predictedTest-iris[-indexes_train,1])**2))
# all looks good so far - but there is a hidden problem...

# b) gallery independent (accross iris classes)
# as we only have 3 classes here we don't keep a separate test partition now
# with real-life scenarios always keep a gallery idependent test partition as well (data from people not used in training at all)
# create the cv-folds per hand = indexes of iris sample split by their classes
#   play attention to the factor(...) --> causes classes excluded from training to be removed from cv partitioning completely
cv_indexes <- llply(unique(iris$Species), function(cls) which(iris$Species!=cls) ) 
names(cv_indexes) <- paste0('subj_', unique(iris$Species))
str(cv_indexes)
# sanity check: each partition = row in the table contains 1 class of flowers
l_ply(cv_indexes, function(x) print(table(iris$Species[x])) )
# train: in trainControl we can leave out the method, number, and repeats parameter:
#   the corresponding behaviour is specified by the index (and poss. indexOut) parameters alone
model <- train(x = iris[,2:3], 
               y = iris[,1], 
               method = 'lm', 
               trControl = trainControl(allowParallel = T,                           
                                        returnData = F, 
                                        returnResamp = 'final',
                                        savePredictions = F,
                                        classProbs = F,
                                        index = cv_indexes))
model # higher overall error...
model$resample # ...because of the "hidden" problem:
#   prediciton works well for some but not all subjects (in our case: classes of flowers)
#   when training from versicolor and virginica, predicting the target variable for the new subject setosa is difficult
#   this has been hidden with gallery dependent data splitting (using sample of all subjects in training)
# review why: comes from different relation between features and target variable over subjects
#   clearly visible in our data (often difficult to find out with real data!)
pairs(iris[,c(1:3)], col=iris$Species, upper.panel = NULL)
