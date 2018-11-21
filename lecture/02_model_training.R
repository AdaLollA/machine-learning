# ! PITFALL ! 
# Keep in mind that those model training examples are only for your
#   understanding - they're done WITHOUT proper data partitioning. 
# ALWAYS USE DATA PARTITIONING ON YOUR DATA!
# Also: always use proper model tuning = hyperparameter grid search
#   instead of fixed hyperparameters

########################################################
# MODEL TRAINING: REGRESSION
########################################################

library(caret) # caret ML framework
library(doMC) # parallelization
registerDoMC(3) # register 3 cores (more cores require more RAM)
# if doMC does not work for you: caret can use different libraries for parallelization (all that implement the parallel "foreach")
# more details: http://topepo.github.io/caret/parallel.html

# model training
# which models does caret provide? a lot: http://topepo.github.io/caret/bytag.html
?getModelInfo
names(getModelInfo())
names(getModelInfo('knn')) # 2nd is the one we use
getModelInfo('knn')[[2]]$parameters # model parameters --> caret documentation/search engines are your friends!

# train a KNN model
modelKnn <- train(x = iris[,2:4], 
                  y = iris[,1], # ensure this is a numeric for regression and a factor for classification
                  method = 'knn', # model type
                  tuneGrid = expand.grid(k=5), # permutations of parameters to try out -- for now we use a fixed parameter here
                  metric = 'RMSE', # RMSE, Accuracy, Kappa, ROC, ... ROC=AUC and requires "summaryFunction = twoClassSummary" below
                  trControl = trainControl(method = 'none', # DANGER never use 'none' here in RL! this is just for demo!
                                           allowParallel = T))

# computed model -- provides further details with model$...
modelKnn

# regression error
predicted <- predict(modelKnn, iris[,2:4]) # let model predict output variable for all training set samples
plot(predicted,iris[,1]) # plot predicted vs real values -- scatter represents error, straight line would mean perfect prediction
sqrt(mean((predicted - iris[,1])**2)) # RMSE on training data

# train linear model
modelLm <- train(x = iris[,2:4], 
                 y = iris[,1], # ensure this is numeric for regression and a factor for classification
                 method = 'lm', # model type
                 metric = 'RMSE', # RMSE, Accuracy, Kappa, ROC, ... ROC=AUC and requires "summaryFunction = twoClassSummary" below
                 trControl = trainControl(method = 'none', # DANGER never use 'none' here in RL! this is just for demo!
                                          allowParallel = T))

# computed model -- provides further details with model$...
modelLm

# regression error
predicted <- predict(modelLm, iris[,2:4]) # let model predict output variable for all training set samples
plot(predicted,iris[,1]) # plot predicted vs real values -- scatter represents error, straight line would mean perfect prediction
abline(a=0, b=1, col=2) # this line represents the ideal fit: with an error free model, all predictions would be on this line
sqrt(mean((predicted - iris[,1])**2)) # RMSE on training data = apparent error

########################################################
# MODEL TRAINING: CLASSIFICATION
########################################################

library(caret) # caret ML framework
library(doMC) # parallelization
registerDoMC(3) # register 3 cores (more cores require more RAM)

# train KNN model
modelKnn <- train(x = iris[,1:4], y = iris[,5], method = 'knn', 
                  tuneGrid = expand.grid(k=5), metric = 'Kappa',
                  trControl = trainControl(method = 'none', # DANGER never use 'none' here in RL! this is just for demo!
                                           allowParallel = T))

# computed model -- provides further details with model$...
modelKnn

# Confusion matrixes -- details about these later
predicted <- predict(modelKnn, newdata = iris[,1:4])
head(predicted)
tail(predicted)
conf <- confusionMatrix(data = predicted, reference = iris[,5]) # create confusion matrix over all involved classes
conf
levelplot(conf$table, col.regions=gray(100:0/100)) # absolute
levelplot(sweep(x = conf$table, STATS = colSums(conf$table), MARGIN = 2, FUN = `/`), col.regions=gray(100:0/100)) # relative
# `+`, `-`, `*`, `/` are "operators"
# sweep just applies a vector to a matrix using an operator, e.g. "divide matrix row-wise by vector" 

########################################################
# MODEL TRAINING: CLASS PROBABILITIES + ROC FOR 2 CLASSES
########################################################

library(caret)
library(pROC)
library(doMC) 
registerDoMC(3) 

d <- iris[iris$Species != 'setosa',]
d$Species <- factor(d$Species)

# train KNN model
modelKnn <- train(x = d[,1:4], y = d[,5], method = 'knn', 
                  tuneGrid = expand.grid(k=5), metric = 'ROC', # we can use the AUC (called ROC) as metric 
                  trControl = trainControl(method = 'none', # DANGER never use 'none' here in RL! this is just for demo!
                                           allowParallel = T, 
                                           classProbs = T, # we need to switch computation of class probabilites on (changes internal things with the model). 
                                           # if this is false, class probabilities for new samples cannot be calculated using the model later.
                                           summaryFunction = twoClassSummary)) # internally changes how class summaries are calculated
modelKnn

# now we can either predict classes as usual (e.g. for confusion matrixes etc.)...
predicted <- predict(modelKnn, newdata = d[,1:4])
predicted
head(predicted)
conf <- confusionMatrix(predicted, d[,5])
levelplot(sweep(conf$table, MARGIN = 2, STATS = colSums(conf$table), FUN = `/`), col.regions = gray(100:0/100))
# ...or class probabilities instead:
predicted <- predict(modelKnn, newdata = d[,1:4], type = 'prob') 
head(predicted)
tail(predicted)

# ROC curve -- this only works for 2 class problems (reduce your n class problem problem to n 2-class-problems if you need ROC curves for individual classes)
# We use the pROC library for this, but other can achieve the same, e.g. the ROCR library
myRoc <- roc(response = d[,5], predictor = predicted[,1])
myRoc # contains all information about the ROC curve, including the AUC
plot(myRoc) 
abline(0,1, col='gray70') # diagonal line for EER

########################################################
# MODEL TRAINING: CLASS PROBABILITIES + ROC FOR N CLASSES
########################################################

library(caret)
library(pROC)
library(doMC) 
registerDoMC(3) 

d <- iris # now we use all 3 classes

# train KNN model
modelKnn <- train(x = d[,1:4], y = d[,5], method = 'knn', 
                  tuneGrid = expand.grid(k=5), metric = 'ROC', 
                  trControl = trainControl(method = 'none', # DANGER never use 'none' here in RL! this is just for demo!
                                           allowParallel = T, 
                                           classProbs = T)) 
modelKnn
predicted <- predict(modelKnn, newdata = d[,1:4], type = 'prob') 
head(predicted)
tail(predicted)

# now we plot the ROC curve for each class. For class X this means "class X vs all other classes".
# ensure the same classes are used for the probability and class-check:
plot(roc(response = d[,5]=='setosa',     predictor = predicted$setosa),     col = 1) 
plot(roc(response = d[,5]=='versicolor', predictor = predicted$versicolor), col = 2, add = T) 
plot(roc(response = d[,5]=='virginica',  predictor = predicted$virginica),  col = 3, add = T)
abline(0,1, col='gray70') # diagonal line for EER
legend('bottomright', legend=c('setosa', 'versicolor', 'virginica'), lty=1, col=1:3, cex=0.9)

########################################################
# UP- AND DOWNSAMPLING
########################################################

# http://topepo.github.io/caret/sampling.html

library(caret)

# we treat cyl as the class label
str(mtcars)
# unbalanced classes
table(factor(mtcars[,2]))

# downsampling: sample from bigger classes only as much samples as there are from smallest class
d1 <- downSample(x = mtcars[,-2], y = factor(mtcars[,2]), list = F)
table(d1$Class)

# upsampling: sample with replacement from all smaller classes until there are as much samples as from the biggest class
d2 <- upSample(x = mtcars[,-2], y = factor(mtcars[,2]), list = F)
table(d2$Class)
