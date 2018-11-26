library(plyr)
library(png)
library(caret)
library(corrplot)
library(doParallel)
registerDoParallel(4)


facesPaths <- dir("./faces_data", full.names=T, pattern='png')

faces <-ldply(facesPaths, function(f) { as.numeric(readPNG(f)) })
classes <-laply(facesPaths, function(f) { paste("P", substr(f, 18, 19), sep="") })  #prepend a 'P' so the classnames are valid R vairable names

faces$class <- as.factor(classes)

dim(faces)
str(faces$class)

#plot first face
plot(1:1, type='n', main="face0, original", xlim=c(1,50), ylim=c(1,50), asp=1)
rasterImage(matrix(as.numeric(faces[1,-2501]), 50,50),0,0,50,50)

#correlation of first (top) 100 pixels (first two rows of pixels)
face00TopPixels = faces[,1:99]
#correlation of 100 pixels (two rows) from the middle
face00CenterPixels = faces[,1201:1300]

corrplot(cor(face00TopPixels), 'circle', tl.cex = 0.3)
corrplot(cor(face00CenterPixels), 'circle', tl.cex = 0.3)

# split data in training and test set
indexes_train <- createDataPartition(faces$class, p = 0.75, list = F)
indexes_test <- (1:nrow(faces))[-indexes_train]

training <- faces[indexes_train,]
testing <- faces[indexes_test,]

#PCA calculation of training data
pcaResult <- prcomp(x = training[,1:2500], retx = T, center = T, scale. = T, tol = 0.1) #tol: higher means omitting more features
dim(training[,1:2500])
dim(pcaResult$x)

#reverse PCA (with less dimensions than original data)
reversedPca <- scale(pcaResult$x %*% t(pcaResult$rotation), scale = 1/pcaResult$scale, center= -pcaResult$center/pcaResult$scale)

#plot first face from PCA (dimensionaly reduced) data
scaleToRgbValues <- function(x){(x-min(x))/(max(x)-min(x))}
plot(1:1, type='n', main="face0, PCA 90%", xlim=c(1,50), ylim=c(1,50), asp=1)
rasterImage(matrix(as.numeric(scaleToRgbValues(reversedPca[1,])), 50,50),0,0,50,50)


#train models
##############

#configuration
trControl <- trainControl(method = 'repeatedcv', 
                          number = 10, 
                          repeats = 2, 
                          returnData = F, 
                          classProbs = T, 
                          returnResamp = 'final', 
                          allowParallel = T,
                          preProcOptions = list(thresh = 0.9) #thresh: higher means retaining more features
                          )

models <- list()
models <- readRDS('models.RData')  #load existing models if possible

#train model with KNN without PCA
models$modelKnn <- train(x = training[,-ncol(training)],
                              y = training$class,
                              preProcess = NULL,
                              method = 'knn',
                              tuneGrid = expand.grid(k=1:3),
                              metric = 'Kappa',
                              trControl = trControl)
plot(models$modelKnn)

#train model with KNN and PCA with retaining only 10% of the features (discard 90% of features)
models$modelKnnPCA01 <- train(x = training[,-ncol(training)],
                              y = training$class,
                              preProcess = c('pca'),
                              method = 'knn',
                              tuneGrid = expand.grid(k=1:3),
                              metric = 'Kappa',
                              trControl = trainControl(method = 'repeatedcv', 
                                                       number = 10, 
                                                       repeats = 2, 
                                                       returnData = F, 
                                                       classProbs = T, 
                                                       returnResamp = 'final', 
                                                       allowParallel = T,
                                                       preProcOptions = list(thresh = 0.1)))
plot(models$modelKnnPCA01)

#train model with KNN and PCA with retaining 90% of the features (discard 10% of features)
models$modelKnnPCA09 <- train(x = training[,-ncol(training)],
                         y = training$class,
                         preProcess = c('pca'),
                         method = 'knn',
                         tuneGrid = expand.grid(k=1:3),
                         metric = 'Kappa',
                         trControl =  trControl
                         )
plot(models$modelKnnPCA09)

#train model with LDA without PCA
models$modelLda <- train(x = training[,-ncol(training)],
                         y = training$class, 
                         preProcess = NULL, 
                         method = 'lda', 
                         tuneGrid = NULL, 
                         metric = 'Kappa', 
                         trControl = trControl)

#train model with LDA and PCA with retaining 90% of the features (discard 10% of features)
models$modelLdaPca09 <- train(x = training[,-ncol(training)],
                              y = training$class, 
                              preProcess = c('pca'), 
                              method = 'lda', 
                              tuneGrid = NULL, 
                              metric = 'Kappa', 
                              trControl = trControl)

models$modelLdaPca09$preProcess

#train model with LDA and PCA with retaining 10% of the features (discard 90% of features)
models$modelLdaPca01 <- train(x = training[,-ncol(training)],
                              y = training$class, 
                              preProcess = c('pca'), 
                              method = 'lda', 
                              tuneGrid = NULL, 
                               metric = 'Kappa', 
                               trControl = trainControl(method = 'repeatedcv', 
                                                        number = 10, 
                                                        repeats = 2, 
                                                        returnData = F, 
                                                        classProbs = T, 
                                                        returnResamp = 'final', 
                                                        allowParallel = T,
                                                        preProcOptions = list(thresh = 0.1)))

models$modelLdaPca01$preProcess


#train model with SVM and PCA with retaining 90% of the features (discard 10% of features)
#takes a long time to train
models$modelSvmRadial<- train(x = training[,-ncol(training)],
                              y = training$class, 
                              preProcess = c('pca'), 
                              method = 'svmRadial', 
                              tuneGrid = expand.grid(sigma=3**(-3:3), C=3**(-3:3)),
                              metric = 'Kappa',
                              trControl = trControl)

#plot accuracy and kappa of all models as boxplot
results <- resamples(models)
summary(results)
bwplot(results)

#confusion matrix of modelLdaPca09 with training data
trainPredicted <- predict(models$modelLdaPca09, newdata = training[,-ncol(training)])
trainConfMatrix <- confusionMatrix(data = trainPredicted, reference = training$class) # create confusion matrix over training data = apparent error
trainConfMatrix
levelplot(sweep(x = trainConfMatrix$table, STATS = colSums(trainConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))

#confusion matrix of modelLdaPca09 with testing data
#modelLdaPca09 performs perfect on training data, but not on testing
testPredicted <- predict(models$modelLdaPca09, newdata = testing[,-ncol(testing)])
testConfMatrix <- confusionMatrix(data = testPredicted, reference = testing$class)
testConfMatrix
levelplot(sweep(x = testConfMatrix$table, STATS = colSums(testConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))

#confusion matrix of modelLda with testing data
#modelLda (without PCA) performs perfect on the testing data
testPredicted <- predict(models$modelLda, newdata = testing[,-ncol(testing)])
testConfMatrix <- confusionMatrix(data = testPredicted, reference = testing$class)
testConfMatrix
levelplot(sweep(x = testConfMatrix$table, STATS = colSums(testConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100), main="Confusion Matrxi LDA (no PCA)")


saveRDS(object = models, file = 'models.RData')
