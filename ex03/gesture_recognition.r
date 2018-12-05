library(caret)
library(doParallel)
registerDoParallel(6)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

filePath <- "./wear_data/raw_data_wear_%s.csv"

# todo: use vector commands
#files <- c(sprintf(filePath, "x"), filePath, "y"), filePath, "z"))
#d <- lapply(files, read.table, sep=',', fill = T, col.names = c('gesture', 'person', 'sample', paste('acc', 1:1000, sep='')))
#summary(d[1])

x <- read.table(sprintf(filePath, "x"), sep=',', fill = T, col.names = c('gesture', 'person', 'sample', paste('acc', 1:1000, sep='')))
y <- read.table(sprintf(filePath, "y"), sep=',', fill = T, col.names = c('gesture', 'person', 'sample', paste('acc', 1:1000, sep='')))
z <- read.table(sprintf(filePath, "z"), sep=',', fill = T, col.names = c('gesture', 'person', 'sample', paste('acc', 1:1000, sep='')))

# todo: use vector commands for interpolation, the for loops are quiet slow

stepwidth <- 0.001 # we want 1000 values

for (i in 1:length(x[,1])){
  x1 <- x[i, -(1:3)][!is.na(x[i, -(1:3)])]
  x[i, -(1:3)] <- approx(x = seq(0,1,1/(length(x1)-1)), y = x1, xout = seq(0,1-stepwidth,stepwidth))$y
}

for (i in 1:length(y[,1])){
  y1 <- y[i, -(1:3)][!is.na(y[i, -(1:3)])]
  y[i, -(1:3)] <- approx(x = seq(0,1,1/(length(y1)-1)), y = y1, xout = seq(0,1-stepwidth,stepwidth))$y
}

for (i in 1:length(z[,1])){
  z1 <- z[i, -(1:3)][!is.na(z[i, -(1:3)])]
  z[i, -(1:3)] <- approx(x = seq(0,1,1/(length(z1)-1)), y = z1, xout = seq(0,1-stepwidth,stepwidth))$y
}

# combine x, y and z acceleration into one dataframe
xyz <- x[,1:3]
xyz[4:1003] <- x[, -(1:3)]
xyz[1004:2003] <- y[, -(1:3)]
xyz[2004:3003] <- z[, -(1:3)]

rm(x, y, z, x1, y1, z1, i, stepwidth, filePath) # clean environment

# save/load for later use
saveRDS(object = xyz, file = 'xyz_raw_interpolated_data.RData')
xyz <- readRDS('xyz_raw_interpolated_data.RData')

xyz[, -(1:3)] <- scale(xyz[, -(1:3)])

#running median of each row (i think this is like a low pass)
library(signal)

test <- apply(xyz[,4:3003], 1, function(x1){
  lowpass(x1, k=201)
})


# plot first and second row of dataset with the 'low pass'
plot((as.numeric(xyz[1,-(1:3)])), type='l')
par(new=TRUE)
plot((as.numeric(test[,1])), type='l', col=3)
par(new=TRUE)
plot((as.numeric(xyz[2,-(1:3)])), type='l', col=2)
par(new=TRUE)
plot((as.numeric(test[,2])), type='l', col=4)

plot(test[,1], type='l', col=2)

# calculate mean of all 'up' and 'down' movements and plot it
# this should give an idea of how the averate up/down movement looks like
plot(4:3003, numeric(3000), type='l')
xyz_means_up <- colMeans(xyz[xyz$gesture == "up", ][,-(1:3)])
lines(xyz_means_up, col=3)
xyz_means_down <- colMeans(xyz[xyz$gesture == "down", ][,-(1:3)])
lines(xyz_means_down, col=2)


set.seed(12345) # make it reproducible
# use a training set with ONLY 30% of the data to speed up training, INCREASE FOR FINAL TRAINING!
indexes_train <- createDataPartition(xyz$gesture, p = 0.3, list = F) 
indexes_test <- (1:nrow(xyz))[-indexes_train]

training <- xyz[indexes_train,]
testing <- xyz[indexes_test,]

# train a knn model with pca preprocessing with all 3000 values just to get a baseline of what accuracy is possible without feature exraction
trControl <- trainControl(method = 'repeatedcv', 
                          number = 5, 
                          repeats = 2, 
                          returnData = F, 
                          classProbs = T, 
                          returnResamp = 'final', 
                          allowParallel = T,
                          preProcOptions = list(thresh = 0.99))

models <- list()
models$modelKnn <- train(x = training[,-(1:3)], 
                         y = training$gesture, 
                         preProcess = c('pca'), 
                         method = 'knn', 
                         tuneGrid = expand.grid(k=1:5), 
                         metric = 'Kappa', 
                         trControl = trControl)

models$modelKnn$preProcess # PCA needed 85 components to capture 99 percent of the variance
models$modelKnn # Accuracy 0.9211654
plot(models$modelKnn) # best Kappa with k=1

#confusion matrix
cvConfMatrix <- confusionMatrix(models$modelKnn)
cvConfMatrix
levelplot(sweep(x = cvConfMatrix$table, STATS = colSums(cvConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))

#test with test data set
testPredicted <- predict(models$modelKnn, newdata = testing[,-(1:3)])
testConfMatrix <- confusionMatrix(data = testPredicted, reference = testing$gesture)
testConfMatrix
levelplot(sweep(x = testConfMatrix$table, STATS = colSums(testConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))
