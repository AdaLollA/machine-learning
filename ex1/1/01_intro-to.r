# Contains content from:
#   http://www.cyclismo.org/tutorial/R/
#   https://learnxinyminutes.com/docs/r/
#   https://www.datacamp.com/
#   http://www.r-bloggers.com/a-crash-course-in-r/
#   http://www.r-bloggers.com/how-to-learn-r/
#   http://www.r-bloggers.com/how-to-learn-r-2/

# If you are stuck somewhere: your favourite search engine 
# is your friend -- R is pretty well documented by now
# 
# Integrated help is good as well: ?FUNCTION, ??FUNCTION

########################################################
# R BASICS
########################################################

# Comments start with #

# Statements can but need not end with ;
1
1;
1+1
1+1;

# Workspace is empty
ls()
# Assignments are done with <-, = works as well but had different effects on scope (will fail when not used top level)
a <- 1
b <- 'fooBar'
# We now have this variable in our workspace
ls()
# Print variable
print(a)
# Printing is done automatically in interactive shell
a
b

# Function calls are done as function(parameters), like ls(), print()
# 2 important functions: str and summary
# Structure of variable = overview of variable. Works on pretty much every object
str(a) 
str(print)
# Summary of variable = some more infos (does not work on function objects)
summary(a)
summary(print)

# remove objects from workspace
rm(a)
# garbage colletion is done automatically, but you can call it manually using gc() if you just deleted GBs of data from your workspace...
gc(reset = T)

# Chosing variable/function names
# We are allowed to overwrite pretty much any variable/function name
# Hence, before creating arbitrary variable/function names, check if it already exists by just typing it + Enter
# Example of used names:
T
F
# we would be able to overwrite the TRUE value -- very dangerous, don't do that stuff...
# T <- F
c
t
df
labels
names
file
x # free!
# variable names cannot start with e.g. a number
123X <- 5
X123 <- 5

# Create a vector with start:end or concatenate c(...) notation
myVector1 <- c(3,5,7,9) 
myVector1
str(myVector1)
myVector2 <- 1:10
myVector2
myVector2.2 <- seq(1,10,2) # sequence with steps -- note that 10 is not included due to the stepwidth
myVector2.2
myVector3 <- c(myVector2, c(100,200))
myVector3
myVector4 <- c('foo', 'bar')
myVector4
str(myVector4)

# Vectors contain only values of 1 data type -- lists are for different data types, see below
myVector <- c(3,'a')
myVector # 3 is cast to character
str(myVector)

# R is index 1 based, indexing is done with [...]
myVector <- 10:1
myVector[0] # not an index of myVector, result is of length 0
myVector[1] # character at 1st positien
myVector[2] # character at 2nd position
myVector[10] # character at 10th position
myVector[11] # NA
# indexing vectors with [[...]] is more safe
# (but it only works with 1 element, therefore it's rarely used)
myVector[[0]] # error
myVector[[1]] # ok
myVector[[11]] # error

# We can index more than 1 element
myVector[1:2]
myVector[c(1,3,5,7,9)]
myVector[c(1:4,8)]

# Function parameters can be named, which allows for chainging parameter order
# Without specifying parameter names, order is specified by function definition
log(100, 10)
log(x = 100, base = 10)
log(base = 10, x = 100)
log(100, base = 10)
log(10, x = 100)
# Left out parameters are "defaulted" with default values specified by the function definition
log(100)
log(x = 100)
# ... parameters which need to be specified cannot be ommited
log(base = 10)
# suggestion: use named parameters!

# Getting help for functions, parameters, etc
# Open help for this function
?log
?interpolate
# Search within help for such functions (may take a while)
??log
??interpolate

# Tons of functions... 
# namespace::functionName can be used do directly use functions without explicitly loading their library first
?base::mean
?stats::kmeans

# Lists can contain multiple variable types
myList <- list('a', 2, 3, 4.0)
str(myList)
# ! PITFALL ! there are two different forms of indexing for lists
# Indexing lists with [...] leads to sub-list even if only 1 element is taken
str(myList[1:2])
str(myList[1])
# Indexing lists with [[...]] takes out the element and 
str(myList[[1]])
# ... of coure that cannot work when indexing more than 1 element
str(myList[[1:2]])

# Almost everything is vectorized = fast
# Keep this is mind! It's a core concept and huge speedup (e.g. avoid loops but use vectorization)
log(100, base=10)
log(1:20, base=10)
# Modulo
10 %% 3
(1:10) %% 3

# Logicals (case sensitive)
TRUE
T
FALSE
F
T==TRUE
1==1
1 < 2
1 > 2
1 <= 2
1 == 1
1 != 2
# ! PITFALL ! regular and vectorized operators (e.g. AND and OR) ar different:
T && T # regular AND
c(T,T) & c(F,T) # entry wise AND
c(T,T) && c(F,T) # regular AND is not vectorized
T || F # regular OR
c(T,F) | c(F,F) # entry wise OR
xor(T,F) # regular XOR
xor(c(T,F), c(T,T)) # entry wise XOR

# some R "specials"
NULL # e.g. a reference to something not (yet) existing or not applicable
NA # "not available" --> information that is missing e.g. a feature that was not measures for a given sample
Inf # infitinte
NaN # not a number

# !! PITFALL !!
NA == NA # you cannot check if something is NA by using ==
# use is.na() instead
is.na(NA) # better
is.na('NA')
is.na(1)
is.na(NULL) 

# when later working with data.frames containing NA we can exclude such samples using na.exclude() and similar functions.

NaN == NaN # same problem...
is.nan(NaN) # better

NULL==NULL # same problem...
is.null(NULL) # better

Inf == Inf # works!
is.infinite(Inf) # still better...

########################################################
# DATA FRAMES, CSV FILES
########################################################

# Data frame = 2D matrix data representation -- designed for data analysis
# Can contain numeric, string, factor (=enum) values
# E.g. data loaded from csv file 
# Usually: 1 sample = 1 row, 1 feature = 1 column
# Therefore: data.frames are for 2D data, for 3D or more use matrix

# 2D matrix example
1:12
myMatrix <- matrix(1:12, nrow=4, byrow = T)
myMatrix
myMatrix <- matrix(1:12, nrow=4, bycol = F)
myMatrix

# data.frame example
heisenberg <- read.csv('http://www.cyclismo.org/tutorial/R/_static/simple.csv')

# Actually it is small enough to print it completely
heisenberg
# Look at names of variables = features = columns
names(heisenberg)
# Look at the stucture
str(heisenberg)
# Look at the summary
summary(heisenberg)

# Dimensions of data.frame: amount of samples = rows, amount of features = columns
dim(heisenberg)

# We can access data frames like a 2D matrix
heisenberg[1,1] # first row, first column
heisenberg[1,] # first row = first sample, all elements
heisenberg[,1] # all elements, first column = first feature
heisenberg[,] # all samples with all features, equal to just 'heisenberg'

# We can access data frames using variable names too
heisenberg$trial
heisenberg[['trial']]

# subset dataframe to where certain conditions are fulfilled
# subset of 1 variable only
heisenberg$mass
heisenberg$mass[heisenberg$velocity > 11]
# subset of complete dataset containing specific samples
heisenberg[heisenberg$mass > 6,]
heisenberg[heisenberg$mass > 6 & heisenberg$velocity < 14,]
# careful with indexing very big data sets --> possibly slow (use data.table then)
# subset of complete dataset containing specific features
heisenberg
heisenberg[,1]
heisenberg[,1:2]
heisenberg[,c(1,3)]
heisenberg[,c('trial', 'velocity')]

# Factors are discrete variables = variables that can only take a certain amount of values
# This is what is e.g. called Enum in other languages
heisenberg$trial
# Using factors can help in preventing errors (e.g. assigning not allowd values causes a warning + NA assignment)
heisenberg$trial[1] <- 'a'
heisenberg
# How to create factors?
factor(x = 1:10) # takes a vector and defines level from its unique entries 
factor(x = c('a', 'b', 'c')) # levels can be created from numerics, chars, ...

# Type conversion
c(1,3,5,7,9) # numeric vector
as.character(c(1,3,5,7,9)) # ... cast to charcter
as.factor(c(1,3,5,7,9)) # ... cast to factor
as.factor(as.character(c(1,3,5,7,9))) # cast to character, then factor
as.character(as.factor(as.character(c(1,3,5,7,9)))) # cast to character, then factor, then back 
# !! PITFALL !! casting factors to numerics numbers them 1:n or n:1, depeding on alphabetical order
factor(c(9,7,5,3))
as.numeric(factor(c(9,7,5,3))) 
# ... cast to character first!
as.numeric(as.character(factor(c(9,7,5,3))))

# How to create data frames?
# foo, bar, myFactor, isMLCool become features of data.frame
# dimensions must agree or be of length 1
myDf <- data.frame(foo=1:10, 
                   bar=(1:10)**2,  # ** = ^ = to the power of
                   myFactor=factor(1:10 %% 2), 
                   isMLCool='yes')
myDf
str(myDf)
# add one more feature
myDf$myOtherVar <- 20:29
myDf
# overwrite feature
myDf$myOtherVar <- 1:10 
myDf

# data sets provided with R
data() 
# load famous iris data set 
data(iris) 
# ...is now in workspace
# what is it?
iris
dim(iris)
names(iris)
str(iris)
summary(iris)

# how to store/load such data.frames to .csv?
# ?write.table, ?write.csv, ?read.table, ?read.csv
# csv
write.table(x = iris, file = 'iris.csv', append = F, quote = F, sep = ',', row.names = F, col.names = T)
# csv.gz
write.table(x = iris, file = gzfile('iris.csv.gz'), append = F, quote = F, sep = ',', row.names = F, col.names = T)
# show in directory listing
dir('.', pattern = 'csv') 
# load
myIris <- read.table(file = 'iris.csv', sep = ',')
myIris <- read.table(file = 'iris.csv.gz', sep = ',')

# we can also store data.frames and non-data.frame R objects into RData files alike:
saveRDS(object = myIris, file = 'iris.RData')
rm(myIris) # remove variable from workspace
myIris <- readRDS('iris.RData')

# Get current working directory (not neccessarily the script's location)
getwd()
# Set current working directory
setwd('...')

########################################################
# LIBRARIES and DATA VISUALIZATION
########################################################

# Install new libraries
install.packages('lattice')
install.packages('caret')
# If a package is not availabinstall.packages('lattice')le for your R version, try:
# install.packages(PACKAGENAME, dependencies=TRUE, repos='http://cran.rstudio.com/')
# install.packages(PACKAGENAME, repos='http://cran.cnr.berkeley.edu')

# Load libraries
library(lattice) # non-standard plotting library
library(caret) # mighty ML toolset

# There are 3 different, important plotting systems in R: base, lattice and ggplot2
# Base contains basic functions like plot, boxplot, etc
# Lattice contains same functionality + some additional functionality like densityplot
# ggplot2: "grammar of graphics": more advanced, good-looking plots, not covered in this course

?plot
plot(5:10)
plot(x = 1:6, y = 5:10)
plot(x = 1:6, y = (5:10) %% 2)

# using iris data again
data(iris)
str(iris)
# Frequencies of label, which is a factor
table(iris$Species)

# plotting factors...
plot(iris$Species)
# ... is just a barplot of their frequencies
?barplot
barplot(table(iris$Species))

plot(x = iris$Sepal.Length, y = iris$Sepal.Width) # plot feature1 vs feature2
plot(iris[,1:2]) # more compact...
plot(iris[,1:2], col=iris$Species) # distinguish species by color
plot(iris[,1:2], col=iris$Species, pch=19) # change plot symbol
plot(iris[,1:2], col=iris$Species, pch=19, xlab = 'xlabel', ylab = 'ylabel', main = 'title') # many more parameters...

# plot more features at once
plot(iris[,1:3], col=iris$Species)
# Is the same as with pairs(), but...
pairs(iris[,1:3], col=iris$Species)
# ...pairs has different parameters
pairs(iris[,1:3], col=iris$Species, upper.panel = NULL)
pairs(iris[,1:4], col=iris$Species, upper.panel = NULL)
# change plot symbol to '.' increases speed with plotting huge datasets
pairs(iris[,1:4], col=iris$Species, upper.panel = NULL, pch='.')

# boxplot
?boxplot
boxplot(iris[,1:4])
# histogram
?hist
hist(iris[,1])
hist(iris[,1], breaks=5)
hist(iris[,1], breaks=10)
hist(iris[,1], breaks=20)

# stripplot -- this is from the lattice library
?stripchart
stripchart(iris[,1:4])
stripchart(iris[,1:4], method='jitter')
stripchart(iris[,1:4], method='stack')
stripchart(iris[,1:4], method='stack', vertical=T)
stripchart(iris[,1:4], method='stack', vertical=F, offset = 0.1)

# densityplot -- this is from the lattice library
?densityplot
densityplot(iris[,1])
densityplot(iris[,1], groups = iris$Species)
densityplot(iris[,1], groups = iris$Species, plot.points = F) # speedup with many many samples...

# this plot is from the caret ML toolbox
# powerful visualization of multiple features separated for different classes
featurePlot(iris[,1], iris$Species)
featurePlot(iris[,1:2], iris$Species)
featurePlot(iris[,1:4], iris$Species)
featurePlot(iris[,1:4], iris$Species, col=iris$Species) # add class color
featurePlot(iris[,1:4], iris$Species, col=iris$Species, scales=list(relation='free')) # let features use all space in their box
featurePlot(iris[,1:4], iris$Species, col=1:3, scales=list(relation='free'), plot = 'box') # ... with boxplot
featurePlot(iris[,1:4], iris$Species, col=1:3, scales=list(relation='free'), plot = 'density') # ... with densities
featurePlot(x = iris[,1:4], y = iris$Species, col=1:3, scales=list(relation='free'), plot = 'density', plot.points=F) # ... with densities without points
featurePlot(iris[,1:4], iris$Species, col=1:3, scales=list(relation='free'), plot = 'pairs') # ... in pairs, like with regular plot function

# R speciality: formulas in the form 'Y ~ X'
# Y and X are just vectors, but the formula can be used to handle *both* vectors to a function in 1 variable
# Although it tends to be a bit slower than using X and Y directly, 
#   you are going to see that sometimes with different function interfaces...
str(iris[,2]~iris[,1])
plot(iris[,2]~iris[,1], col = iris$Species, pch = 19)
plot(x = iris[,1], y = iris[,2], col = iris$Species, pch = 19)
# e.g. with boxplot: plot one boxplot per class specified in formula
boxplot(iris[,1]~iris[,5])

# expand.grid, head/tail, and levelplot
# expand.grid creates a data.frame with all possible combinations of variable specified
myGrid <- expand.grid(a=-5:5, b=0:10)
myGrid
# head and tail extract the first or last portion of data - good for e.g. giving examples
head(myGrid)
str(myGrid)
attributes(myGrid) # has some additional attributes "attr" - not important for now
# add a new variable
myGrid$c <- myGrid$a + myGrid$b
head(myGrid)
str(myGrid)
summary(myGrid)
boxplot(myGrid)
# levelplot: powerful plot z values for given x and y coordinate values, uses formula interface
# formula is "Z ~ X * Y", where Z, X and Y are equally long vectors, X and Y become x and y coordinates, and Z become the color
# useful for e.g. performance visualization (Z) over 2 parameters (X, Y)
levelplot(x = c ~ a * b, data = myGrid)
levelplot(x = c ~ a * b, data = myGrid, col.regions = gray(100:0/100)) # many different color schemes available

# saving plots to files
# 1) open file device, e.g. png(...), svg(...)
# 2) print(...) with ... being the plot you want
# 3) close the file device with dev.off()
png('my.png')
print(plot(1:10))
dev.off()
svg('my.svg')
print(densityplot(iris[,1]))
dev.off()
pdf('my.pdf')
print(featurePlot(iris[,1:2], iris$Species))
dev.off()

########################################################
# FUNCTIONAL PROGRAMMING in R
#   ! this is stronly simplified ! 
########################################################

# we can create function objects
myFun <- function(x,y) {
  x*y # last line in function is an implicit return statement
}
# what is myFun?
myFun
str(myFun)
# short forms
myFun <- function(x,y) { x*y }
myFun <- function(x,y) x*y # careful when not using the {} brackets!
# call function
myFun(3,4)

# we can handle functions as parameters to other functions
myFun1 <- function(l) {
  l ** 2 # square each element
}
myFun2 <- function(l, fun) {
  fun(l + 10) # add 10 to each element of l, then call fun on the list
}

# call myFun2 with parameters l = 1:5 and fun = myFun2 
myFun2(l = 1:5, fun = myFun1)

########################################################
# VECTORIZATION in R = how to avoid for, while, ...
#   ! this is heavily simplified ! 
########################################################

# In R we try to work with vectors, lists and data.frames. Therefore:
# a) try to write functions that work with these instead of single objects, and
# b) try to handle data between functions as vector, list, data.frame etc.
# --> this will make your life easier on the long run

# Why? This enables you to use vectorized commands through your code,  instead 
# of cumbersome code like for loops, which are slow anyways. Some examples: 

# some 1D data
myData <- iris[,1]
str(myData)

# EXAMPLE: filter vector to contain only elements > X
# ! DO NOT ! 
myDataFiltered <- c()
for(i in 1:length(myData)) {
  if(myData[i] > 5.5) {
    myDataFiltered <- c(myDataFiltered, myData[i])
  }
}
# you will see how ugly and complex this is once you are used to vectorized and functional operations - which tend to be 1-liners:
# BETTER
# a)
myData[myData > 5.5]
# b) ?Filter
Filter(x = myData, f = function(x) x > 5.5) # An anonymous function is handled as f parameter to Filter

# EXAMPLE: apply 1 operation (square in our case) to each element of list/vector/row of data.frame/...
# ! DO NOT ! 
myDataApplied <- c()
for(i in 1:length(myData)) {
  myDataApplied <- c(myDataApplied, myData[i] ** 2)
}
# BETTER
# a)
myData ** 2 # ** == ^
# b) ?lapply
lapply(X = myData, FUN = function(x) x**2 ) # packed in a list. We can unpack it using "unlist" if data types of list content is compatible:
unlist(lapply(X = myData, FUN = function(x) x**2 ))
# c) ?plyr
library(plyr) # install.packages('plyr')
?plyr # very effective toolset for transforming l,d,a,... with l=list, d=data.frame, a=array, ...
# highspeed, can be paralellized
laply(.data = myData, .fun = function(x) x ** 2)

# apply stuff per row/column of data.frames
# = 2D matrix ops as 1-liners
# some 4D data
myData <- iris[,1:4]
str(myData)
# EXAMPLE: median per column
# a) ?apply
# MARGIN 2 = applied on each column
apply(X = myData, MARGIN = 2, FUN = function(col) {
  median(col)
})
apply(X = myData, MAR = 2, FUN = median) 
# b) ?plyr
#   first letter stand for input data type, second letter stand for output data type. methods for all combinations of those data types exist
#   plyr can work with arrays, lists, data.frames and more, you can even ommit input/output using _
#   plyr is fast by default and further be parallelized
#   I would recommend using these!
# array --> array
aaply(.data = as.matrix(myData), .margins = 2, .fun = median)
# array --> dataframe
adply(.data = as.matrix(myData), .margins = 2, .fun = median)
# array --> list
alply(.data = as.matrix(myData), .margins = 2, .fun = median)