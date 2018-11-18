library(ggplot2)
library(lattice) # non-standard plotting library
data(diamonds)

# 2.1
# samples and features
samples <- dim(diamonds)[1]
features <- dim(diamonds)[2]

# data types
str(diamonds)

# balance
plot(diamonds$color) 
plot(diamonds$cut)
plot(diamonds$clarity)

# 2.2
hist(diamonds$price) 
boxplot(diamonds$price)  
densityplot(diamonds$price, plot.points = F)

# Is there a Trend? yes:
# What Trend? Exponential shrinking.
# From which plots can you derive it? The trend can be derived best from the density (and hist) plot.
