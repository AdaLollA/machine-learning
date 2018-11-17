library(ggplot2)
data(diamonds)

# 2.7
# amount of diamonds with value greater than 9500
nrow(diamonds[diamonds["price"] > 9500,])

# amount of diamonds with value greater than 9500 and color 'D'
nrow(diamonds[diamonds["price"] > 9500 & diamonds["color"] == 'D',])

# summary of carat and price for diamonds with color 'D' and cut 'Fair'
d_fair <- diamonds[diamonds["color"] == 'D' & diamonds["cut"] == 'Fair',]
summary(d_fair[1])
summary(d_fair[7])

# summary of carat and price for diamonds with color 'J' and cut 'Ideal'
j_ideal <- diamonds[diamonds["color"] == 'J' & diamonds["cut"] == 'Ideal',]
summary(j_ideal[1])
summary(j_ideal[7])

# 2.8
# mean value of all numeric features
sapply(diamonds, function(x) if (!is.numeric(x)) NA else mean(x))

# median value of all numeric features
sapply(diamonds, function(x) if (!is.numeric(x)) NA else median(x))

# standard deviation of all numeric features
sapply(diamonds, function(x) if (!is.numeric(x)) NA else sd(x))

# median absolute deviation with constant = 1 of all numeric features
sapply(diamonds, function(x) if (!is.numeric(x)) NA else mad(x, constant = 1))
