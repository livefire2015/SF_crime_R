rm(list=ls())
gc()
library("pacman")
pacman::p_load(LiblineaR, MLmetrics, doParallel, glmnet, vcd, MASS, caret, plyr, Amelia, data.table, reshape2, readr, lubridate, cluster)

######READ DATA##############
readData <- function(path.name, file.name, column.types, missing.types) {
  read.csv(paste(path.name, file.name, sep=""),
           colClasses=column.types,
           na.strings=missing.types)
}

parseDate <- function(df) {
  DateTime <- strptime(df$Dates, "%Y-%m-%d %H:%M:%S")
  df$Year <- as.numeric(format(ymd_hms(DateTime), "%Y"))
  df$Month <- as.numeric(format(ymd_hms(DateTime), "%m"))
  df$Day <- as.numeric(format(ymd_hms(DateTime), "%d"))
  df$Hour <- as.numeric(format(ymd_hms(DateTime), "%H"))
  return(df)
}

SF.path <- "C:/Users/Yangye_Zhu/Box Sync/Data_Science/machine_learning/SF_Crime/"
train.data.file <- "train.csv"
test.data.file <- "test.csv"
train.column.types <- c(
  'character', # Dates 
  'factor', # Category 
  'character', # Descript 
  'factor', # DayOfWeek 
  'factor', # PdDistrict 
  'factor', # Resolution 
  'character', # Address 
  'numeric', # X Longitude
  'numeric' # Y Latitude
)
test.column.types <- c(
  'integer', # Id
  'character', # Dates 
  'factor', # DayOfWeek 
  'factor', # PdDistrict 
  'character', # Address 
  'numeric', # X Longitude
  'numeric' # Y Latitude
)
missing.types <- c("NA","") # Note that readr read_csv na requires String only, not c("NA", "")

train.raw <- readData(SF.path, train.data.file, train.column.types, missing.types)
df.train <- parseDate(train.raw)
test.raw <- readData(SF.path, test.data.file, test.column.types, missing.types)
df.infer <- parseDate(test.raw)
#######READ DATA#############

# load("SF_df_train.RData")
# load("SF_df_infer.RData")

#######DATA VISUALIZATION#########
barplot(table(df.train$Category), 
        main="Category Statistics", 
        ylab="Count", 
        las=2, 
        horiz=F, 
        col=rainbow(nrow(table(df.train$Category))),
        cex.axis = 0.75,
        cex.names = 0.75)
barplot(table(df.train$DayOfWeek), 
        main="DayOfWeek Statistics", 
        ylab="Count", 
        las=2, 
        horiz=F, 
        col=rainbow(nrow(table(df.train$DayOfWeek))),
        cex.axis = 0.75,
        cex.names = 0.75)
barplot(table(df.train$PdDistrict), 
        main="PdDistrict Statistics", 
        ylab="Count", 
        las=2, 
        horiz=F, 
        col=rainbow(nrow(table(df.train$PdDistrict))),
        cex.axis = 0.75,
        cex.names = 0.75)
barplot(table(df.train$Resolution), 
        main="Resolution Statistics", 
        ylab="Count", 
        las=2, 
        horiz=F, 
        col=rainbow(nrow(table(df.train$Resolution))),
        cex.axis = 0.75,
        cex.names = 0.75)
hist(df.train$Year, main="Year", xlab=NULL, col="brown")
hist(df.train$Month, main="Month", xlab=NULL, col="blue")
hist(df.train$Day, main="Day", xlab=NULL, col="brown")
hist(df.train$Hour, main="Hour", xlab=NULL, col="blue")
boxplot(df.train$X ~ df.train$Category,
        main="Category by Longitude", 
        xlab="Category", 
        ylab="Longitude",
        las = 2)
boxplot(df.train$Y ~ df.train$Category,
        main="Category by Latitude", 
        xlab="Category", 
        ylab="Latitude",
        las = 2)
boxplot(df.train$Coordinate ~ df.train$Category,
        main="Category by Coordinate (original)", 
        xlab="Category", 
        ylab="Coordinate",
        las = 2)
##########DATA VISUALIZATION##############

##########FEATURE ENGINEERING###############
getStreetName <- function(data) {
  streets.no.block <- gsub("[0-9]{1,} Block of ", "", data$Address)
  streets.cleaned <- strsplit(streets.no.block, " / ")
  f <- function(l) paste(sort(l), collapse = " ")  
  data$StreetName <- sapply(streets.cleaned, f)
  return (data$StreetName)
}

featureEngrg <- function(data) {
  data$CrimeClass <- data$Category
  data$CrimeClass <- revalue(data$CrimeClass, c(
    "ARSON" = "C1",
    "ASSAULT" = "C2",
    "BAD CHECKS" = "C3",
    "BRIBERY" = "C4",
    "BURGLARY" = "C5",
    "DISORDERLY CONDUCT" = "C6",
    "DRIVING UNDER THE INFLUENCE" = "C7",
    "DRUG/NARCOTIC" = "C8",
    "DRUNKENNESS" = "C9",
    "EMBEZZLEMENT" = "C10",
    "EXTORTION" = "C11",
    "FAMILY OFFENSES" = "C12",
    "FORGERY/COUNTERFEITING" = "C13",
    "FRAUD" = "C14",
    "GAMBLING" = "C15",
    "KIDNAPPING"  = "C16",
    "LARCENY/THEFT" = "C17",
    "LIQUOR LAWS"  = "C18",
    "LOITERING"  = "C19",
    "MISSING PERSON"  = "C20",
    "NON-CRIMINAL"  = "C21",
    "OTHER OFFENSES" = "C22",
    "PORNOGRAPHY/OBSCENE MAT" = "C23",
    "PROSTITUTION"  = "C24",
    "RECOVERED VEHICLE" = "C25",
    "ROBBERY" = "C26",
    "RUNAWAY" = "C27",
    "SECONDARY CODES" = "C28",
    "SEX OFFENSES FORCIBLE"  = "C29",
    "SEX OFFENSES NON FORCIBLE"  = "C30",
    "STOLEN PROPERTY" = "C31",
    "SUICIDE" = "C32",
    "SUSPICIOUS OCC" = "C33",
    "TREA" = "C34",
    "TRESPASS" = "C35",
    "VANDALISM" = "C36",
    "VEHICLE THEFT" = "C37",
    "WARRANTS" = "C38",
    "WEAPON LAWS" = "C39"
  ))
  data$Coordinate <- sqrt(data$X ^ 2 + data$Y ^ 2)
  data$StreetName <- getStreetName(data)
  return (data)
}

df.train <- featureEngrg(df.train)
df.train$CrimeClass <- as.factor(df.train$CrimeClass)
df.train$StreetName <- as.factor(df.train$StreetName)
df.train <- df.train[-which(df.train$Coordinate > 130),] # exclude possible outliers

boxplot(df.train$Coordinate ~ df.train$Category,
        main="Category by Coordinate (revised)", 
        xlab="Category", 
        ylab="Coordinate",
        las = 2)
mosaicplot(df.train$DayOfWeek ~ df.train$Category,
           main="Category by DayOfWeek", 
           shade=FALSE, 
           color=TRUE,
           xlab="DayOfWeek", 
           ylab="Category")
mosaicplot(df.train$PdDistrict ~ df.train$Category,
           main="Category by PdDistrict", 
           shade=FALSE, 
           color=TRUE,
           xlab="PdDistrict", 
           ylab="Category")
barplot(table(df.train$StreetName), 
        main="StreetName Statistics", 
        ylab="Count", 
        las=2, 
        horiz=F, 
        col=rainbow(nrow(table(df.train$StreetName))),
        cex.axis = 0.75,
        cex.names = 0.75)
##########FEATURE ENGINEERING###############


##########FIT AND EVALUATE MODELS##############
train.keeps <- c("CrimeClass", "DayOfWeek", "PdDistrict",
                 "X", "Y", "Year", "Month", "Day",
                 "Hour", "Coordinate", "StreetName")
df.train.munged <- df.train[train.keeps]

df.infer$Coordinate <- sqrt(df.infer$X ^ 2 + df.infer$Y ^ 2)
df.infer$StreetName <- getStreetName(df.infer)
df.infer$StreetName <- as.factor(df.infer$StreetName)
test.keeps <- c("DayOfWeek", "PdDistrict",
                 "X", "Y", "Year", "Month", "Day",
                 "Hour", "Coordinate", "StreetName")
df.infer.munged <- df.infer[test.keeps]


# load("SF_df_train_munged.RData")
# load("SF_df_infer_munged.RData")

set.seed(23) # ensure same random sampling
training.rows <- createDataPartition(df.train.munged$CrimeClass, 
                                     p=0.8, list=FALSE)
train.batch <- df.train.munged[training.rows,]
test.batch <- df.train.munged[-training.rows,]

cv.ctrl <- trainControl(method="repeatedcv", 
                        repeats=3, 
                        summaryFunction=mnLogLoss, 
                        classProbs=TRUE)
glmnGrid <- expand.grid(.alpha = (1:10)*0.1,
                        .lambda = (1:10)*0.1)

nnGrid <- expand.grid(.size = seq(1,7,by=2),
                      .decay = c(0,0.1),
                      .bag = T)

# fit a glmnet model using all predictors
cl <- makeCluster(detectCores(), type = 'PSOCK')
registerDoParallel(cl)
set.seed(35)
x <- data.matrix(train.batch[,-1])
y <- train.batch$CrimeClass
glmn.tune.1 <- train(x, 
                     y, 
                     method = "glmnet", 
                     tuneGrid = glmnGrid,
                     preProcess = c("center","scale"),
                     metric = "logLoss",  
                     trControl = cv.ctrl)
stopCluster(cl)

save(glmn.tune.1, file = "SF_glmn_1.RData")

# the test batch of training data is a good indicator of the test set performance
load("SF_glmn_1.RData") 
print(glmn.tune.1)
newx <- data.matrix(test.batch[,-1])
test.batch.pred <- predict(glmn.tune.1$finalModel, newx=newx, s=0.2, type = "response")

y.pred <- matrix(test.batch.pred, 
                 ncol = nlevels(test.batch$CrimeClass), 
                 nrow = nrow(test.batch))
y.true.vec <- strtoi(gsub("C","", test.batch$CrimeClass))
y.true <- matrix(0, ncol = nlevels(test.batch$CrimeClass), 
                 nrow = nrow(test.batch))
for(i in 1:nrow(test.batch)) {
  y.true[i, y.true.vec[i]] <- 1
}
MultiLogLoss(y.true, y.pred)


# fit a lda model using just four predictors
cl <- makeCluster(detectCores(), type = 'PSOCK')
registerDoParallel(cl)
set.seed(35)
x <- data.matrix(train.batch[,c("X","Y","Year","Hour")])  
y <- train.batch$CrimeClass
lda.tune.1 <- train(x, 
                     y, 
                     method = "lda",
                     # tuneLength = 3,
                     preProcess = c("center","scale"),
                     metric = "logLoss",  
                     trControl = cv.ctrl)
stopCluster(cl)

load("SF_lda_1.RData") 
print(lda.tune.1)
newx <- data.matrix(test.batch[,c("X","Y","Year","Hour")])
test.batch.pred <- predict(lda.tune.1, newx, type = "prob")

y.pred <- as.matrix(test.batch.pred)
y.true.vec <- strtoi(gsub("C","", test.batch$CrimeClass))
y.true <- matrix(0, ncol = nlevels(test.batch$CrimeClass), 
                 nrow = nrow(test.batch))
for(i in 1:nrow(test.batch)) {
  y.true[i, y.true.vec[i]] <- 1
}
MultiLogLoss(y.true, y.pred)


# fit a neural network model using just four predictors
predictorsUsed <- c("X", "Y", "Year", "Hour")

maxSize <- max(nnGrid$.size)
numWts <- nlevels(df.train.munged$CrimeClass)*(maxSize * (length(predictorsUsed) + 1) + maxSize + 1)
cl <- makeCluster(detectCores(), type = 'PSOCK')
registerDoParallel(cl)
set.seed(35)
x <- data.matrix(train.batch[,predictorsUsed])
y <- train.batch$CrimeClass
nn.tune.2 <- train(x, 
                    y, 
                    method = "avNNet",
                    tuneGrid = nnGrid,
                    preProcess = c("center","scale", "spatialSign"),
                    metric = "logLoss",  # ROC later
                    trace = F,
                    maxit = 1000,
                    MaxNWts = numWts,
                    trControl = cv.ctrl)
stopCluster(cl)
save(nn.tune.2, file = "SF_nn_2.RData")

load("SF_nn_2.RData") 
print(nn.tune.2)
newx <- data.matrix(test.batch[,predictorsUsed])
test.batch.pred <- predict(nn.tune.2, newx, type = "prob")

y.pred <- as.matrix(test.batch.pred)
y.true.vec <- strtoi(gsub("C","", test.batch$CrimeClass))
y.true <- matrix(0, ncol = nlevels(test.batch$CrimeClass), 
                 nrow = nrow(test.batch))
for(i in 1:nrow(test.batch)) {
  y.true[i, y.true.vec[i]] <- 1
}
MultiLogLoss(y.true, y.pred)
##########FIT AND EVALUATE MODELS##############


#########PRELIMINARY MODEL COMPARISON################
#glmnet: 2.6802
#lda: 2.6168
#nn: 2.6122

#running time: nn > glmnet > lda
#########PRELIMINARY MODEL COMPARISON################


#########REFERENCE#############
#https://github.com/wehrley/wehrley.github.io/blob/master/SOUPTONUTS.md
#########REFERENCE#############