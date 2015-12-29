## Mariah Harvey
## December 26, 2015
## Guidlines for picking hidden units: http://cowlet.org/2014/01/12/understanding-data-science-classification-with-neural-networks-in-r.html

## load packages
library(nnet)

##read in data
Hdma <- read.csv("Hdma.csv")

## set seed
set.seed(45)

## examine data
dim(Hdma)
head(Hdma)
summary(Hdma)
table(Hdma$deny)
prop.table(table(Hdma$deny))


## Hdma is a dataframe containing the following variables:
# dir - debt payments to total income ratio
# hir - housing expenses to income ratio
# lvr - ratio of size of loan to assessed value of property
# ccs - consumer credit score from 1 to 6 (a low value being a good score)
# mcs - mortgage credit score from 1 to 4 (a low value being a good score)
# pbcr - public bad credit record?
# dmi - denied mortgage insurance?
# self - self employed?
# single - is the applicant single?
# uria - 1989 Massachusetts unemployment rate in the applicant’s industry
# condo - is unit condominium?
# black is the applicant black?
# deny mortgage application denied?

### Data Source
# Federal Reserve Bank of Boston.
# Munnell, Alicia H., Geoffrey M.B. Tootell, Lynne E. Browne and James McEneaney (1996) “Mortgage lending in Boston: Interpreting HDMA data”, American Economic Review, 25-53.


## change categorical variables into factors
Hdma$deny <- factor(Hdma$deny, levels = c("yes","no"))
Hdma$pbcr <- factor(Hdma$pbcr, levels = c("yes","no"))
Hdma$dmi <- factor(Hdma$dmi, levels = c("yes","no"))
Hdma$self <- factor(Hdma$self, levels = c("yes","no"))
Hdma$single <- factor(Hdma$single, levels = c("yes","no"))
Hdma$black <- factor(Hdma$black, levels = c("yes","no"))
Hdma$deny <- factor(Hdma$deny, levels = c("yes","no"))

## Randomly separate train (80 percent) and hold out data (20 percent):

nrow(Hdma)
sample_rows<- sample(1:nrow(Hdma), size = round(0.8*nrow(Hdma)))

train <- Hdma[sample_rows,]
hold_out <- Hdma[-sample_rows,]

## Randomly separate training (80 percent) and testing (20 percent):

cross_validate<- sample(1:nrow(train), size = round(0.8*nrow(train)))
training<-train[cross_validate,]
testing<-train[-cross_validate,]
                       

dim(train)
dim(hold_out)
dim(training)
dim(testing)

## calculating number of hidden neurons to use, h
r<-round(nrow(training)*(1/30))
hidden_units<-c(30, 50, r, 70)

results <- data.frame(pred=character())
models <- list()

for (h in hidden_units)
{
  tempname<-paste(h, sep="-")
  print(tempname)
  
  fitnn<-nnet(deny ~ dir+hir+lvr+ccs+mcs+pbcr+dmi+self+single+uria+condo+black, data=train, size=h, decay=5e-4, maxit=200)
  models[[tempname]] <- fitnn
  pred <- predict(fitnn, newdata=testing, type="class")
  results <- rbind(results, data.frame(pred))
}

testing$h1<-results[1:376,]
testing$h2<-results[377:752,]
testing$h3<-results[753:1128,]
testing$h4<-results[1129:1504,]

## compute confusion matrix where hidden_units=c(30, 50, r=63, 70)
table(testing$deny, testing$h1)
table(testing$deny, testing$h2)
table(testing$deny, testing$h3)
table(testing$deny, testing$h4)

## A model with 63 hidden units works best according to the confusion matrix

## Final test on hold out group

fitnn_final<-nnet(deny ~ dir+hir+lvr+ccs+mcs+pbcr+dmi+self+single+uria+condo+black, data=train, size=63, decay=5e-4, maxit=200)
prediction <- predict(fitnn_final, newdata=hold_out, type="class")
hold_out <- cbind(hold_out, prediction)

table(hold_out$deny, hold_out$prediction)


