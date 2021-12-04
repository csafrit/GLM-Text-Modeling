library(quanteda)
library(quanteda.textplots)
library(quanteda.textstats)
library(quanteda.textmodels)
library(stopwords)
library(topicmodels)
library(tidytext)
library(ggplot2)
library(dplyr)
library(caret)
library(e1071)
library(car)
library(pROC)
library(tree)
library(ISLR)
library(rpart)
library(rpart.plot)

dfCS <- read.csv('~/_DSBA_UNCC/DBSA_6211_Advanced_Business_and_Analytics/gastext.csv', stringsAsFactors = F)
View(dfCS)
summary(dfCS)

table(dfCS$Loyal_Status)

myCorpusCS <- corpus(dfCS$Comment)
summary(myCorpusCS)

#Preprocessing

myDfmCS <- dfm(tokens(myCorpusCS))
dim(myDfmCS)

tstat_freqCS <- textstat_frequency(myDfmCS)
head(tstat_freqCS,20)

myDfmCS <- dfm(tokens(myCorpusCS,
                    remove_punct=T))
myDfmCS <- dfm_remove(myDfmCS, stopwords('english'))
myDfmCS <- dfm_wordstem(myDfmCS)


myDfmCS %>% 
  textstat_frequency(n = 40) %>% 
  ggplot(aes(x = reorder(feature, frequency), y = frequency)) +
  geom_point() +
  labs(x = NULL, y = "Frequency") +
  theme_minimal()

stopwords1 <- c('can', 'cant', 'well', 'just', 'also', 'wont', 'much', 'without', 'per', 'seem','set', 'lot','get')
myDfmCS <- dfm_remove(myDfmCS, stopwords1)
topfeatures(myDfmCS, 50)

#Wordcloud

textplot_wordcloud(myDfmCS, max_words = 200)

#Similar terms

term_simprice <- textstat_simil(myDfmCS,
                           selection = 'price',
                           margin = 'feature',
                           method = 'correlation')
as.list(term_simprice,n=5)

term_simservice <- textstat_simil(myDfmCS,
                                selection = 'servic',
                                margin = 'feature',
                                method = 'correlation')
as.list(term_simservice,n=5)

myDfmCS <- dfm_remove(myDfmCS, c('drink','clean','shower','get', 'use', 'productx','servic',2,1,'sure','alway','anyth','even','take','price'))
stopwords2 <- c('drink','clean','shower','get', 'use', 'productx','servic',2,1,'sure','alway','anyth','even','take','price')
myDfmCS <- as.matrix(myDfmCS)
myDfmCS <-myDfmCS[which(rowSums(myDfmCS)>0),]
myDfmCS <- as.dfm(myDfmCS)

#topic modeling

myLdaCS <- LDA(myDfmCS,k=4,control=list(seed=101))
myLdaCS

# Term-topic probabilities
myLdaCS_td <- tidy(myLdaCS)
myLdaCS_td
View(myLdaCS_td)

top_termsCS <- myLdaCS_td %>%
  group_by(topic) %>%
  top_n(4, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

top_termsCS %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

# View topic 4 terms in each topic
Lda_termCS<-as.matrix(terms(myLdaCS,4))
View(Lda_termCS)

# Document-topic probabilities
ap_documentsCS <- tidy(myLdaCS, matrix = "gamma")
ap_documentsCS

# View document-topic probabilities in a table
Lda_documentCS<-as.data.frame(myLdaCS@gamma)
View(Lda_documentCS)

#Model1 Uses non-text info
dfCS1 = select(dfCS,-1,-2)
View(dfCS1)
str(dfCS1)
summary(dfCS1)
# Change DV data type
dfCS1$NewCust_Flag <- factor(dfCS1$NewCust_Flag)
dfCS1$Comp_card_flag <- factor(dfCS1$Comp_card_flag)
dfCS1$AcctType_flag <- factor(dfCS1$AcctType_flag)
dfCS1$Contact_Flag2 <- factor(dfCS1$Contact_Flag2)
dfCS1$HQ_flag <- factor(dfCS1$HQ_flag)
dfCS1$Contact_flag <- factor(dfCS1$Contact_flag)
dfCS1$new_flag <- factor(dfCS1$new_flag)
dfCS1$Choice_flag <- factor(dfCS1$Choice_flag)
dfCS1$Multi_flag <- factor(dfCS1$Multi_flag)
dfCS1$Service_flag <- factor(dfCS1$Service_flag)
dfCS1$CustType_flag <- factor(dfCS1$CustType_flag)
dfCS1$Target <- factor(dfCS1$Target)
dfCS1$Loyal_Status <- factor(dfCS1$Loyal_Status)

# Using VIF to test multicollinearity
vif(glm(formula=Target ~ . , family = binomial(link='logit'),data = dfCS1))

# Data partition with the Caret package
# Set a random see so your "random" results are the same as me (optional)
set.seed(101)
trainIndex <- createDataPartition(dfCS1$Target,
                                  p=0.7,
                                  list=FALSE,
                                  times=1)


# Create Training Data
df.trainCS1 <- dfCS1[trainIndex,]

# Create Validation Data
df.validCS1 <-dfCS1[-trainIndex,]
# Run a very simple baseline model with the training dataset
baseline.model1 <- train(Target~.,
                        data=df.trainCS1,
                        method='glm',
                        family='binomial',
                        na.action=na.pass)


# View the model results
summary(baseline.model1)

#Evaluation model performance using the validation dataset

#Criteria 1: the confusion matrix
predictionCS1 <- predict(baseline.model1,newdata=df.validCS1)

#Need to remove missing values from the validation dataset for evaluation
df.valid.nonmissingCS1 <- na.omit(df.validCS1)

confusionMatrix(predictionCS1,df.valid.nonmissingCS1$Target)

#Criteria 2: the ROC curve and area under the curve
pred.probabilitiesCS1 <- predict(baseline.model1,newdata=df.validCS1,type='prob')

regression.ROCCS1 <- roc(predictor=pred.probabilitiesCS1$`1`,
                      response=df.valid.nonmissingCS1$Target,
                      levels=levels(df.valid.nonmissingCS1$Target))
plot(regression.ROCCS1)
regression.ROCCS1$auc

treeCS1 <- train(Target~.,
                data=df.trainCS1,
                method='rpart',
                na.action=na.pass)
treeCS1

prp(treeCS1$finalModel,type=2,extra=106)

predictionCS1tree <- predict(treeCS1, newdata=df.validCS1,na.action=na.pass)
confusionMatrix(predictionCS1tree, df.validCS1$Target)

tree.probabilities.CS1 <- predict(treeCS1,
                                 newdata=df.validCS1,
                                 type='prob',
                                 na.action=na.pass)
tree.ROC.CS1 <- roc(predictor=tree.probabilities.CS1$`1`,
                   response=df.validCS1$Target,
                   levels=levels(df.validCS1$Target))
plot(tree.ROC.CS1)
tree.ROC.CS1$auc


#Model2 Use non-text and text info
#Preprocessing Model 2


modelDfmCS2 <- dfm(tokens(myCorpusCS,
                       remove_punct=T))
modelDfmCS2 <- dfm_remove(modelDfmCS2, stopwords('english'))
modelDfmCS2 <- dfm_remove(modelDfmCS2, stopwords1)
modelDfmCS2 <- dfm_wordstem(modelDfmCS2)

modelDfmCS2 <- dfm_trim(modelDfm, min_termfreq = 4, min_docfreq = 2)
dim(modelDfmCS2)

modelDfm_tfidfCS2 <- dfm_tfidf(modelDfmCS2)
dim(modelDfm_tfidfCS2)

modelSvdCS2<-textmodel_lsa(modelDfm_tfidfCS2,nd=8)
head(modelSvdCS2$docs)
 

modelData2 <- cbind(dfCS1, as.data.frame(modelSvdCS2$docs))

# Set a random see so your "random" results are the same as me (optional)
set.seed(101)
trainIndex2 <- createDataPartition(modelData2$Target,
                                  p=0.7,
                                  list=FALSE,
                                  times=1)


# Create Training Data
df.trainCS2 <- modelData2[trainIndex2,]

# Create Validation Data
df.validCS2 <-modelData2[-trainIndex,]
# Run a very simple baseline model with the training dataset
baseline.model2 <- train(Target~.,
                         data=df.trainCS2,
                         method='glm',
                         family='binomial',
                         na.action=na.pass)


# View the model results
summary(baseline.model2)

#Evaluation model performance using the validation dataset

#Criteria 1: the confusion matrix
predictionCS2 <- predict(baseline.model2,newdata=df.validCS2)

#Need to remove missing values from the validation dataset for evaluation
df.valid.nonmissingCS2 <- na.omit(df.validCS2)

confusionMatrix(predictionCS2,df.valid.nonmissingCS2$Target)

#Criteria 2: the ROC curve and area under the curve
pred.probabilitiesCS2 <- predict(baseline.model2,newdata=df.validCS2,type='prob')

regression.ROCCS2 <- roc(predictor=pred.probabilitiesCS2$`1`,
                         response=df.valid.nonmissingCS2$Target,
                         levels=levels(df.valid.nonmissingCS2$Target))
plot(regression.ROCCS2)
regression.ROCCS2$auc

treeCS2 <- train(Target~.,
                 data=df.trainCS2,
                 method='rpart',
                 na.action=na.pass)
treeCS2

prp(treeCS2$finalModel,type=2,extra=106)

predictionCS2tree <- predict(treeCS2, newdata=df.validCS2,na.action=na.pass)
confusionMatrix(predictionCS2tree, df.validCS2$Target)

tree.probabilities.CS2 <- predict(treeCS2,
                                  newdata=df.validCS2,
                                  type='prob',
                                  na.action=na.pass)
tree.ROC.CS2 <- roc(predictor=tree.probabilities.CS2$`1`,
                    response=df.validCS2$Target,
                    levels=levels(df.validCS2$Target))
plot(tree.ROC.CS2)
tree.ROC.CS2$auc

