# Ensemble Technique

import numpy as np
import scipy
import scipy.io
import sys
import math
from random import sample
import time
import pandas as p
from sklearn import metrics, preprocessing, cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier
import lda
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer

debug = 0


# LDA parameters
numTopics = 1000
numIterations = 10 # Verify

class LemmaTokenizer(object): 
  def __init__(self):
    self.wnl = WordNetLemmatizer()
  def __call__(self, doc):
    return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def convert(val):
  if val == '?':
    return 0
  elif isinstance(val, basestring):
    return float(val)
  else:
    return val


# Check if classes are imbalanced
def isImbalanced(y):
  class0 = 0.0
  class1 = 0.0
  for e in y:
    if e == 0:
      class0 += 1
    else:
      class1 += 1
  if debug == 1:
    print 'Class 0:', class0, 'Class 1:', class1
  if (max(class0, class1) / min(class0, class0) > 3):
    print 'Imbalanced Classes'
  else:
    print 'No Imbalanced Classes'


# To find TF-IDF 
def tfidf(data):
  filename = 'cache/tfidf_'
  try:
    X = scipy.io.mmread(filename + '.mtx').tocsr()
    print "Loaded tfidf model from", filename
  except IOError:
    print filename + " not present. Building TFIDF again."
    tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                          analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=1, smooth_idf=1,
                          sublinear_tf=1, stop_words='english')
    tfv.fit(data)
    X = tfv.transform(data)
    scipy.io.mmwrite(filename, X)
  return X


def documentFrequencyVector(trainData):
  tfv = CountVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                        analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), stop_words='english')
  tfv.fit(trainData)
  X = tfv.transform(trainData)
  return X


def getLdaTopicsVector(X):
  doc_topic = transformInLDATopics(X)
  topicVector = []
  for i in range(len(X)):
    topicVector.append(doc_topic[i].argmax())
  return topicVector


def transformInLDATopics(X):
  filename = 'cache/doc_topic_' + str(numTopics) + '_iterations_' + str(numIterations)
  try:
    doc_topic = np.load(filename + '.npy')
    print "Loaded doc_topic from", filename
    print "doc_topic size", doc_topic.shape
    print "doc_topic type", type(doc_topic)
  except IOError:
    print filename + " not present. Building LDA again."
    X = documentFrequencyVector(X)
    model = lda.LDA(n_topics=numTopics, n_iter=numIterations, random_state=1)
    model.fit(X)  # model.fit_transform(X) is also available
    doc_topic = model.doc_topic_
    np.save(filename, doc_topic)
  return doc_topic


# Function to select the classifier
def selectModel(classifier):
  if classifier == 0:
    print 'Model: Logistic Regression'
    model = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True,
                                  intercept_scaling=1.0, class_weight=None, random_state=None)
  elif classifier == 1:
    print 'Model: Naive Bayes'
    model = MultinomialNB(alpha=1)
  elif classifier == 2:
    print 'Model: Random Forest'
    #model = RandomForestClassifier(n_estimators=10, max_features="auto", max_depth=2, bootstrap=True, n_jobs=4)
    model = RandomForestClassifier(n_estimators=50, max_features="auto", max_depth=2, bootstrap=True, n_jobs=4, criterion="entropy")
    #model = RandomForestClassifier(n_estimators=200, max_features=None, max_depth=None, bootstrap=True, n_jobs=-1, verbose=1)
    #model = RandomForestClassifier(n_estimators=10)
  elif classifier == 3:
    print 'Model: AdaBoost'
    model = AdaBoostClassifier(n_estimators=300)
  else:
    print 'Invalid model: ', classifier
    sys.exit(-1)
  return model


def featureEng(trainFile):
  trainData = np.array(p.read_table(trainFile, converters={'alchemy_category_score': convert, 'is_news': convert,
                                                           'news_front_page': convert}))
  # Delete features which are not useful for classification (eg. URL ID, feature with same value for most of the rows, etc.)
  # exp = experimental
  # Delete URL 0 , URLID 1, BolierPlate 2, Alchemy_Category 3,  embed_ratio 11 mostly 0,  
  # framebased 12 mostly 0, hasDomainLink 14 mostly 0, is_news 17 mostly 1, lengthyLinkDomain 18 (exp), 
  # news_front_page 20 mostly 0,  non_markup_alphanum_characters 21 (exp), 
  # numberOfLinks 22, Label 26
  trainData = np.delete(trainData, [0,1,2,3,11,12,14,17,18,20,26], 1)

  # This fixed the attribute error(while predicting)
  trainData = np.array(trainData, dtype=np.float)
  return trainData


def featureEngTest(testFile):
  testData = np.array(p.read_table(testFile, converters={'alchemy_category_score': convert, 'is_news': convert,
                                                         'news_front_page': convert}))
  # Delete features which are not useful for classification (eg. URL ID, feature with same value for most of the rows, etc.)
  # Delete URL 0 , URLID 1, BolierPlate 2, Alchemy_Category 3, embed_ratio 11 mostly 0,  
  # framebased 12 mostly 0, hasDomainLink 14 mostly 0, Label
  testData = np.delete(testData, [0,1,2,3,11,12,14,17,18,20], 1)

  # This fixed the attribute error(while predicting)
  testData = np.array(testData, dtype=np.float)
  return testData


# Function to find the most important features
def plotFeatureImportances(forest):
  importances = forest.feature_importances_
  std = np.std([tree.feature_importances_ for tree in forest.estimators_],
               axis=0)
  indices = np.argsort(importances)[::-1]

  # Print the feature ranking
  print("Feature ranking:")

  # for f in range(len(indices)):
    # print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

  # Plot the feature importances of the forest
  plt.figure()
  plt.title("Feature importances")
  plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
  plt.xticks(range(len(indices)), indices)
  plt.xlim([-1, len(indices)])
  plt.show()


def genSubmission(model, classifier, X, y, X_test, preds, featureSelection):
  print 'Generating Submission'
  filename = 'benchmark_classifier_' + str(classifier) + '_featureSelection_' + str(featureSelection) + '.csv'
  if classifier == 2:
    #model.fit(X.toarray(), y)
    model.fit(X, y) # toarray is not required after applying SVD
    preds = model.predict_proba(X_test.toarray())[:,1]
  else:
    model.fit(X, y)
    if featureSelection == 1 or featureSelection == 3:
      preds = model.predict_proba(X_test)[:,1]
    else:
      preds = model.predict_proba(testData)[:,1]
  testfile = p.read_csv('test.tsv', sep="\t", na_values=['?'], index_col=1)
  pred_df = p.DataFrame(preds, index=testfile.index, columns=['label'])
  pred_df.to_csv(filename)


def getPredsCV(model, X, target, datasize):
  folds = 10
  cv = cross_validation.KFold(datasize, n_folds=folds)

  finalPreds = []
  fold = 0
  for traincv, testcv in cv:
    X_train, X_cv, y_train, y_cv = X[traincv], X[testcv], target[traincv], target[testcv]
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_cv)[:, 1]
    finalPreds.extend(preds)
    fold = fold + 1
  return finalPreds


def getPredsForTest(model, X, target, testData):
  model.fit(X, target)
  preds = model.predict_proba(testData)[:, 1]
  return preds


def egon(rawData, trainFile, testFile, classifier, featureSelection):
  # Feature Engineering and Formatting Data
  trainData = featureEng(trainFile)
  print 'Train Data shape without boilerplate: ', trainData.shape
  if debug == 1:
    print "trainData shape:", trainData.shape
  # For Submission
  testData = featureEngTest(testFile)

  # TFIDF on boilerplate
  trainDataBoilerplate = list(np.array(p.read_table('train.tsv'))[:,2])
  testDataBoilerplate = list(np.array(p.read_table('test.tsv'))[:,2])
  
  lentrain = len(trainDataBoilerplate)
  y = list(np.array(p.read_table('train.tsv'))[:,-1])
  target = np.array(p.read_table('train.tsv'))[:,-1]
  isImbalanced(y)

  model = selectModel(classifier)

  print "Feature selection type: TFIDF on boiler plate"
  X_tfidf = tfidf(trainDataBoilerplate + testDataBoilerplate)
  '''
  if classifier != 1: # Do not use SVD for NB
    #SVD
    print 'Applying SVD'
    svd = TruncatedSVD(n_components=300, random_state=42)
    svd.fit(X_all)
    X_tfidf = svd.transform(X_all)
  '''
  print 'boilerplate train+test data shape: ', X_tfidf.shape
  X_tfidf_train = X_tfidf[:lentrain]
  X_tfidf_test = X_tfidf[lentrain:]

  '''
  print "Feature selection type: Non boiler plate features + topics extracted using LDA"
  topicsVector = getLdaTopicsVector(trainDataBoilerplate + testDataBoilerplate)
  t = np.array(topicsVector)[np.newaxis].T
  trainData = np.append(trainData, t[:lentrain], 1)
  X_lda_other = trainData
  '''
  '''
  print "Feature selection type: topics extracted using LDA"
  #X = transformInLDATopics(trainDataBoilerplate)
  X_lda = transformInLDATopics(trainDataBoilerplate + testDataBoilerplate)
  X_lda_train = X_lda[:lentrain]
  X_lda_test = X_lda[lentrain:]
  '''


  preds1 = getPredsCV(selectModel(0), X_tfidf_train, target, X_tfidf_train.shape[0])
  preds2 = getPredsCV(selectModel(0), X_tfidf_train, target, X_tfidf_train.shape[0])

  trainB = np.zeros([trainData.shape[0], 2])
  trainB[:,0] = preds1
  trainB[:,1] = preds2

  print "trainB size", trainB.shape

  preds1 = getPredsForTest(model, X_tfidf_train, target, X_tfidf_test)
  preds2 = getPredsForTest(model, X_tfidf_train, target, X_tfidf_test)
  testB = np.zeros([testData.shape[0], 2])
  testB[:,0] = preds1
  testB[:,1] = preds2
  print "testB size", testB.shape

  lr_model = selectModel(0)

  if debug == 1:
    print "===== Starting k fold validation ====="
  aucAvg = 0.0
  folds = 10
  cv = cross_validation.KFold(len(trainData), n_folds=folds)
  fold = 0
  for traincv, testcv in cv:
    X_train, X_cv, y_train, y_cv = trainB[traincv], trainB[testcv], target[traincv], target[testcv]
    lr_model.fit(X_train, y_train)
    preds = lr_model.predict_proba(X_cv)[:, 1]
    fold = fold + 1
    auc = metrics.roc_auc_score(y_cv, preds)
    if debug == 1:
      print "Fold:", fold, " Auc:", auc
    aucAvg += auc
  aucAvg = aucAvg / folds
  print fold, "fold AUC : %f" % aucAvg


  '''
  # Feature importances

  if debug == 1:
    print "===== Starting random split ====="
  aucAvg = 0.0
  numcross = 10
  for i in xrange(numcross):
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_cv)[:, 1]
    auc = metrics.roc_auc_score(y_cv, preds)
    if debug == 1:
      print "NumCross:", i, " Auc:", auc
    aucAvg += auc
  aucAvg = aucAvg / numcross
  print numcross, "Random split AUC : %f" % aucAvg

  # plotFeatureImportances(model)

  if debug == 1:
    print "===== Starting k fold validation ====="
  aucAvg = 0.0
  folds = 10
  cv = cross_validation.KFold(len(trainData), n_folds=folds)
  fold = 0
  for traincv, testcv in cv:
    X_train, X_cv, y_train, y_cv = X[traincv], X[testcv], target[traincv], target[testcv]
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_cv)[:, 1]
    fold = fold + 1
    auc = metrics.roc_auc_score(y_cv, preds)
    if debug == 1:
      print "Fold:", fold, " Auc:", auc
    aucAvg += auc
  aucAvg = aucAvg / folds
  print fold, "fold AUC : %f" % aucAvg

  # Feature importances
  #if classifier == 2: 
    #plotFeatureImportances(model)

  
  # For Submission
  genSubmission(model, classifier, X, y, X_test, preds, featureSelection)
  '''


def main():
  global debug
  if len(sys.argv) < 6:
    print 'Usage: python main.py <rawData> <trainFile> <testFile> <classifier - 0-lr, 1-nb, 2-rf, 3-ab>'
    print '<feature selection 0-useOtherFeatures, 1-only Boilerplate, , 2-userOtherFeatures + LDA, 3-LDA>'
    sys.exit(2)
  if len(sys.argv) == 7:
    debug = int(sys.argv[6])
  egon(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))


main()
