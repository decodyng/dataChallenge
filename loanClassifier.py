from __future__ import division
import requests
import zipfile
import StringIO
import pandas as pd
import cPickle
import numpy as np
import re
from sklearn.svm import LinearSVC
from sklearn.metrics import make_scorer
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction import DictVectorizer

### auxilery functions to convert variables to usable formats, meant to be used in an apply structure
### as well as one to convert a pandas dataframe into sklearn-usable format
def transform_csv(data, target_col=0, ignore_cols=None):
    my_dictionary = {}
    my_dictionary["target"] = data.ix[:,[target_col]].values

    #check for numeric index on target
    try:
        target_col = float(target_col)
        xdata = data.drop(data.columns[target_col], axis=1)
    except ValueError:
        xdata = data.drop(target_col, axis=1)

    #check for numeric index on ignore columns
    try:
        ig = ignore_cols[0]
        try:
            float(ig)
            xdata = xdata.drop(data.columns[ignore_cols], axis=1)
        except ValueError:
            xdata = xdata.drop(ignore_cols, axis=1)
    except TypeError:
        pass
    my_dictionary["data"] = [list(x) for x in xdata.itertuples(index=False)]
    my_dictionary["feature_names"] = xdata.columns.values
    my_dictionary["target_names"] = "target"
    my_dictionary["DESCR"] = "Hi there!"
    return my_dictionary


def evaluationMetric(trueY, predictY):
    """
    :param trueY:
    :param predictY:
    :return: approval-rate subject to repayment > 0.85 constraint
    """
    predictY = np.asarray(predictY)
    trueY = np.asarray(trueY)
    approval = np.sum(predictY)/len(predictY)

    repaymentNum = 0
    repaymentDenom = 0
    for ind, pred in enumerate(predictY):
        if pred == 1 and trueY[ind] == 1:
            repaymentNum += 1
            repaymentDenom += 1
        elif pred == 1:
            repaymentDenom += 1

    repayment = repaymentNum/repaymentDenom
    repaymentOverThreshold = repayment > 0.85
    return approval*repaymentOverThreshold

def nMonths(varStr):
    """
    :param varStr: a string representation of unit of time following the pattern [digit] [years/months]
    :return: numeric month-representation of unit of time
    """
    if varStr == '< 1 year':
        return 0
    elif varStr == 'n/a':
        return -1
    else:
        yrMatch = re.match('(\d+)\+? years?', varStr)
        mnthMatch = re.match('(\d+)\+? months', varStr)
        if yrMatch:
            return int(yrMatch.group(1))*12
        elif mnthMatch:
            return int(mnthMatch.group(1))
        else:
            return 0

def percToDecimal(varStr):
    """
    :param varStr: string representation of percentage (i.e. 54.76%
    :return: numeric representation: i.e. .5476
    """
    try:
        percMatch = re.match('([\d\.]+)%', varStr)
    except:
        #in case of NAN
        return varStr
    if percMatch:
        return float(percMatch.group(1))*.01

def descriptionLength(varStr):
    """
    :param varStr: description given by user
    :return: length of description, including zero-length for empty string or null value
    """
    if type(varStr) == float and np.isnan(varStr):
        return 0
    else:
        return len(varStr)

def monthsSince(varNum):
    """
    :param varNum: number or NA representing months since an event
    :return: number projecting months since event onto 0-1 space
    """
    #transforms 'months since X event' into a usable and comparable format
    #by designating NA - assumed to be no record of said event, to 0, and then
    #transforming months since to 1/monthsince, so that the more months since
    #an event has occured, the more similar it is to zero, i.e. an event not being
    #on record as occuring
    if np.isnan(varNum):
        return 0
    elif varNum == 0:
        return 1
    else:
        return 1/float(varNum)





##functions in this section are somewhat higher-level than the prior section: structured to manipulate data,
# adding or converting columns, and then return modified dataframe

def binarizeTarget(dat, style='noCurrent', currentThreshold=.10):
    """
    :param dat: pandas dataframe
    :param style: string specifying style-type
    :param currentThreshold: decimal
    :return: pandas dataframe with target variable
    """
    #handles different potential varieties of target binarization
    #'noCurrent':  ignores 'Current' loans entirely from training data
    #'allCurrent': includes all 'Current' loans as positive examples
    #'thresholdCurrent': includes all current loans where percent of remaining principal is less than a threshold amt

    #remove null values of target:
    dat = dat[pd.notnull(dat['loan_status'])]
    if style =='noCurrent':
        dat = dat[dat['loan_status'] != 'Current']
        dat['binaryTarget'] = 0
        dat.loc[dat['loan_status'] == 'Fully Paid', 'binaryTarget'] = 1
    elif style =='allCurrent':
        dat['binaryTarget'] = 0
        dat.loc[dat['loan_status'] == 'Fully Paid', 'binaryTarget'] = 1
        dat.loc[dat['loan_status'] == 'Current', 'binaryTarget'] = 1

    elif style == 'thresholdCurrent':
        dat['belowCurrentThreshold'] = dat['out_prncp']/dat['funded_amnt'] < currentThreshold
        dat = dat[dat['belowCurrentThreshold' or dat['loan_status'] != 'Current']]
        dat['binaryTarget'] = 0
        dat.loc[dat['loan_status'] == 'Fully Paid', 'binaryTarget'] = 1
        dat.loc[dat['belowCurrentThreshold'], 'binaryTarget'] = 1
    return dat


def convertDates(dat):
    """

    :param dat: pandas dataframe with string dates
    :return: pandas dataframe with datetime dates
    """
    #issue_d, earliest_cr_line, last_pymnt_d
    dat['issue_d'] = pd.to_datetime(dat['issue_d'], format='%b-%Y')
    dat['earliest_cr_line'] = pd.to_datetime(dat['earliest_cr_line'], format='%b-%Y')
    dat['last_pymnt_d'] = pd.to_datetime(dat['last_pymnt_d'], format='%b-%Y')
    dat['next_pymnt_d'] = pd.to_datetime(dat['next_pymnt_d'], format='%b-%Y')
    dat['last_credit_pull_d'] = pd.to_datetime(dat['last_credit_pull_d'], format='%b-%Y')
    return dat



def binarizeCategorical(dat, test=False, dv=None):
    """

    :param dat: pandas dataframe
    :param test: boolean indicating whether we're on the train or test
    :param dv: DictVectorizer, passed to function if in 'test' mode to ensure compatibility with train vectorization
    :return: pandas dataframe
    """
    categoricalColumns = ['sub_grade', 'home_ownership', 'is_inc_v', 'purpose', 'initial_list_status', 'pymnt_plan']
    ids = dat['id']
    categDat = dat[categoricalColumns]
    noCategDat = dat.drop(categoricalColumns, axis=1)
    categDat = categDat.fillna('NA')
    categDatDict = categDat.to_dict(orient='records')
    if test:
        categOneHot = dv.transform(categDatDict)
    else:
        dv = DictVectorizer(sparse=False)
        categOneHot = dv.fit_transform(categDatDict)
    categOneHotDF = pd.DataFrame(categOneHot, columns=dv.feature_names_)
    categOneHotDF['id'] = ids
    outputDat = noCategDat.merge(categOneHotDF, how='left', on='id')
    return outputDat, dv

def createNewFeatures(dat):
    """

    :param dat: pandas dataframe
    :return: pandas dataframe with generated features included
    """
    #lenDesc: did the borrower include a description, and if so, how long
    dat['lenDesc'] = dat['desc'].apply(descriptionLength)
    #how long has it been since this user got their first credit? Are they new to the market or relatively 0ld?
    dat['timeSinceFirstCredit'] = dat['issue_d'] - dat['earliest_cr_line']
    dat['timeSinceFirstCredit'] = dat['timeSinceFirstCredit'].dt.days
    dat['noNextPayment'] = pd.isnull(dat['next_pymnt_d']).astype(int)
    #how much, in units of installment, was the last payment made on the account
    dat['ratioInstallmentLast'] = dat['last_pymnt_amnt']/dat['installment']
    dat['percOutstandingPrinciple'] = dat['out_prncp']/dat['loan_amnt']
    return dat

### these functions don't get used during the final run of the script, but were used to conduct
### grid searches over model parameters for two different models (LinearSVC and RandomForest)
### using the given constrained-approval rate metric as the built-in scorer

def gridSearchSVC(trainX, trainY):
    #Best estimator:
    #class_weight: {0:1}. C = 0.7
    #metric score: .735
    svc = LinearSVC()
    imp = Imputer()
    pipe = Pipeline([('imputer', imp), ('svc', svc)])
    metricScorer = make_scorer(evaluationMetric)
    #adding weights on underrepresented negative class
    #initially included 0:3, 0:4, 0:5, but removed on later iterations when those proved
    #rarely to be successful
    classWeights = ['auto', {0:1}, {0:2}]
    print classWeights
    CRange = np.arange(0.1, 1, 0.1)
    print CRange
    #grid search over class weight (2:10), C:
    searcher = GridSearchCV(pipe, dict(svc__C=CRange, svc__class_weight=classWeights), scoring=metricScorer)
    i = 0
    for train, test in KFold(len(trainX), n_folds=4, shuffle=True):
        print "Fold " + str(i)
        trainFoldX, trainFoldY = trainX[train], trainY[train]
        testFoldX, testFoldY = trainX[test], trainY[test]
        print len(trainFoldX)
        print len(trainFoldY)
        print trainFoldX[0]
        print trainFoldY[0]
        searcher.fit(trainFoldX, trainFoldY)
        print searcher.best_params_
        print searcher.best_estimator_
        fileName = "bestLinearSVC" + str(i) + ".pkl"
        with open(fileName, 'wb') as fid:
            cPickle.dump(searcher.best_estimator_, fid)

        predictions = searcher.predict(testFoldX)
        print evaluationMetric(testFoldY, predictions)
        i += 1

def gridSearchRandomForest(trainX, trainY):
    #Best estimator:
    # max_depth: 250, n_estimators: 100, max_features: auto
    # cross-val evaluation metric: .
    randomForest = RandomForestClassifier(n_jobs=2)
    imp = Imputer()
    pipe = Pipeline([('imputer', imp), ('randomForest', randomForest)])
    metricScorer = make_scorer(evaluationMetric)
    maxDepths = [50, 100, 200, 250]
    nEstimators = [50, 100, 500, 1000]
    maxFeatures = ["auto", "log2"]
    #grid search over maxDepth, num-Estimators
    #utilizes the given metric scorer (approval weight given repayment constraint) as built-in scorer
    searcher = GridSearchCV(pipe, dict(randomForest__max_depth=maxDepths, randomForest__n_estimators=nEstimators,
                         randomForest__max_features=maxFeatures,), scoring=metricScorer)
    i = 0
    for train, test in KFold(len(trainX), n_folds=4, shuffle=True):
        print "Fold " + str(i)
        trainFoldX, trainFoldY = trainX[train], trainY[train]
        testFoldX, testFoldY = trainX[test], trainY[test]
        print len(trainFoldX)
        print len(trainFoldY)
        print trainFoldX[0]
        print trainFoldY[0]
        searcher.fit(trainFoldX, trainFoldY)
        print searcher.best_params_
        print searcher.best_estimator_
        fileName = "bestLinearSVC" + str(i) + ".pkl"
        with open(fileName, 'wb') as fid:
            cPickle.dump(searcher.best_estimator_, fid)

        predictions = searcher.predict(testFoldX)
        print evaluationMetric(testFoldY, predictions)
        i += 1


## these functions operate on the second-highest level of abstraction after the main method: one downloads
## and saves the data, the other chains all necessary transformation functions to get from
## a dataframe to a sklearn-usable dataset

def downloadData(directoryName='data'):
    trainURL = "https://resources.lendingclub.com/LoanStats3b.csv.zip"

    testURL = "https://resources.lendingclub.com/LoanStats3c.csv.zip"

    for url in [trainURL, testURL]:
        print "Downloading and unzipping data from %s" % url
        r = requests.get(url)
        if r.status_code == requests.codes.ok:
            z = zipfile.ZipFile(StringIO.StringIO(r.content))
            z.extractall(path=directoryName)
        else:
            print "Something went wrong downloading the files from the URL %s" % url

def transformData(dat, test=False, binaryStyle='noCurrent', dicVec=None):

    #dat = dat.drop('url', axis=1, inplace=True)
    dat = binarizeTarget(dat, binaryStyle)
    dat = convertDates(dat)
    dat['emp_length'] = dat['emp_length'].apply(nMonths)
    dat['term'] = dat['term'].apply(nMonths)
    dat['revol_util'] = dat['revol_util'].apply(percToDecimal)
    dat['int_rate'] = dat['int_rate'].apply(percToDecimal)
    dat['mths_since_last_delinq'] = dat['mths_since_last_delinq'].apply(monthsSince)
    dat['mths_since_last_record'] = dat['mths_since_last_record'].apply(monthsSince)
    dat['mths_since_last_major_derog'] = dat['mths_since_last_major_derog'].apply(monthsSince)
    dat = createNewFeatures(dat)
    #during test-set data creation, utilize dic-Vectorizer created during training
    if test:
        dat, dv = binarizeCategorical(dat, test=True, dv=dicVec)
    else:
        dat, dv = binarizeCategorical(dat)
    datSklDict = transform_csv(dat, target_col='binaryTarget', ignore_cols=['id', 'member_id', 'funded_amnt_inv', 'grade',
    'emp_title', 'url', 'title', 'zip_code', 'addr_state', 'earliest_cr_line', 'issue_d', 'out_prncp_inv', 'total_pymnt_inv',
    'total_pymnt', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 'policy_code', 'loan_status', 'desc'])
    X = np.asarray(datSklDict['data'])
    y = datSklDict['target']
    y = np.asarray([yElem[0] for yElem in y])

    return X, y, dv



if __name__ == "__main__":
    downloadData()
    inputTrainDat = pd.read_csv('data/LoanStats3b.csv', skiprows=1)
    inputTestDat = pd.read_csv('data/LoanStats3c.csv', skiprows=1)
    testIndex = inputTestDat['loan_status']
    trainX, trainY, vectTransformer = transformData(inputTrainDat)
    #gridSearchSVC(trainX, trainY)
    #gridSearchRandomForest(trainX, trainY)
    testX, testY, vectTransformer = transformData(inputTestDat, test=True, dicVec=vectTransformer)
    svc = LinearSVC(class_weight={0: 1}, C=0.7)
    imp = Imputer()
    finalModel = Pipeline([('imputer', imp), ('svc', svc)])
    finalModel.fit(trainX, trainY)
    predictions = finalModel.predict(testX)

    predictInd = 0
    nonPredictInd = 0
    #since I didn't use current data in training, and it's not being used for evaluation
    #I use this structure with 'NA' for current loans to make sure indexes match up
    #with original data set
    for ind, status in enumerate(testIndex):

        if isinstance(status, float):
            print "Row Index: %s Classification: %s" % (ind, 'NA: nan')
            nonPredictInd += 1
        elif status.rstrip() == 'Current':
            print "Row Index: %s Classification: %s" % (ind, 'NA: Current Loan')
            nonPredictInd += 1
        else:
            print "Row Index: %s Classification: %s" % (ind, predictions[predictInd])
            predictInd += 1

    print "Classified as Fully_Paid: " + str(np.sum(predictions))
    print "Classified as not Fully_Paid (i.e. late or default): " + str(len(predictions) - np.sum(predictions))
    print "Not Classified: Current Loans or status='nan: " + str(nonPredictInd)
    print "Evaluation Metric: " + str(round(evaluationMetric(testY, predictions), 3))

