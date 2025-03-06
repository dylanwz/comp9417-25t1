import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv("heart.csv")

###################################################################################################
#
#   QUESTION 1: Data Wrangling
#
###################################################################################################

y = df['Heart_Disease']

# Part (A): - create X,y
#           - remove column
X = df.drop(columns=['Heart_Disease', 'Last_Checkup'])

# Part (B): - modify a column
#           - age -> |age|
#           - `['coln'] = ` directly, like int variable
X['Age'] = X['Age'].abs()

# Part (C): - modify a column
#           - gender -> category(gender), smoker -> category(smoker)
#           - `replace` operates coln-wise with a dict map
gender_category_map = {'Male': 0, 'M': 0, 'Female': 1, 'F': 1, 'Unknown': 2}
smoker_category_map = {'No': 0, 'N': 0, 'Yes': 1, 'Y': 1, np.nan: 2}
X['Gender'] = X['Gender'].replace(gender_category_map)
X['Smoker'] = X['Smoker'].replace(smoker_category_map)

# Part (D): - split a column
#           - blood pressure -> systolic, diastolic
#           - # `.str.split()` splits, `expand` returns dataframes to be put into cols
X[['Systolic', 'Diastolic']] = X['Blood_Pressure'].str.split('/', expand=True).astype(float)
X.drop(columns=['Blood_Pressure'], inplace=True)

# Part (E): - form datasets from dataframes i.e. arrays
#           - shuffle and proportionally split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# PART (F): - modify a column
#           - impute age values
#           - `groupby` separates the dataset along rows based on the indicator i.e. gender, gives a df-like obj
def impute_age(df, col):
    medians = df.groupby('Gender')['Age'].median()      # doesn't include NaN values
    def do_impute_age(row):
        if pd.isnull(row['Age']):
            return medians[row['Gender']]               # use index medians with gender value of row
        return row['Age']
    df[col] = df.apply(do_impute_age, axis=1)           # axis=1 -> row-wise/down the coln
impute_age(X_train, 'Age')                                      
impute_age(X_test, 'Age')              

# Part (G): - modify a column
#           - noramlise key numerical columns
#           - as in (B)
cols_to_noramlise = ['Age', 'Height_feet', 'Weight_kg', 'Cholesterol', 'Systolic', 'Diastolic']
def normalise_cols(df):
    for col in cols_to_noramlise:
        min_val =  df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val)/(max_val - min_val)       # separately
normalise_cols(X_train)
normalise_cols(X_test)                                          # for this, test doesn't use metrics from train      

# Part (H): - plotting histogram from list of values i.e. how many of each value occurs
#           - for each value of heart disease, how manny instances
#           - `plt.hist` makes 'plot' instance to be added with axis title, labels, data, etc.
plt.hist(y_train, bins=20, edgecolor='black')       # split heart disease values into 20 intervals 
plt.xlabel('Heart disease')
plt.ylabel('Frequency')
plt.title('Histogram of heart disease values')
plt.savefig('hist_initial.png')                     # save because can't plot

# Part (H): - modify labels
#           - quantise heart disease values
#           - act on coln as if it's an 'elt-wise operating' variable
k = 0.1
y_train_qnt, y_test_qnt = (y_train > k).astype(int), (y_test > k).astype(int)
plt.clf()                                               # clear histogram
plt.hist(y_train_qnt, bins=20, edgecolor='black')       # split vals into 20 intervals
plt.xlabel('Heart Disease')
plt.ylabel('Frequency')
plt.title('Target Variable Histogram')
plt.savefig('hist_quantised.png')

###################################################################################################
#
#   QUESTION 2: Logistic Regression and Hyper-parameter Tuning
#
###################################################################################################

# Part (B): - fit a model to a set of hyperparameters
#           - `logspace(p1, p2, p3)` generates p3 values between p1 and p2 log-equally distributed
#           - `LogisticRegression(c, p, s)` makes a shell logistic regression hypothesis
#           - `fit(X, y)` trains hypothesis on X and y
#           - `predict_proba(X)` predicts probability of each class for each vec in X
#           - `log_loss(y_true, y_pred)` gives sum of cross entropy loss of each sample
vals = np.logspace(-4, 4, 100)
train_losses = []
test_losses = []
for val in vals:
    m = LogisticRegression(C=val, penalty='l2', solver='lbfgs')
    m.fit(X_train, y_train_qnt) 
    y_train_prob, y_test_prob = m.predict_proba(X_train), m.predict_proba(X_test)
    train_losses.append(log_loss(y_train_qnt, y_train_prob))
    test_losses.append(log_loss(y_test_qnt, y_test_prob))
plt.figure(figsize=(8, 6))
plt.title('Log Loss vs C')
plt.plot(vals, train_losses, label='Train log loss', marker='o')
plt.plot(vals, test_losses, label='Test log loss', marker='o')
plt.xlabel('C')
plt.xscale('log')                                                       # scale in log for visibility
plt.ylabel('Log loss')
plt.legend()
plt.grid()
plt.savefig('loss_vs_C.png')

# Part (C): - perform k-fold cross validation
# (i)       - `df.iloc(*)` = integer-location based indexing based on *; [i1:i2] gives [i1, i2) 
vals = np.logspace(-4, 4, 100)
fold_size = len(X_train) // 5

cv_results = []                                                                             # loss values for each of 20 test samples over 5 folds
for val in vals:
    fold_losses = []

    for i in range(5):
        start, end = i * fold_size, (i + 1) * fold_size
        
        X_test_fold, y_test_fold = X_train.iloc[start:end], y_train_qnt.iloc[start:end]     # set testing fold to be ith fold
        X_train_fold = pd.concat([X_train.iloc[:start], X_train.iloc[end:]])                # training is everything outside iloc
        y_train_fold = pd.concat([y_train_qnt.iloc[:start], y_train_qnt.iloc[end:]])        # labels match training
        
        m = LogisticRegression(C=val, penalty='l2', solver='lbfgs')
        m.fit(X_train_fold, y_train_fold)
        y_pred = m.predict_proba(X_test_fold)

        fold_losses.append(log_loss(y_test_fold, y_pred))                                   # cross-entropy from tute3
    
    cv_results.append(fold_losses)

plt.figure(figsize=(10, 6))
plt.title('CV Loss vs C Values')                                                            # for legibility, read every 10th label
plt.xlabel('log(C)')
plt.ylabel('Log loss')
plt.boxplot(cv_results, positions=np.log10(vals))
plt.xticks(np.log10(vals)[::10], labels=[f'$10^{{{int(np.log10(val))}}}$' for val in vals[::10]])
plt.grid()
plt.savefig('CV_loss_vs_C.png')

# Part (C): - select the C with the 'best' results
# (ii)      - print results with this model
mean_losses = [np.mean(losses) for losses in cv_results]        # use mean over folds as per k-fold CV
min_mean_C = vals[np.argmin(mean_losses)]
print(f'C for smallest mean log loss: {min_mean_C}')
m2 = LogisticRegression(C=min_mean_C, penalty='l2', solver='lbfgs')
m2.fit(X_train, y_train_qnt)
y_train_pred, y_test_pred = m2.predict(X_train), m2.predict(X_test)
train_accuracy, test_accuracy = accuracy_score(y_train_qnt, y_train_pred), accuracy_score(y_test_qnt, y_test_pred)
print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Part (D): - use sklearn implementation of grid search
#           - `param_grid` = dict of parameters and possible vals
#           - `GridSearchCV(est, cv, grid)` applies grid search to find best value for the est with additional params i.e. cv
param_grid = {'C': vals}
grid_lr = GridSearchCV(
    estimator=LogisticRegression(penalty='l2',solver='lbfgs'),
    cv=5,
    param_grid=param_grid)
grid_lr.fit(X_train, y_train_qnt)
found_C = grid_lr.best_params_['C']
print(f'GridSearchCV found: {found_C}')

# source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV.score
kf = KFold(n_splits=5, shuffle=False)
grid_lr = GridSearchCV(
    estimator=LogisticRegression(penalty='l2',solver='lbfgs'),
    cv=kf,                      # is normally stratified (preserves labels)
    param_grid=param_grid,
    scoring='neg_log_loss',     # normally uses mean accuracy rather than log loss!!  
    refit=False)
grid_lr.fit(X_train, y_train_qnt)
found_C = grid_lr.best_params_['C']
print(f'Adjusted GridSearchCV found: {found_C}')