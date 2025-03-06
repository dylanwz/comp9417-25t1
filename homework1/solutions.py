import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

# pd.set_option('future.no_silent_downcasting', True)

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
X.drop(columns=['Blood_Pressure'], inplace=True)                                                # 

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
            return medians[row['Gender']]                   # use index medians with gender value of row
        return row['Age']
    df[col] = df.apply(do_impute_age, axis=1)
impute_age(X_train, 'Age')                                      # axis=1 -> row-wise/down the coln
impute_age(X_test, 'Age')              

# Part (G): - modify a column
#           - noramlise key numerical columns
#           - as in (B)
cols_to_noramlise = ['Age', 'Height_feet', 'Weight_kg', 'Cholesterol', 'Systolic', 'Diastolic']
def normalise_cols(df):
    for col in cols_to_noramlise:
        min_val =  df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val)/(max_val - min_val)
normalise_cols(X_train)
normalise_cols(X_test)                                  # for this, test doesn't use metrics from train       

# Part (H): - plotting histogram from list of values i.e. how many of each value occurs
#           - for each value of heart disease, how manny instances
#           - `plt.hist` makes 'plot' instance to be added with axis title, labels, data, etc.
plt.hist(y_train, bins=20, edgecolor='black')
plt.xlabel('Heart disease')
plt.ylabel('Frequency')
plt.title('Histogram of heart disease values')
plt.savefig('hist_initial.png')

# Part (H): - modify labels
#           - quantise heart disease values
#           - act on coln as if it's an 'elt-wise operating' variable
k = 0.1
y_train_qnt = (y_train > k).astype(int)
y_test_qnt = (y_test > k).astype(int)
plt.clf()                                               # clear histogram
plt.hist(y_train_qnt, bins=20, edgecolor='black')       # split vals into 20 intervals
plt.xlabel('Heart Disease')
plt.ylabel('Frequency')
plt.title('Target Variable Histogram')
plt.savefig('hist_quantised.png')                       # save because can't plot

###################################################################################################
#
#   QUESTION 2: Logistic Regression and Hyper-parameter Tuning
#
###################################################################################################

# Part (A): - 
vals = np.logspace(-4, 4, 100)
train_losses = []
test_losses = []
for val in vals:
    m = LogisticRegression(C=val, penalty='l2', solver='lbfgs')
    m.fit(X_train, y_train_qnt) 
    y_train_prob = m.predict_proba(X_train)
    y_test_prob = m.predict_proba(X_test)
    train_losses.append(log_loss(y_train_qnt, y_train_prob))
    test_losses.append(log_loss(y_test_qnt, y_test_prob))
plt.figure(figsize=(8, 6))
plt.plot(vals, train_losses, label='Train log loss', marker='o')
plt.plot(vals, test_losses, label='Test log loss', marker='s')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Log loss')
plt.title('Log loss vs C')
plt.legend()
plt.grid()
plt.savefig('loss_vs_C.png')

# C:
# (j) Perform 5-fold cross-validation for different C values
C_values = np.logspace(-4, 4, 100)
cv_results = []

N = len(X_train)
fold_size = N // 5

for C in vals:
    fold_losses = []
    
    for i in range(5):
        start, end = i * fold_size, (i + 1) * fold_size
        X_val, y_val = X_train.iloc[start:end], y_train_qnt.iloc[start:end]
        X_train_fold = pd.concat([X_train.iloc[:start], X_train.iloc[end:]])
        y_train_fold = pd.concat([y_train_qnt.iloc[:start], y_train_qnt.iloc[end:]])
        
        model = LogisticRegression(C=C, penalty='l2', solver='lbfgs')
        model.fit(X_train_fold, y_train_fold)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        fold_losses.append(log_loss(y_val, y_val_prob))
    
    cv_results.append(fold_losses)

# (k) Plot boxplot of log-loss for different C values
plt.figure(figsize=(10, 6))
plt.boxplot(cv_results, positions=np.log10(C_values))
plt.xlabel('log10(C)')
plt.ylabel('Log-Loss')
plt.title('5-Fold Cross-Validation Log-Loss for Different C Values')
plt.grid()
plt.savefig('CV_loss_vs_C.png')

# (l) Select best C based on median log-loss
median_losses = [np.median(losses) for losses in cv_results]
best_C = C_values[np.argmin(median_losses)]
print(f'Best C: {best_C}')

# (m) Train final model with best C
final_model = LogisticRegression(C=best_C, penalty='l2', solver='lbfgs')
final_model.fit(X_train, y_train_qnt)

y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

train_accuracy = accuracy_score(y_train_qnt, y_train_pred)
test_accuracy = accuracy_score(y_test_qnt, y_test_pred)

print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

