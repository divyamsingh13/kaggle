import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df = pd.read_csv('train.csv') #load the training set
df1 = pd.read_csv('test.csv')
#df1=df1.drop('Survived',axis=1)
df["Age"].fillna(df["Age"].median(), inplace=True) #fill missing age with mean values
df1["Age"].fillna(df1["Age"].median(), inplace=True)
#to improve speed of training the model remove unnecessary data
df["HasCabin"]=df["Cabin"]
df.HasCabin.loc[df.Cabin.notnull()]=1
df.HasCabin.loc[df.Cabin.isnull()]=0
df.Sex=df.Sex.astype('category').cat.codes #male=1 female=0
df.Embarked=df.Embarked.astype('category').cat.codes #s=2 c=0 q=1
df.Fare = df.Fare.map(lambda x: np.nan if x==0 else x)
df1.Fare = df1.Fare.map(lambda x: np.nan if x==0 else x)

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if str.find(big_string, substring) != -1:
            return substring
    print(big_string)
    return np.nan

title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']
df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))

title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']
df1['Title']=df1['Name'].map(lambda x: substrings_in_string(x, title_list))

def replace_titles(x):
    title = x['Title']
    if title in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Master']:
        return 'Master'
    elif title in ['Countess', 'Mme', 'Mrs']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms', 'Miss']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex'] == 'Male':
            return 'Mr'
        else:
            return 'Mrs'
    elif title == '':
        if x['Sex'] == 'Male':
            return 'Master'
        else:
            return 'Miss'
    else:
        return title


df['Title'] = df.apply(replace_titles, axis=1)
df1['Title'] = df1.apply(replace_titles, axis=1)

df['AgeFill'] = df['Age']
mean_ages = np.zeros(4)
mean_ages[0] = np.average(df[df['Title'] == 'Miss']['Age'].dropna())
mean_ages[1] = np.average(df[df['Title'] == 'Mrs']['Age'].dropna())
mean_ages[2] = np.average(df[df['Title'] == 'Mr']['Age'].dropna())
mean_ages[3] = np.average(df[df['Title'] == 'Master']['Age'].dropna())
df.loc[(df.Age.isnull()) & (df.Title == 'Miss'), 'AgeFill'] = mean_ages[0]
df.loc[(df.Age.isnull()) & (df.Title == 'Mrs'), 'AgeFill'] = mean_ages[1]
df.loc[(df.Age.isnull()) & (df.Title == 'Mr'), 'AgeFill'] = mean_ages[2]
df.loc[(df.Age.isnull()) & (df.Title == 'Master'), 'AgeFill'] = mean_ages[3]

df1['AgeFill'] = df1['Age']
mean_ages = np.zeros(4)
mean_ages[0] = np.average(df1[df1['Title'] == 'Miss']['Age'].dropna())
mean_ages[1] = np.average(df1[df1['Title'] == 'Mrs']['Age'].dropna())
mean_ages[2] = np.average(df1[df1['Title'] == 'Mr']['Age'].dropna())
mean_ages[3] = np.average(df1[df1['Title'] == 'Master']['Age'].dropna())
df1.loc[(df1.Age.isnull()) & (df1.Title == 'Miss'), 'AgeFill'] = mean_ages[0]
df1.loc[(df1.Age.isnull()) & (df1.Title == 'Mrs'), 'AgeFill'] = mean_ages[1]
df1.loc[(df1.Age.isnull()) & (df1.Title == 'Mr'), 'AgeFill'] = mean_ages[2]
df1.loc[(df1.Age.isnull()) & (df1.Title == 'Master'), 'AgeFill'] = mean_ages[3]


df1["HasCabin"]=df1["Cabin"]
df1.HasCabin.loc[df1.Cabin.notnull()]=1
df1.HasCabin.loc[df1.Cabin.isnull()]=0
df1.Sex=df1.Sex.astype('category').cat.codes #male=1 female=0
df1.Embarked=df1.Embarked.astype('category').cat.codes #s=2 c=0 q=1

#drop these columns they do not help in prediction





df['AgeCat'] = df['AgeFill']
df.loc[(df.AgeFill <= 10), 'AgeCat'] = 'child'
df.loc[(df.AgeFill > 60), 'AgeCat'] = 'aged'
df.loc[(df.AgeFill > 10) & (df.AgeFill <= 30), 'AgeCat'] = 'adult'
df.loc[(df.AgeFill > 30) & (df.AgeFill <= 60), 'AgeCat'] = 'senior'

df.Embarked = df.Embarked.fillna('S')

df1['AgeCat'] = df1['AgeFill']
df1.loc[(df1.AgeFill <= 10), 'AgeCat'] = 'child'
df1.loc[(df1.AgeFill > 60), 'AgeCat'] = 'aged'
df1.loc[(df1.AgeFill > 10) & (df1.AgeFill <= 30), 'AgeCat'] = 'adult'
df1.loc[(df1.AgeFill > 30) & (df1.AgeFill <= 60), 'AgeCat'] = 'senior'

# Creating new family_size column
df['Family_Size'] = df['SibSp'] + df['Parch']
df['Family'] = df['SibSp'] * df['Parch']

# imputing nan values
df.loc[(df.Fare.isnull()) & (df.Pclass == 1), 'Fare'] = np.median(df[df['Pclass'] == 1]['Fare'].dropna())
df.loc[(df.Fare.isnull()) & (df.Pclass == 2), 'Fare'] = np.median(df[df['Pclass'] == 2]['Fare'].dropna())
df.loc[(df.Fare.isnull()) & (df.Pclass == 3), 'Fare'] = np.median(df[df['Pclass'] == 3]['Fare'].dropna())

# Creating new family_size column
df1['Family_Size'] = df1['SibSp'] + df1['Parch']
df1['Family'] = df1['SibSp'] * df1['Parch']

# imputing nan values
df1.loc[(df1.Fare.isnull()) & (df1.Pclass == 1), 'Fare'] = np.median(df1[df1['Pclass'] == 1]['Fare'].dropna())
df1.loc[(df1.Fare.isnull()) & (df1.Pclass == 2), 'Fare'] = np.median(df1[df1['Pclass'] == 2]['Fare'].dropna())
df1.loc[(df1.Fare.isnull()) & (df1.Pclass == 3), 'Fare'] = np.median(df1[df1['Pclass'] == 3]['Fare'].dropna())

le.fit(df['AgeCat'])
x_age=le.transform(df['AgeCat'])
df['AgeCat'] =x_age.astype(np.float)

le.fit(df1['AgeCat'])
x_age=le.transform(df1['AgeCat'])
df1['AgeCat'] =x_age.astype(np.float)

le.fit(df['Ticket'])
x_Ticket = le.transform(df['Ticket'])
df['Ticket'] = x_Ticket.astype(np.float)

le.fit(df['Title'])
x_title = le.transform(df['Title'])
df['Title'] = x_title.astype(np.float)

le.fit(df1['Ticket'])
x_Ticket = le.transform(df1['Ticket'])
df1['Ticket'] = x_Ticket.astype(np.float)

le.fit(df1['Title'])
x_title = le.transform(df1['Title'])
df1['Title'] = x_title.astype(np.float)

df1.Embarked = df1.Embarked.fillna('S')

dropcolumns = ['Name','Cabin']
df.drop(dropcolumns,axis=1,inplace=True)
df1.drop(dropcolumns,axis=1,inplace=True)

#drop empty rows
df = df.dropna(axis=0)
x = df
x = x.drop('Survived', axis=1)

y = df.Survived

# LR= LogisticRegression()
# LR.fit(x,y)
# Y_pred1=LR.predict(df1)
# print(LR.score(x,y))

random_forest = RandomForestClassifier(n_estimators= 10, max_features='log2', min_samples_leaf= 3, max_depth= 4, min_samples_split= 10, bootstrap=True)
random_forest.fit(x,y)
Y_pred = random_forest.predict(df1)
random_forest.score(x,y)
acc_random_forest = round(random_forest.score(x,y) * 100, 2)
print(acc_random_forest)


# parameter_grid = {
#                  'max_depth' : [4, 6, 8],
#                  'n_estimators': [50, 10,100,150,75],
#                  'max_features': ['sqrt', 'auto', 'log2'],
#                  'min_samples_split': [1.0, 3, 10],
#                  'min_samples_leaf': [1, 3, 10],
#                  'bootstrap': [True, False],
#                  }
#
# cross_validation = StratifiedKFold(y, n_folds=5)
#
# grid_search = GridSearchCV(random_forest,
#                            param_grid=parameter_grid,
#                            cv=cross_validation)
#
# grid_search.fit(x,y)
#
# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))
#
# print("finished")
pred_df = pd.DataFrame({
        "PassengerId": df1["PassengerId"],
        "Survived": Y_pred
    })
pred_df.to_csv('kaggle-titanic-competition.csv', index=False)