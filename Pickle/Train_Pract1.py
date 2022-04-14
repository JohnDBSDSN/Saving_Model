import pickle
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle # Part2 ; for pickle part...replace all the joblib to pickle

# Refer Pract1_test.py/Pract1.py  for loading this program using pkl (diabetic1.pkl) concept


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']

df  = pd.read_csv(url , names=names)

print(df)
print('Working fine!!')
array = df.values
X = array[:,0:8]
y = array[:,8]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y , test_size = 0.2, random_state=101)

# fit the model
model = LogisticRegression()
model.fit(X_train , y_train)

#accuracy

result = model.score(X_test , y_test)
print(f'the accuracy of the model is {result}')

# Save the Model (Part 2)
pickle.dump(model,open('diabetic1.pkl', 'wb')) # U have to give in a file handling way if its a pickle 
                                               # with the syntax