import pickle
# Refer progeam Pract1.py for getting the details of diabetic1.pkl, We can directly do the prediction instead of 
# loading/writing  the entire program again.

# Load the Model

model = pickle.load(open('diabetic1.pkl','rb'))

result = model.predict([[1,2,3,4,5,6,7,8]])[0]

print('Result is:', result) # Result is: 1.0 (1.0 ppl has diabetic)

