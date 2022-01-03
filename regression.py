import  numpy as np #Lib for read arraies
from sklearn.linear_model import LinearRegression #lib for perform linear and polynomial regression
#x = regressors, y = predictors
x = np.array([160, 165, 171, 156, 166, 172, 167, 159, 162, 174]).reshape((-1,1))
y = np.array([65, 60, 68, 61, 65, 70, 66, 61, 66, 75]).reshape((-1,1))
print(x)
print(y)
#creating a model
model = LinearRegression()
model.fit(x, y)
#for shorter model = LinearRegression().fit(x,y)
r_sq = model.score(x, y) #get the value of R^2
print('Cofficient of determination: ', r_sq)
attribute = model.intercept_
slope = model.coef_
print('intercept: ', attribute) #illustrates that your model predicts the response 5.63 when 𝑥 is zero.
print('slope:', slope) #means that the predicted response rises by 0.54 when 𝑥 is increased by one.
#predict response
y_pred = model.predict(x)
print('predicited response: ', y_pred, sep='\n')
'''
the same way to predict the response
y_pred = model.intercept_ + model.coef_ * x
print('predicited response: ', y_pred, sep='\n')
'''
userInput = input('enter your number: ')
userInput_int = int(userInput)
new_ypred = model.predict(np.array([userInput_int]).reshape(-1,1))
print('predicited value: ',new_ypred)


