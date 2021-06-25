import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as mpp


df = pd.read_csv(r'C:\Users\User\Documents\ENGENHARIA UERJ\Python\Curso Hashtag Lira\Aula 3\advertising.csv')
#print(df)
#print(df.info())

#sbn.pairplot(df) #Criar o gráfico via seaborn
#mpp.show() #visualizar o gráfico criado anteriormente com o matplot
#sbn.heatmap(df.corr(), cmap = 'YlGnBu', annot = True)
#mpp.show()

from sklearn.model_selection import train_test_split
x = df.drop('Vendas', axis = 1)
y = df['Vendas']
x_train, x_test, y_train, y_test  = train_test_split(x,y,test_size = 0.3, random_state = 1)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np

#Treino AI

lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)

rf_reg = RandomForestRegressor()
rf_reg.fit(x_train,y_train)

#Teste AI

test_pred_lin = lin_reg.predict(x_test)
test_pred_rf = rf_reg.predict(x_test)

r2_lin = metrics.r2_score(y_test, test_pred_lin)
rmse_lin = np.sqrt(metrics.mean_squared_error(y_test, test_pred_lin))
print(f'R² da Regressão Linear: {r2_lin}')
print(f'RSME da Regressão Linear: {rmse_lin}')

r2_rf = metrics.r2_score(y_test, test_pred_rf)
rsme_rf = np.sqrt(metrics.mean_squared_error(y_test, test_pred_rf))
print(f'R² do Random Forest: {r2_rf}')
print(f'RSME do Random Forest: {rsme_rf}')

#Analise Grafica

df_resultado = pd.DataFrame()
#df_resultado.index = x_test
df_resultado['y_test'] = y_test
df_resultado['y_previsao_lin'] = test_pred_lin
df_resultado['y_previsao_rf'] = test_pred_rf
df_resultado = df_resultado.reset_index(drop=True)
fig = mpp.figure(figsize=(15,5))
sbn.lineplot(data = df_resultado)
mpp.show()
print(df_resultado)
