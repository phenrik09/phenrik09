# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:25:45 2021

@author: User
"""

import pandas as pd
import plotly.express as px

tabela = pd.read_csv(r"C:\Users\User\Documents\ENGENHARIA UERJ\Python\Curso Hashtag Lira\Aula 2\telecom_users.csv")

print(tabela)

print(tabela.columns)
print(tabela.info())

tabela = tabela.dropna(how="all", axis=1) #Exclui colunas nulas, no caso de linhas axis=0
tabela = tabela.dropna(how="any") #Exclui qualquer linha ou coluna com valor nulo

print(tabela)
print(tabela.info())

print(tabela['Churn'].value_counts(normalize=True))

for coluna in tabela:
    grafico = px.histogram(tabela, x=coluna ,color="Churn")
    grafico.show()
