import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report




class dataHandler:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = None
        self.input_data = None
        self.output_data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_name)

    def deleteUnusedCol(self):
        self.data.drop(columns=['id'], inplace=True)
        self.data.drop(columns=['CustomerId'], inplace=True)
        self.data.drop(columns=['Surname'], inplace=True)
        self.data.drop(columns=['Unnamed: 0'], inplace=True)

    def fillingNull(self):
        self.data['CreditScore'].fillna(self.data['CreditScore'].mean(), inplace=True)

    def encodeCategorical(self):
        geography_mapping = {'France': 0, 'Germany': 1, 'Spain': 2}
        self.data['Geography'] = self.data['Geography'].replace(geography_mapping)

        gender_mapping = {'Female': 0, 'Male': 1}
        self.data['Gender'] = self.data['Gender'].replace(gender_mapping)


    def createInputOutput(self, var_target):
        self.input_data = self.data.drop(var_target, axis = 1)
        self.output_data = self.data[var_target]



class modelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.x_train = None 
        self.x_test = None 
        self.y_train = None 
        self.y_test = None
        self.y_pred = None
        self.XGmodel = None

    def splitData(self, test_size,rd_state): 
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=rd_state)
        
    def createXGmodel(self, tree_method):
        self.XGmodel = xgb.XGBClassifier(tree_method = tree_method)

    def trainXGmodel(self):
        self.XGmodel.fit(self.x_train, self.y_train)

    def makePrediction(self):
        self.y_pred = self.XGmodel.predict(self.x_test)

    def createReport(self):
        print("XGBoost Model Report\n")
        print(classification_report(self.y_test, self.y_pred))

    def saveToFile(self, model_name):
        with open(model_name, 'wb') as file: 
            pkl.dump(self.XGmodel, file)



data_handler = dataHandler("data_D.csv")
data_handler.load_data()
data_handler.deleteUnusedCol()
data_handler.fillingNull()
data_handler.encodeCategorical()
data_handler.createInputOutput("churn")
input_data = data_handler.input_data
output_data = data_handler.output_data

model_handler = modelHandler(input_data, output_data)
model_handler.splitData(0.2, 32)
model_handler.createXGmodel("auto")
model_handler.trainXGmodel()
model_handler.makePrediction()
model_handler.createReport()
model_handler.saveToFile("oopPicklefile_2602100120.pkl")


