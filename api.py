from flask import Flask, render_template, request, jsonify, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
import random
import mysql.connector
from mysql.connector import Error

mydb = mysql.connector.connect(
	  host="localhost",
	  user="root",
	  password="",
	  database="fog"
	)
mycursor = mydb.cursor()


import sys

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config["DEBUG"] = True

@app.route('/database')   
def BPdatabase(pred1, pred2, pred3,):

	#To get last entry of patient to update his record
	rows = mycursor.execute("SELECT * FROM records") 
	mycursor.execute(rows)
	upd = len(mycursor.fetchall())
	sql = "UPDATE records SET Pred1 = %s, Pred2 = %s, Pred3 = %s WHERE rid = %s"
	val = (int(pred1), int(pred2), int(pred3), upd)

	mycursor.execute(sql, val)

	mydb.commit()

	print(mycursor.rowcount, "record(s) affected")
def Dbdatabase(pred1, pred2, pred3,):

	#To get last entry of patient to update his record
	rows = mycursor.execute("SELECT * FROM diabetespatients") 
	mycursor.execute(rows)
	upd = len(mycursor.fetchall())
	sql = "UPDATE diabetespatients SET pred1 = %s, pred2 = %s, pred3 = %s WHERE did = %s"
	val = (int(pred1), int(pred2), int(pred3), upd)

	mycursor.execute(sql, val)

	mydb.commit()

	print(mycursor.rowcount, "record(s) affected")
	
@app.route('/diabetes')   
def diabetes(did, glucose, systolic, bmi, glrisk):
	data = pd.read_csv('diabetes.csv')
	#now separate the data what we need to training 
	newdata = data.drop(['Pregnancies','SkinThickness', 'Insulin', 'DiabetesPedigreeFunction','Outcome'], axis=1)
	# create a list of our conditions
	conditions = [
		(newdata['Glucose'] <=35), #Extremly Low
		(newdata['Glucose'] >35) & (newdata['Glucose'] <= 79) , #low
		(newdata['Glucose'] >= 80) & (newdata['Glucose'] <= 149) , #Normal
		(newdata['Glucose'] >= 150) & (newdata['Glucose'] <= 314) , #Prediabetes
		(newdata['Glucose'] >= 315) #Very Hight
		]
	# create a list of the values we want to assign for each condition
	values = [1, 2, 3, 4, 5]
	# create a new column and use np.select to assign values to it using our lists as arguments
	newdata['stage'] = np.select(conditions, values)
	#rename the column name according to patient data
	newdata.rename(columns = {'Glucose':'glucose','BloodPressure':'bp','BMI':'bmi','Age':'age'}, inplace = True) 
	#Count we have maximnum 15 value in patient data or not 
	ID = did
	print("Fetching Patient data is: ", did); mycursor.execute("SELECT p.Age AS patients, dp.diabetes, dp.bp, dp.bmi, dp.Status AS records FROM diabetespatients dp LEFT JOIN patients p ON dp.pid = p.pid WHERE dp.pid = %s", (did,));
	myresult = mycursor.fetchall()
	
	#now convert this data first in array formate then we convert in dataframe using pandas
	test_dataset = np.array(myresult)
	patient_dataset = pd.DataFrame(test_dataset, columns=['age','glucose','bp','bmi','stage'])
	print(patient_dataset)
	
	
	# ==========================================***********************============================================
	
	
	#Split the data into 50% for training and 50% for testing the data validation
	train_dataset, temp_test_dataset = train_test_split(newdata, test_size=0.5)
	#Split the data into 80% for training and 20% for testing the data validation
	test_dataset, valid_dataset = train_test_split(temp_test_dataset, test_size=0.5)
	#now here i replace the valid data with user dataset
	valid_dataset = patient_dataset
	
	#Statistics on the datset to make sure it is a good shape
	train_stats = train_dataset.describe()
	train_stats.pop("stage")
	train_stats = train_stats.transpose()

	train_label = train_dataset.pop('stage')
	test_label = test_dataset.pop('stage')
	valid_label = valid_dataset.pop('stage')

	#Data Normalization/Scaling
	#define a fuunction to normalize the data set.

	def norm(x):
		return (x - train_stats['mean']) / train_stats['std']
	normed_train_data = norm(train_dataset)
	normed_test_data = norm(test_dataset)
	normed_valid_dataset = norm(valid_dataset)

	#Create a SVM Classifier
	model = svm.SVC(C=1, kernel='linear',)
	#Training the model using Training set
	model.fit(normed_train_data, train_label)
	#Predict the response for test dataset
	y_pred = model.predict(normed_test_data)

	example_batch = normed_test_data[:10]
	example_result = model.predict(example_batch)


	# Test Patient Data

	validdata = normed_valid_dataset.tail(14)
	validdata1 = model.predict(validdata)
	
	val1 = validdata1[11]
	val2 = validdata1[12]
	val3 = validdata1[13]
	
	Dbdatabase(val1, val2, val3,)
	print("Accuracy with Patient Data(Diabetes): ",metrics.accuracy_score(valid_label.tail(14), validdata1))


@app.route('/bloodpressure')   
def bloodpressure(pid, plusrate, bprisk, age, weight, height, systolic, diastolic, bmi):

	
	data = pd.read_csv('BPDataset.csv')
	# create a list of our conditions
	conditions = [
		(data['ap_hi'] <120) &(data['ap_lo'] <80),
		(data['ap_hi'] >= 120) & (data['ap_hi'] <= 130) , (data['ap_lo'] <=80) ,
		(data['ap_hi'] >= 140) & (data['ap_hi'] <= 140) , (data['ap_lo'] >= 80) & (data['ap_lo'] <= 90),
		(data['ap_hi'] >= 140) & (data['ap_hi'] < 180) , (data['ap_lo'] >= 90) & (data['ap_lo'] < 120),
		(data['ap_hi'] >= 180) & (data['ap_lo'] >= 120)
		]

	# create a list of the values we want to assign for each condition
	values = [1, 2, 2, 3, 3, 4, 4, 5]

	# create a new column and use np.select to assign values to it using our lists as arguments
	data['stage'] = np.select(conditions, values)
	# display updated DataFrame
	data = data.head(2000)
	#now here we drop the column becaues patient dataset have no column
	data.drop('alco', axis='columns', inplace=True)
	#here we replacing the name of column according to patient dataset
	data.rename(columns = {'ap_hi':'bphi','ap_lo':'bplo'}, inplace = True); mycursor.execute("SELECT p.Age, p.Gender, p.Height, p.Smoking, p.Cardio, p.Active, p.Glucose, p.Cholestrol,p.Weight AS patients, r.Hiper, r.Loper,r.Status AS records FROM records r LEFT JOIN patients p ON r.RPid = p.Pid WHERE r.RPid = %s", (pid,));
	

	myresult = mycursor.fetchall()
	# now convert this data first in array formate then we convert in dataframe using pandas
	test_dataset = np.array(myresult)
	patient_dataset = pd.DataFrame(test_dataset, columns=['age','gender','height','smoke','cardio','active','gluc','cholesterol','weight','bphi','bplo','stage'])
	print(patient_dataset)
	
	#Split the data into 80% for training and 20% for testing the data validation
	train_dataset, temp_test_dataset = train_test_split(data, test_size=0.4)
	#Split the data into 80% for training and 20% for testing the data validation
	test_dataset, valid_dataset = train_test_split(temp_test_dataset, test_size=0.5)
	#now here i replace the valid data with user dataset
	valid_dataset = patient_dataset
	
	#Statistics on the datset to make sure it is a good shape
	train_stats = train_dataset.describe()
	train_stats.pop("stage")
	train_stats = train_stats.transpose()

	train_label = train_dataset.pop('stage')
	test_label = test_dataset.pop('stage')
	valid_label = valid_dataset.pop('stage')

	def norm(x):
		return (x - train_stats['mean']) / train_stats['std']
	normed_train_data = norm(train_dataset)
	normed_test_data = norm(test_dataset)
	normed_valid_dataset = norm(valid_dataset)

	#Create a SVM Classifier
	model = svm.SVC(C=1, kernel='linear',)
	#Training the model using Training set
	model.fit(normed_train_data, train_label)

	validdata = normed_valid_dataset.tail(14)
	validdata1 = model.predict(validdata)
	
	val1 = validdata1[11]
	val2 = validdata1[12]
	val3 = validdata1[13]

	BPdatabase(val1, val2, val3) 
	print("Accuracy with Patient Data(BloodPressure): ",metrics.accuracy_score(valid_label.tail(14), validdata1))
	return "Task Done"

	
	
	
    

@app.route('/api', methods=['POST'])
def index():
    content = request.json
    print(content)
    systolic = content['systolic']
    diastolic = content['diastolic']
    pid = content['pid'] ; 
    plusrate = content['plusrate']
    glucose = content['glucose'] 
    did = content['pid']
	
    # Fetching Patient data from Database
    mydb = mysql.connector.connect( host="localhost", user="root", password="", database="fog")
    mycursor = mydb.cursor()
    sql = "SELECT * FROM patients WHERE Pid LIKE pid"
    mycursor.execute(sql)
    myresult = mycursor.fetchall()
    age = myresult[0][3]
    height = myresult[0][5]
    weight = myresult[0][6]
    height = height/3.281
    bmi = round(weight / (height ** 2))
    print("your BMI is: ", bmi)
	
    max_HR = 208 - 0.7 * int(age)

	# Scale the BloodPressure Level
    if int(systolic) < 120 and  int(diastolic) < 80:
        bprisk = 1
    elif (int(systolic) >= 120 and int(systolic) < 130) or (int(diastolic) <= 80):
        bprisk = 2
    elif (int(systolic) >= 130 and int(systolic) < 140) or (int(diastolic) >= 80 and int(diastolic) < 90):
        bprisk = 3
    elif (int(systolic) >= 140 and int(systolic)<180) or (int(diastolic) >= 90 and int(diastolic)<120):
        bprisk = 4
    elif int(systolic) >= 180 or int(diastolic) >= 120:
        bprisk = 5
    else:
	    bprisk = "Error"
	
	#Now Check the Diabetes Level
    # glucose = int(glucose)
    if int(glucose) < 33:
	    glrisk = 1
    elif (int(glucose) >= 33 and int(glucose) <= 79):
        glrisk = 2
    elif (int(glucose) >= 80 and int(glucose) < 149):
        glrisk = 3
    elif (int(glucose) >= 150 and int(glucose)<=314):
        glrisk = 4
    elif int(glucose) >= 315:
        glrisk = 5
    else:
	    glrisk = "Error"
		
    print("Current Diabetes Level: ", glrisk)
	
    patientid = pid
    mydb = mysql.connector.connect(host='localhost', database='fog', user='root', password='')
    cursor = mydb.cursor(buffered=True)
    sql_select_query = """select * from records where RPid = %s"""
    cursor.execute(sql_select_query, (patientid,))

    if (cursor.rowcount > 0):
        print("Already Existing Patient")	
    else:
        print("New Patient (Creating Sample Data base on Curent Data of Paitent)")
        hi = int(diastolic)
        hil = int(hi)-15
        lo = int(systolic)
        lol = int(lo)-15
        for i in range(0,15):
            dia = random.randint(hil,hi)
            sys = random.randint(lol,lo)
            if int(sys) < 120 and  int(dia) < 80:
                bp = 1
            elif (int(sys) >= 120 and int(sys) < 130) or (int(dia) <= 80):
                bp = 2
            elif (int(sys) >= 130 and int(sys) < 140) or (int(dia) >= 80 and int(dia) < 90):
                bp = 3
            elif (int(sys) >= 140 and int(sys)<180) or (int(dia) >= 90 and int(dia)<120):
                bp = 4
            elif int(sys) >= 180 or int(dia) >= 120:
                bp = 5
            else:
                bprisk = "Error"
            sql = "INSERT INTO records (RPid, Hiper, Loper, BMI, Plusrate, Status) VALUES (%s, %s, %s, %s, %s, %s)" 
            val = (pid,sys, dia, bmi, plusrate,bp)
            mydb = mysql.connector.connect( host="localhost", user="root", password="", database="fog")
            mycursor = mydb.cursor()
            mycursor.execute(sql, val)
            mydb.commit()
    mydb = mysql.connector.connect( host="localhost", user="root", password="", database="fog")
    mycursor = mydb.cursor()
    sql = "INSERT INTO records (RPid, Hiper, Loper, BMI, Plusrate, Status) VALUES (%s, %s, %s, %s, %s, %s)"
    val = (pid,systolic, diastolic, bmi, plusrate,bprisk)
    mycursor.execute(sql, val)
    mydb.commit()
    print("Saving Current Data on Database")
	
	
	# =============================+++++++++++++++++++==========================++++======================++++++++++++++++++++++=================================
	
	# Now Check the Diabetes and asign the Values in database
	
    mydb = mysql.connector.connect(host='localhost', database='fog', user='root', password='')
    cursor = mydb.cursor(buffered=True)
    sql_select_query = """select * from diabetespatients where pid = %s"""
    cursor.execute(sql_select_query, (pid,))
	
    # print(cursor.rowcount)
    if (cursor.rowcount>0):
        print("Patient Already Existing")
    else:
        print("New Patient (Creating Sample Data base on Curent Data of Paitent)")
        cg = int(glucose)
        lm = int(cg)-15
        bp = int(systolic)
        bpl = int(bp)-5
        for i in range(0,15):
            glc = random.randint(lm,cg)
            bps = random.randint(bpl,bp)
            if glc <=35:
                stg = 1
            elif glc > 35 and glc <=79:
                stg = 2
            elif glc >=80 and glc <=149:
                stg = 3
            elif glc >= 150 and glc <=314:
                stg = 4
            elif glc >= 315:
                stg = 5
            else:
                stg = 0
            mydb = mysql.connector.connect( host="localhost", user="root", password="", database="fog")
            mycursor = mydb.cursor()
            sql = "INSERT INTO diabetespatients (pid, diabetes, bp, bmi, status) VALUES (%s, %s, %s, %s, %s)"
            val = (pid,glc, bps, bmi,stg)
            mycursor.execute(sql, val)
            mydb.commit()
            print("Insert Record")
    mydb = mysql.connector.connect( host="localhost", user="root", password="", database="fog")
    mycursor = mydb.cursor()
    sql1 = "INSERT INTO diabetespatients (pid, diabetes, bp, bmi, status) VALUES (%s, %s, %s, %s, %s)"
    val1 = (pid,glucose, systolic, bmi, glrisk)
    mycursor.execute(sql1, val1)
    mydb.commit()
    print("Enter Current Data on Database done")
	
	
    diabetes(did, glucose, systolic, bmi, glrisk)
    bloodpressure(pid,plusrate, bprisk, age, weight, height, systolic, diastolic, bmi)    

    return jsonify({ "hibp":systolic, "lobp":diastolic, "ma":age, "bp":bprisk, "bm":bmi, "MHR":max_HR})
	
	
@app.route('/dia', methods=['POST'])
def dia():
    content = request.json
    print(content)
    glucose = content['glucose']
    systolic = content['systolic']
    did = content['pid']
	# Fetching Patient data from Database
    mydb = mysql.connector.connect( host="localhost", user="root", password="", database="fog")
    mycursor = mydb.cursor()
    sql = "SELECT * FROM patients WHERE Pid LIKE pid"
    mycursor.execute(sql)
    myresult = mycursor.fetchall()
    age = myresult[0][3]
    height = myresult[0][5]
    weight = myresult[0][6]
    height = height/3.281
    bmi = round(weight / (height ** 2))
    print("your BMI is: ", bmi)
    # max_HR = 208 - 0.7 * int(age)
	# Scale the Diabetes Level
	#  -----------------------------------------------------------------------------
    # (newdata['Glucose'] <=35), #Extremly Low									   |
    # (newdata['Glucose'] >35) & (newdata['Glucose'] <= 79) , #low				   |
    # (newdata['Glucose'] >= 80) & (newdata['Glucose'] <= 149) , #Normal		   |
    # (newdata['Glucose'] >= 150) & (newdata['Glucose'] <= 314) , #Prediabetes	   |
    # (newdata['Glucose'] >= 315) #Very Hight									   |
	#  -----------------------------------------------------------------------------
    
    if int(glucose) < 33:
	    glrisk = 1
    elif (int(glucose) >= 33 and int(glucose) <= 79):
        glrisk = 2
    elif (int(glucose) >= 80 and int(glucose) < 149):
        glrisk = 3
    elif (int(glucose) >= 150 and int(glucose)<=314):
        glrisk = 4
    elif int(glucose) >= 315:
        glrisk = 5
    else:
	    glrisk = "Error"
		
    print("Current Diabetes Level: ", glrisk)
	
    mydb = mysql.connector.connect(host='localhost', database='fog', user='root', password='')
    cursor = mydb.cursor(buffered=True)
    sql_select_query = """select * from diabetespatients where pid = %s"""
    cursor.execute(sql_select_query, (did,))

	
    print(cursor.rowcount)
    if (cursor.rowcount>0):
        print("Patient Already Existing")
    else:
        print("New Patient (Creating Sample Data base on Curent Data of Paitent)")
        cg = int(glucose)
        lm = int(cg)-15
        bp = int(systolic)
        bpl = int(bp)-5
        for i in range(0,15):
            glc = random.randint(lm,cg)
            bps = random.randint(bpl,bp)
            if glc <=35:
                stg = 1
            elif glc > 35 and glc <=79:
                stg = 2
            elif glc >=80 and glc <=149:
                stg = 3
            elif glc >= 150 and glc <=314:
                stg = 4
            elif glc >= 315:
                stg = 5
            else:
                stg = 0
            mydb = mysql.connector.connect( host="localhost", user="root", password="", database="fog")
            mycursor = mydb.cursor()
            sql = "INSERT INTO diabetespatients (pid, diabetes, bp, bmi, status) VALUES (%s, %s, %s, %s, %s)"
            val = (pid,glc, bps, bmi,stg)
            mycursor.execute(sql, val)
            mydb.commit()
            print("Insert Record")
    mydb = mysql.connector.connect( host="localhost", user="root", password="", database="fog")
    mycursor = mydb.cursor()
    sql1 = "INSERT INTO diabetespatients (pid, diabetes, bp, bmi, status) VALUES (%s, %s, %s, %s, %s)"
    val1 = (pid,glucose, systolic, bmi, glrisk)
    mycursor.execute(sql1, val1)
    mydb.commit()
    print("Enter Current Data on Database done")
    diabetes(pid, glucose, systolic, bmi, glrisk)
	
    return jsonify({ "hibp":systolic, "glucose":glucose, "diabetes":glrisk , "BMI": bmi})
    
app.run()