from flask import Flask, flash, redirect, render_template, \
     request, url_for

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import math
import scipy
import os
from scipy.stats import poisson,geom,binom
pd.options.mode.chained_assignment = None  # default='warn'


app = Flask(__name__)

@app.route('/')
def index():
	print("hello world")
	return render_template('index.html')

@app.route("/test" ,methods = ['GET','POST'])
def test():
	select = request.form.get('stateName')
	print(select)
	return execute()
	# return render_template('index.html',val=select)


def execute():

	
		#Loading mandatory dataset 
	#data=pd.read_csv('States Data/25.csv')
	THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
	my_file = os.path.join(THIS_FOLDER, 'States Data/25.csv')
	print(my_file)
	data=pd.read_csv(my_file)

	#created two columns months and year 
	data['Month']=data['Date'].apply(lambda x: (int)(x.split("-")[1]))
	data['Year']=data['Date'].apply(lambda x: (int)(x.split("-")[0]))

	State1_confirmed='WI confirmed'
	State2_confirmed='WV confirmed'
	State1_deaths='WI deaths'
	State2_deaths='WV deaths'
	dataOriginal=data

	#################Step 1#################      Checking for Null values in all the columns
	print(data['Date'].isna().sum())
	print(data[State1_confirmed].isna().sum())
	print(data[State2_confirmed].isna().sum())
	print(data[State1_deaths].isna().sum())
	print(data[State2_deaths].isna().sum())

	#################Step 2#################      Plotting the originial data

	plt.figure()
	p1,=plt.plot(data[State1_confirmed],color='blue',label=State1_confirmed,marker="*",markersize=2)
	p2,=plt.plot(data[State2_confirmed],color='Red',label=State2_confirmed,marker=".",markersize=2)
	plt.ylabel("Number of confirmed cases")
	plt.xlabel("Row number")
	plt.legend(handles=[p1,p2],  bbox_to_anchor=(0.6, 1), loc='upper left')
	plt.grid()

	plt.savefig('static/images/originalPlot1.svg',format='svg', dpi=1200)

	plt.figure()

	p1,=plt.plot(data[State1_deaths],color='blue',label=State1_deaths,marker="*",markersize=2)
	p2,=plt.plot(data[State2_deaths],color='Red',label=State1_deaths,marker=".",markersize=2)
	plt.ylabel("Number of deaths")
	plt.xlabel("Row number")
	plt.legend(handles=[p1,p2], bbox_to_anchor=(0.6, 1), loc='upper left')
	plt.grid()
	plt.savefig('static/images/originalPlot2.svg',format='svg', dpi=1200)


	def handleNegativeCumilativeValues(data):
		states=[State1_confirmed,State2_confirmed,State1_deaths,State2_deaths]
		n=data[states[0]].shape[0]

		for m_state in states:
			for i in range(1,n):
				if data[m_state][i]<data[m_state][i-1]:
					data[m_state][i]=data[m_state][i-1]

		return data

	data=handleNegativeCumilativeValues(data)

	#   Plotting the data after handling negative cumilative data

	plt.figure()
	p1,=plt.plot(data[State1_confirmed],color='blue',label=State1_confirmed,marker="*",markersize=2)
	p2,=plt.plot(data[State2_confirmed],color='Red',label=State2_confirmed,marker=".",markersize=2)
	plt.ylabel("Number of confirmed cases")
	plt.xlabel("Row number")
	plt.legend(handles=[p1,p2], bbox_to_anchor=(0.6, 1), loc='upper left')
	plt.grid()
	plt.savefig('static/images/HandNegCumPlot1.svg',format='svg', dpi=1200)



	plt.figure()
	
	p1,=plt.plot(data[State1_deaths],color='blue',label=State1_deaths,marker="*",markersize=2)
	p2,=plt.plot(data[State2_deaths],color='Red',label=State1_deaths,marker=".",markersize=2)
	plt.ylabel("Number of deaths")
	plt.xlabel("Row number")
	plt.legend(handles=[p1,p2],  bbox_to_anchor=(0.6, 1), loc='upper left')
	plt.grid()
	plt.savefig('static/images/HandNegCumPlot2.svg',format='svg', dpi=1200)



	def getInstantaneousDataFromCumilative(data):
		states=[State1_confirmed,State2_confirmed,State1_deaths,State2_deaths]
		for m_state in states:
			data[m_state]=data[m_state].diff().fillna(data[m_state])
		return data

	data=getInstantaneousDataFromCumilative(data)

	#   Plotting the data after converting cumilative data to instantaneous data(daily data)

	plt.figure()
	p1,=plt.plot(data[State1_confirmed],color='blue',label=State1_confirmed,marker="*",markersize=2)
	p2,=plt.plot(data[State2_confirmed],color='Red',label=State2_confirmed,marker=".",markersize=2)
	plt.ylabel("Number of confirmed cases")
	plt.xlabel("Row number")
	plt.legend(handles=[p1,p2],  bbox_to_anchor=(0.2, 1), loc='upper left')
	plt.grid()
	plt.savefig('static/images/InstPlot1.svg',format='svg', dpi=1200)


	plt.figure()
	p1,=plt.plot(data[State1_deaths],color='blue',label=State1_deaths,marker="*",markersize=2)
	p2,=plt.plot(data[State2_deaths],color='Red',label=State1_deaths,marker=".",markersize=2)
	plt.ylabel("Number of deaths")
	plt.xlabel("Row number")
	plt.legend(handles=[p1,p2], bbox_to_anchor=(0.2, 1), loc='upper left')
	plt.grid()
	plt.savefig('static/images/InstPlot2.svg',format='svg', dpi=1200)


	#################Step 4 #################  Applying Tukey Rule for Outlier detection
	def getOutlierIndexesUsingTukeyRule(data):
    
	    D=np.sort(data)
	    Q1=D[(int)(np.ceil((25/100)*len(D)))-1] 
	    Q3=D[(int)(np.ceil((75/100)*len(D)))-1]
	    IQR=Q3-Q1

	    #Elements are outlier if element E > Q3 + alpha*IQR  OR E < Q1 - alpha*IQR    
	    #Using default value of alpha as discussed in the class=1.5
	    alpha=1.5
	    indexes=[idx for idx,i in enumerate(data) if i> (Q3 + alpha*IQR) or i< (Q1 - alpha*IQR)]

	    return indexes

	def checkForZeroValues(state,outlierIndexes):
		indexes=[idx for idx,i in enumerate(data[state].values[outlierIndexes]) if i!=0]
		return indexes

	#Using Tukey's rule for outlier detection

	#removing outliers based on each state data
	outlierIndexesBasedOnWIconfirmed =getOutlierIndexesUsingTukeyRule(data[State1_confirmed].values)
	outlierIndexesBasedOnWVconfirmed =getOutlierIndexesUsingTukeyRule(data[State2_confirmed].values)
	outlierIndexesBasedOnWIdeaths =getOutlierIndexesUsingTukeyRule(data[State1_deaths].values)
	outlierIndexesBasedOnWVdeaths =getOutlierIndexesUsingTukeyRule(data[State2_deaths].values)


	print("Number of Outliers based on ",State1_confirmed," is ",len(outlierIndexesBasedOnWIconfirmed))
	print("Number of Outliers based on ",State2_confirmed," is ",len(outlierIndexesBasedOnWVconfirmed))
	print("Number of Outliers based on ",State1_deaths," is ",len(outlierIndexesBasedOnWIdeaths))
	print("Number of Outliers based on ",State2_deaths," is ",len(outlierIndexesBasedOnWVdeaths))


	# Check if the indexes does not correspond to the zero values
	print("\n\nNow extracting the outlier indexes which does not corresponds to the zero values \n\n")
	print("Number of Outliers based on ",State1_confirmed," is ",len(checkForZeroValues(State1_confirmed,outlierIndexesBasedOnWIconfirmed)))
	print("Number of Outliers based on ",State2_confirmed," is ",len(checkForZeroValues(State2_confirmed,outlierIndexesBasedOnWVconfirmed)))
	print("Number of Outliers based on ",State1_deaths," is ",len(checkForZeroValues(State1_deaths,outlierIndexesBasedOnWIdeaths)))
	print("Number of Outliers based on ",State2_deaths," is ",len(checkForZeroValues(State2_deaths,outlierIndexesBasedOnWVdeaths)))

	print("\n####Above suggests that we don't have any outliers which corresponds to zero values####")

	print("#################Printing values which are detected as outliers#################")
	print("-------------------------------------")
	print("Outlier values based on ",State1_confirmed," are as follows : ")
	print("Indexes: ",outlierIndexesBasedOnWIconfirmed)
	print("Values: ", data[State1_confirmed].values[outlierIndexesBasedOnWIconfirmed])
	print("-------------------------------------")

	print("-------------------------------------")
	print("Outlier values based on ",State2_confirmed," are as follows : ")
	print("Indexes: ",outlierIndexesBasedOnWVconfirmed)
	print("Values: ", data[State2_confirmed].values[outlierIndexesBasedOnWVconfirmed])
	print("-------------------------------------")

	print("-------------------------------------")
	print("Outlier values based on ",State1_deaths," are as follows : ")
	print("Indexes: ",outlierIndexesBasedOnWIdeaths)
	print("Values: ", data[State1_deaths].values[outlierIndexesBasedOnWIdeaths])
	print("-------------------------------------")

	print("-------------------------------------")
	print("Outlier values based on ",State2_deaths," are as follows : ")
	print("Indexes: ",outlierIndexesBasedOnWVdeaths)
	print("Values: ", data[State2_deaths].values[outlierIndexesBasedOnWVdeaths])
	print("-------------------------------------")

	OutliersUnionIndexes = set(outlierIndexesBasedOnWIconfirmed).union(set(outlierIndexesBasedOnWVconfirmed), set(outlierIndexesBasedOnWIdeaths),set(outlierIndexesBasedOnWVdeaths))

	print("############## Now removing all the outliers##############")
	#Removing the outliers
	data.drop(data.index[list(OutliersUnionIndexes)],inplace=True)


	#   Plotting the data after removing outliers

	plt.figure()
	p1,=plt.plot(data[State1_confirmed],color='blue',label=State1_confirmed,marker="*",markersize=2)
	p2,=plt.plot(data[State2_confirmed],color='Red',label=State2_confirmed,marker=".",markersize=2)
	plt.ylabel("Number of confirmed cases")
	plt.xlabel("Row number")
	plt.legend(handles=[p1,p2], bbox_to_anchor=(0.2, 1), loc='upper left')
	plt.grid()
	plt.savefig('static/images/outlierRemoved1.svg',format='svg', dpi=1200)


	plt.figure()
	p1,=plt.plot(data[State1_deaths],color='blue',label=State1_deaths,marker="*",markersize=2)
	p2,=plt.plot(data[State2_deaths],color='Red',label=State1_deaths,marker=".",markersize=2)
	plt.ylabel("Number of deaths")
	plt.xlabel("Row number")
	plt.legend(handles=[p1,p2],  bbox_to_anchor=(0.2, 1), loc='upper left')
	plt.grid()
	plt.savefig('static/images/outlierRemoved2.svg',format='svg', dpi=1200)






	#Loading data required for this part of the question

	febData2021_WI_confirmed=data[(data['Month']==2) & (data['Year']==2021)][State1_confirmed].values
	marchdata2021_WI_confirmed=data[(data['Month']==3) & (data['Year']==2021)][State1_confirmed].values

	febData2021_WI_deaths=data[(data['Month']==2) & (data['Year']==2021)][State1_deaths].values
	marchdata2021_WI_deaths=data[(data['Month']==3) & (data['Year']==2021)][State1_deaths].values

	febData2021_WV_confirmed=data[(data['Month']==2) & (data['Year']==2021)][State2_confirmed].values
	marchdata2021_WV_confirmed=data[(data['Month']==3) & (data['Year']==2021)][State2_confirmed].values

	febData2021_WV_deaths=data[(data['Month']==2) & (data['Year']==2021)][State2_deaths].values
	marchdata2021_WV_deaths=data[(data['Month']==3) & (data['Year']==2021)][State2_deaths].values

	def getcorrectedSanpleStdDeviationForEntireDataset(State):
		d=data[State].values
		return np.sqrt((np.var(d)*(len(d))/(len(d)-1)))

	#One Sample Tests code

	#Walds Test (Assuming that the data is Poisson distributed and using MLE as estimator)
	def oneSampleWaldsTest(febData,MarchData):
	  theta_knot=np.mean(febData)
	  theta_hat=np.mean(MarchData)
	  se_hat=np.sqrt(theta_hat/len(MarchData)) 
	  z=(theta_hat-theta_knot)/(se_hat)
	  return z

	def oneSampleZtest(febData,MarchData,correctedStd):  
	  meu_knot=np.mean(febData)
	  x_bar=np.mean(MarchData)
	  z=(x_bar-meu_knot)/(correctedStd/np.sqrt(len(MarchData)))
	  return z
	  
	def oneSampleTtest(febData,MarchData,corrected=True):
	  x_bar=np.mean(MarchData)
	  meu_knot=np.mean(febData)

	  if corrected==False:
	      Std=np.sqrt((np.var(MarchData)))
	  else:
	      Std=np.sqrt((np.var(MarchData)*(len(MarchData))/(len(MarchData)-1)))
	      
	  t=(x_bar-meu_knot)/(Std/np.sqrt(len(MarchData)))
	  return t

	OneSampleTestforStateWIconfirmed=[]
	#OneSampleTestforStateWIconfirmed.append("-------------One Sample Test for State WI confirmed -------------")
	
	OneSampleTestforStateWIconfirmed.append("=================== One Sample Walds Test=======================")

	Ho=np.mean(febData2021_WI_confirmed)
	OneSampleTestforStateWIconfirmed.append(f"True Hypothesis Ho={Ho}")
	OneSampleTestforStateWIconfirmed.append(f"Alternative Hypothesis H1!={Ho}")

	Wstat=oneSampleWaldsTest(febData2021_WI_confirmed,marchdata2021_WI_confirmed)

	Zalphaby2=1.96
	OneSampleTestforStateWIconfirmed.append(f"Value of one Sample Wald's test for the state WI confirmed case is {Wstat},  and the value of alpha=0.05 and Zalphaby2 is {Zalphaby2}")

	if(np.abs(Wstat)>Zalphaby2):
	  OneSampleTestforStateWIconfirmed.append(f"We reject the true hypothesis, i.e Ho not equal to {Ho}")
	else:
	  OneSampleTestforStateWIconfirmed.append(f"We accept the true hypothesis, i.e Ho is equal to {Ho}")

	OneSampleTestforStateWIconfirmed.append("==============================================================")

	OneSampleTestforStateWIconfirmed.append("=================== One Sample Z Test=======================")

	Ho=np.mean(febData2021_WI_confirmed)
	OneSampleTestforStateWIconfirmed.append(f"True Hypothesis Ho={Ho}")
	OneSampleTestforStateWIconfirmed.append(f"Alternative Hypothesis H1!={Ho}")
	Zalphaby2=1.96

	Zstat=oneSampleZtest(febData2021_WI_confirmed,marchdata2021_WI_confirmed,getcorrectedSanpleStdDeviationForEntireDataset(State1_confirmed))
	OneSampleTestforStateWIconfirmed.append(f"Value of one Sample Zstat test for the state WI confirmed case is {Zstat}  and the value of alpha=0.05 and Zalphaby2 is {Zalphaby2}")

	if(np.abs(Zstat)>Zalphaby2):
	  OneSampleTestforStateWIconfirmed.append(f"We reject the true hypothesis, i.e Ho not equal to {Ho}")
	else:
	  OneSampleTestforStateWIconfirmed.append(f"We accept the true hypothesis, i.e Ho is equal to {Ho}")
	OneSampleTestforStateWIconfirmed.append("==============================================================")

	OneSampleTestforStateWIconfirmed.append("=================== One Sample T Test=======================")

	Ho=np.mean(febData2021_WI_confirmed)
	OneSampleTestforStateWIconfirmed.append(f"True Hypothesis Ho={Ho}")
	OneSampleTestforStateWIconfirmed.append(f"Alternative Hypothesis H1!={Ho}")

	OneSampleTestforStateWIconfirmed.append(f"Here the value of the number of days according to the data is n= {len(febData2021_WI_confirmed)}")

	Tstat=oneSampleTtest(febData2021_WI_confirmed,marchdata2021_WI_confirmed)
	tnMinusOneAlphaBy2=2.10 # have used T18,0.025 as n=19 in our case

	OneSampleTestforStateWIconfirmed.append(f"Value of one Sample Tstat test for the state WI confirmed case is {Tstat} and the value of alpha=0.05,n-1=19-1=18 and  tn-1,α/2:  t18,0.025 is {tnMinusOneAlphaBy2}")

	if(np.abs(Tstat)>tnMinusOneAlphaBy2):
	  OneSampleTestforStateWIconfirmed.append(f"We reject the true hypothesis, i.e Ho not equal to {Ho}")
	else:
	  OneSampleTestforStateWIconfirmed.append(f"We accept the true hypothesis, i.e Ho is equal to {Ho}")
	OneSampleTestforStateWIconfirmed.append("==============================================================")

	#OneSampleTestforStateWIconfirmed.append("\n---------------------------------------------------------------")



	def waldsTestForTwoPopulation(data1,data2):
	  p1=np.mean(data1)
	  p2=np.mean(data2)
	  delta_hat=p1-p2
	  se_hat=np.sqrt((p1/data1.shape[0])+(p2/data2.shape[0]))  
	  z=delta_hat/se_hat
	  return np.abs(z)


	def twoSampleUnpairedT_Test(data1,data2):
	  p1=np.mean(data1)
	  p2=np.mean(data2)
	  delta_hat=p1-p2
	  corrvarianceata1=((np.var(data1)*(len(data1))/(len(data1)-1)))
	  corrvarianceata2=((np.var(data2)*(len(data2))/(len(data2)-1)))
	  denominator=np.sqrt((corrvarianceata1/len(data1)) + (corrvarianceata2/len(data2)))
	  T=delta_hat/denominator
	  return np.abs(T)


	TwoPopulationTestforStateWIconfirmed=[]
	#TwoPopulationTestforStateWIconfirmed.append("-------------Two population Test for State WI Confirmed for  Feb'21 and March '21-------------")

	TwoPopulationTestforStateWIconfirmed.append("=================== Two population test for WALD's Test=======================")

	TwoPopulationTestforStateWIconfirmed.append("Ho=mean of confirmed cases for feb'21 and march'21 is equal for the state WI")
	TwoPopulationTestforStateWIconfirmed.append("H1=mean of confirmed cases for feb'21 and march'21 is not equal for the state WI")


	Wstat=waldsTestForTwoPopulation(febData2021_WI_confirmed,marchdata2021_WI_confirmed)
	Zalphaby2=1.96

	TwoPopulationTestforStateWIconfirmed.append(f"Value of Wald's test is {Wstat} and the value of alpha=0.05 and Zalphaby2 is {Zalphaby2}")

	if(np.abs(Wstat)>Zalphaby2):
	  TwoPopulationTestforStateWIconfirmed.append("We reject the true hypothesis, i.e mean of confirmed cases for feb'21 and march'21 is not equal for the state WI")
	else:
	  TwoPopulationTestforStateWIconfirmed.append("We accept the true hypothesis, i.e mean of confirmed cases for feb'21 and march'21 is equal for the state WI")

	TwoPopulationTestforStateWIconfirmed.append("==============================================================")



	TwoPopulationTestforStateWIconfirmed.append("=================== Two population test for T Test=======================")

	TwoPopulationTestforStateWIconfirmed.append("Ho=mean of confirmed cases for feb'21 and march'21 is equal for the state WI")
	TwoPopulationTestforStateWIconfirmed.append("H1=mean of confirmed cases for feb'21 and march'21 is not equal for the state WI")

	TwoPopulationTestforStateWIconfirmed.append(f"Here the value of the number of days according to the data is m= {len(febData2021_WI_confirmed)}")
	TwoPopulationTestforStateWIconfirmed.append(f"Here the value of the number of days according to the data is n= {len(marchdata2021_WI_confirmed)}")
	TwoPopulationTestforStateWIconfirmed.append("So, m+n-2=40 in our case which has to be used for finding the constant for T test")


	Tstat=twoSampleUnpairedT_Test(febData2021_WI_confirmed,marchdata2021_WI_confirmed)
	TConst=2.02  # have used T40,0.025 as m=19 n=23 in our case and we have used Tm+n-2,

	TwoPopulationTestforStateWIconfirmed.append(f"Value of T test stat is {Tstat} and the value of alpha=0.05,m+n-2=19+23-2=40 and  tm+n-2,α/2:  t40,0.025 is {TConst}")

	if(np.abs(Tstat)>TConst):
	  TwoPopulationTestforStateWIconfirmed.append("We reject the true hypothesis, i.e mean of confirmed cases for feb'21 and march'21 is not equal for the state WI")
	else:
	  TwoPopulationTestforStateWIconfirmed.append("We accept the true hypothesis, i.e mean of confirmed cases for feb'21 and march'21 is equal for the state WI")

	TwoPopulationTestforStateWIconfirmed.append("==============================================================")





	# Loading data for this part of the question
	octData2020_WI_confirmed=data[(data['Month']==10) & (data['Year']==2020)][State1_confirmed].values
	novData2020_WI_confirmed=data[(data['Month']==11) & (data['Year']==2020)][State1_confirmed].values
	decData2020_WI_confirmed=data[(data['Month']==12) & (data['Year']==2020)][State1_confirmed].values
	total_WI_confirmed=np.insert(octData2020_WI_confirmed,octData2020_WI_confirmed.shape[0],novData2020_WI_confirmed)
	total_WI_confirmed=np.insert(total_WI_confirmed,total_WI_confirmed.shape[0],decData2020_WI_confirmed)


	octData2020_WI_deaths=data[(data['Month']==10) & (data['Year']==2020)][State1_deaths].values
	novData2020_WI_deaths=data[(data['Month']==11) & (data['Year']==2020)][State1_deaths].values
	decData2020_WI_deaths=data[(data['Month']==12) & (data['Year']==2020)][State1_deaths].values
	total_WI_deaths=np.insert(octData2020_WI_deaths,octData2020_WI_deaths.shape[0],novData2020_WI_deaths)
	total_WI_deaths=np.insert(total_WI_deaths,total_WI_deaths.shape[0],decData2020_WI_deaths)


	octData2020_WV_confirmed=data[(data['Month']==10) & (data['Year']==2020)][State2_confirmed].values
	novData2020_WV_confirmed=data[(data['Month']==11) & (data['Year']==2020)][State2_confirmed].values
	decData2020_WV_confirmed=data[(data['Month']==12) & (data['Year']==2020)][State2_confirmed].values
	total_WV_confirmed=np.insert(octData2020_WV_confirmed,octData2020_WV_confirmed.shape[0],novData2020_WV_confirmed)
	total_WV_confirmed=np.insert(total_WV_confirmed,total_WV_confirmed.shape[0],decData2020_WV_confirmed)


	octData2020_WV_deaths=data[(data['Month']==10) & (data['Year']==2020)][State2_deaths].values
	novData2020_WV_deaths=data[(data['Month']==11) & (data['Year']==2020)][State2_deaths].values
	decData2020_WV_deaths=data[(data['Month']==12) & (data['Year']==2020)][State2_deaths].values
	total_WV_deaths=np.insert(octData2020_WV_deaths,octData2020_WV_deaths.shape[0],novData2020_WV_deaths)
	total_WV_deaths=np.insert(total_WV_deaths,total_WV_deaths.shape[0],decData2020_WV_deaths)

	

	OneSampleKSTest=[]
	#one sample KS test with Poisson, Geometric, and Binomial for both the states and for both cases,i.e confirmed and deaths cases
	

	def ks_test(firstStateData,secondStateData,dist,state1,state2):

		OneSampleKSTest.append(f"Ho: {state1} ,distribution is equivalent to {state2} distribution incase when the given distribution is {dist}")
		OneSampleKSTest.append(f"H1: {state1} ,distribution is not equivalent to {state2} distribution incase when the given distribution is {dist}")
	
		threshold=0.05
		x=np.sort(secondStateData)

		if dist=="Poisson":
		    lambdaMME=np.mean(firstStateData)
		    fyx= [poisson.cdf(a, lambdaMME) for a in x]

		if dist=="Geometric":
		    pMME=1/np.mean(firstStateData)
		    fyx= [geom.cdf(a, pMME) for a in x]

		if dist=="Binomial":
		    mean=np.mean(firstStateData)
		    SummationXiSquareByN=sum([a*a for a in firstStateData])/len(firstStateData)
		    pMME=1-((SummationXiSquareByN-(mean*mean))/mean)
		    nMME=(int)(mean/pMME)
		    fyx= [binom.cdf(a, nMME, pMME) for a in x]

		fx_minus=[ i/len(x) for i in range(len(x))]
		fx_plus=[ i/len(x) for i in range(1,len(x)+1)]
		fx_minus_fyx_diff_abs=[abs(fx_minus[i]-fyx[i]) for i in range(len(x))]
		fx_plus_fyx_diff_abs=[abs(fx_plus[i]-fyx[i]) for i in range(len(x))]

		maxDiff=max(fx_minus_fyx_diff_abs+fx_plus_fyx_diff_abs)
		OneSampleKSTest.append(f"{dist}: maximum difference val {maxDiff}" )
		if(maxDiff>threshold):
			OneSampleKSTest.append(f"Rejecting the Hypothesis,i.e {state1} distribution is not equivalent to {state2} distribution incase when the given distribution is {dist}")
		else:
			OneSampleKSTest.append(f"Accepting the Hypothesis,i.e {state1} distribution is equivalent to {state2} distribution incase when the given distribution is {dist}")
	    
		return (pd.DataFrame({'x':x,'fyx':fyx,'fx_minus':fx_minus,'fx_plus':fx_plus,'fx_minus_fyx_diff_abs':fx_minus_fyx_diff_abs,'fx_plus_fyx_diff_abs':fx_plus_fyx_diff_abs}))
	
	def ecdf(a,thresh):
	    ans=sum([1 for i in a if i<thresh])/len(a)
	    return ans

	def ks_test_two_population(firstStateData,secondStateData,state1,state2):

		TwoSampleKSTest.append(f"Ho: {state1} distribution is equivalent to {state2} distribution")
		TwoSampleKSTest.append(f"H1: {state1} distribution is not equivalent to {state2} distribution")

		threshold=0.05
		data1=np.sort(firstStateData)
		data2=np.sort(secondStateData)

		if(len(data1)<=len(data2)):
		  d2=data1
		  d1=data2
		elif(len(data2)<len(data1)):  
		  d1=data1
		  d2=data2

		x=d2
		y=d1
	    
		fd1x=[ecdf(y,i) for i in x]
	    
		fd2x_minus=[ i/len(x) for i in range(len(x))]
		fd2x_plus=[ i/len(x) for i in range(1,len(x)+1)]

		fd1x_minus_fd2x_minus_diff_abs=[abs(fd2x_minus[i]-fd1x[i]) for i in range(len(x))]
		fd1x_minus_fd2x_plus_diff_abs=[abs(fd2x_plus[i]-fd1x[i]) for i in range(len(x))]

		maxDiff=max(fd1x_minus_fd2x_minus_diff_abs+fd1x_minus_fd2x_plus_diff_abs)
		TwoSampleKSTest.append(f"maximum difference val {maxDiff}" )
		if(maxDiff>threshold):
			TwoSampleKSTest.append(f"Rejecting the Hypothesis,i.e {state1} distribution is not equivalent to {state2} distribution")
		else:
			TwoSampleKSTest.append(f"Accepting the Hypothesis,i.e {state1} distribution is equivalent to {state2} distribution")


		return (pd.DataFrame({'x':x,'fd1x':fd1x,'fd2x_minus':fd2x_minus,'fd2x_plus':fd2x_plus,'fd1x_minus_fd2x_minus_diff_abs':fd1x_minus_fd2x_minus_diff_abs,'fd1x_minus_fd2x_plus_diff_abs':fd1x_minus_fd2x_plus_diff_abs}))

	#OneSampleKSTest.append("################One Sample K-S Test for WI_confirmed and WV_confirmed################\n")
	# print("\n\n----------------------------Poisson------------------------------\n\n")
	# print(ks_test(total_WI_confirmed,total_WV_confirmed,"Poisson","WI_confirmed","WV_confirmed"))
	#OneSampleKSTest.append("\n\n----------------------------Geometric------------------------------\n\n")
	df_oneSampleKSTest=ks_test(total_WI_confirmed,total_WV_confirmed,"Geometric","WI_confirmed","WV_confirmed")
	# print("\n\n-----------------------------Binomial----------------------------\n\n")
	# print(ks_test(total_WI_confirmed,total_WV_confirmed,"Binomial","WI_confirmed","WV_confirmed"))
	# print("\n\n\----------------------------------------------------------\n\n")






	TwoSampleKSTest=[]

	#print("################Two Sample K-S Test ################\n")

	#TwoSampleKSTest.append("-------------------------For WI_confirmed and WV_confirmed---------------------------------\n\n")
	df_TwoSampleKSTest=ks_test_two_population(total_WI_confirmed,total_WV_confirmed,"WI_confirmed","WV_confirmed")
	#print("----------------------------------------------------------\n\n")



	def returnKpermutations(data,k):
	  
	  m_data=data.copy()

	  returnList=[]
	  for i in range(k):
	    idx=np.random.permutation(len(m_data))
	   
	    append_list=[m_data[m_idx] for m_idx in idx]
	    returnList.append(append_list)
	  return returnList
	  
	def permutation_test(data1,data2,state1,state2,k=1000):

	    threshold=0.05
	    PermutationList.append(f"Ho: {state1} distribution is equivalent to {state2} distribution")
	    PermutationList.append(f"H1: {state1} distribution is not equivalent to {state2} distribution")

	  
	    n=len(data1)+len(data2)
	    Tobs=abs(np.mean(data1)-np.mean(data2))
	    PermutationList.append(f"Tobs:{Tobs}")
	    permutations=returnKpermutations(list(np.insert(data1,data1.shape[0],data2)),k)
	    
	    
	    Separate_permutations=[(i[:-len(data2)],i[-len(data2):]) for i in permutations]
	    
	    Ti=[abs(np.mean(list(j[0]))-np.mean(list(j[1]))) for i,j in enumerate(Separate_permutations)]
	    TiGreaterThanTobs=[1 if i>Tobs else 0 for i in Ti ]

	    returnval=pd.DataFrame({'Ti':Ti,"TiGreaterThanTobs":TiGreaterThanTobs}) 
	    pval=sum(returnval['TiGreaterThanTobs'])/len(returnval['TiGreaterThanTobs'])

	    PermutationList.append(f"Pval is : {pval}")
	    if(pval<threshold):
	      PermutationList.append(f"Rejecting the Hypothesis,i.e {state1} distribution is not equivalent to {state2} distribution")
	    else:
	      PermutationList.append(f"Accepting the Hypothesis,i.e {state1} distribution is equivalent to {state2} distribution")
	    
	    return returnval,pval
	    


	#print("################permutation_test ################\n")
	PermutationList=[]
	#print("--------------------For WI_deaths and WV_deaths------------------------------------\n\n")
	df_PermutationList,pval=permutation_test(total_WI_deaths,total_WV_deaths,"WI_deaths","WV_deaths")
	


	# print("----------------------------------------------------------\n\n")


	# print("-------------------------For WI_confirmed and WV_confirmed---------------------------------\n\n")
	# table,pval=permutation_test(total_WI_confirmed,total_WV_confirmed,"WI_confirmed","WV_confirmed")
	# print(table)

	# #note print the result whether it is rejected or not and also check if it is correct
	# print("----------------------------------------------------------\n\n")


	return render_template('index.html',flag=True,PermutationList=PermutationList,TwoSampleKSTest=TwoSampleKSTest,OneSampleKSTest=OneSampleKSTest,TwoPopulationTestforStateWIconfirmed=TwoPopulationTestforStateWIconfirmed,OneSampleTestforStateWIconfirmed=OneSampleTestforStateWIconfirmed,outlierRemoved1='static/images/outlierRemoved1.svg',outlierRemoved2='static/images/outlierRemoved2.svg',InstPlot1='static/images/InstPlot1.svg',InstPlot2='static/images/InstPlot2.svg',HandNegCumPlot1='static/images/HandNegCumPlot1.svg',HandNegCumPlot2='static/images/HandNegCumPlot2.svg',originalPlot2='static/images/originalPlot2.svg',originalPlot1='static/images/originalPlot1.svg',df_oneSampleKSTest=[df_oneSampleKSTest.to_html()], df_TwoSampleKSTest=[df_TwoSampleKSTest.to_html()],df_PermutationList=[df_PermutationList.to_html()],tables=[dataOriginal.to_html(classes='datatablesSimple')], )


if __name__=='__main__':
    app.run(debug=True,port=4455)
    



