
# coding: utf-8

# # Lending Club Loan Data Analysis
# 
# 

# <div class="span5 alert alert-info">
# <h3>Introduction</h3>
# <b>Assessing the risk involved in a loan application is one of the most important concerns for any lending institution. Better risk assessment techniques are always in demand which could help these institutions to take better decisions and prevent losses by predicting likely defaulters.
# 
# Through this project we intend to extract patterns from Lending Club loan approved dataset and build a model to predict the likely loan defaulters by using classification algorithms.The historical data of the customers like their annual income, loan
# amount, employment length, home ownership status, verification staus will be used for analysis. 

# <div class="span5 alert alert-info">
# <h3>Importing Libraries</h3>
# <b>

# In[1]:

import numpy as np
import pandas as pd 
import re  # Regex 
pd.options.mode.chained_assignment = None

# Visualizations
import plotly.plotly as py
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
from ggplot import *
get_ipython().magic('matplotlib notebook')
get_ipython().magic('matplotlib inline')
import seaborn as sns

#Missing Values
import missingno as msno

#SQL
import pandasql as pdsql
from pandasql import sqldf


# <div class="span5 alert alert-info">
# <h3>Reading Data</h3>
# 
# <b>

# In[2]:

custdata14 = pd.read_csv('C:\LoanStats14.csv',low_memory=False)


# In[3]:

custdata14.shape


# In[4]:

custdata15 = pd.read_csv('C:\LoanStats15.csv',low_memory=False)


# In[5]:

custdata15.shape


# In[6]:

custdata = pd.concat([custdata14,custdata15])


# In[7]:

custdata.shape


# <div class="span5 alert alert-info">
# <h3>Quick Data Exploration</h3>
# 
# <b>

# In[8]:

custdata.info()


# In[9]:

custdata.columns.tolist()#List of features


# In[10]:

custdata.head(3)


# In[11]:

custdata.describe()


# Quick inferences based on the above.
# 1. loan_amnt and funded_amnt are having same values.
# 2. Maximum loan_amnt is 35000
# 3. Member_id ,url fields are not populated.
# 4. Many records have NaN entries.

# <div class="span5 alert alert-info">
# <h3>Identifying and visualizing missing values</h3>
# <b>

# In[12]:

# custdata.apply(lambda x: sum(x.isnull()),axis=0) 


# In[13]:

msno.bar(custdata)#columns with missing values


# Above bar chart helps identifying columns with missing values in the dataset. It appears that there are quite a number of columns with missing values.
# For example:- "url" column is empty. "desc" column has also some missing values. "months since last delinq", "monthsince last record" are not populated. We will be dropping the columns which have missing values upto 75-80%.

# <div class="span5 alert alert-info">
# <h3>Handling missing values</h3>
# <b>

# In[14]:

custdata = custdata.iloc[:, :-11]#Removed last 11 columns with sparse data


# In[15]:

custdata.shape


# #### Removing columns with nan values
# Removed annual_inc_joint,dti_joint,verification_status_joint

# In[16]:

custdata.drop(['annual_inc_joint','dti_joint','verification_status_joint'],1, inplace=True)


# <div class="span5 alert alert-info">
# <h3>Selecting features</h3>
# 
# We will be focusing on the subset of the data and base our analysis on a selected set of variables. I explored the features in batches, dropped the irrelevant columns which will not be needed in the context of intended analysis. I will be pulling those columns in a separate df 'loandata'.
# <b>

# In[17]:

loandata = custdata[['loan_amnt','funded_amnt','int_rate','emp_length','loan_status','home_ownership','grade','annual_inc','verification_status','term','dti','revol_bal','total_acc']]


# In[18]:

loandata.dropna().info()


# #### Now we will be pulling the records which have NaN entries. We will drop all those rows which have NaN entries for all the columns.

# In[19]:

nan_rows = loandata[loandata.isnull().T.any().T]
print(nan_rows)


# In[20]:

### loandata.info()
loandata = loandata.dropna(how='all') 
loandata.info()


# In[21]:

loandata.apply(lambda x: sum(x.isnull()),axis=0) 


# <div class="span5 alert alert-info">
# <h3>Data Munging</h3>
# 
# We will be analysing each of selected features against our target variable and make adjustments , numerical conversion if required.
# <b>

# ###  Analysing  "loan_status"

# There are 7 distinct Loan Status categories.  Lets look at the distribution:-

# #### Finding no of applicants per loan status

# In[22]:

loan_status_dist=pd.DataFrame(loandata['loan_status'].value_counts())
loan_status_dist.reset_index(inplace=True)
loan_status_dist.columns=['Status_Category','No. of applicants']
loan_status_dist


# In[23]:

colors = ['purple', 'green', 'y', 'pink','orange','cyan']
loandata.loan_status.value_counts().plot(kind='bar',color=colors,alpha=.20,legend=True,figsize = [16,4])
plt.title('Loan Distribution by Loan Status', fontsize = 10)


# In[24]:

from ggplot import *
# ggplot(loandata, aes(loan_amnt, col = grade)) + geom_histogram(bins = 50) + facet_grid(grade)
gg = ggplot(loandata, aes('loan_amnt',col='loan_status')) + geom_histogram(binwidth=300)+facet_grid('loan_status')
print (gg)


# #### We see a variation of loan amount in the Charged off and Late (31-120 days) status category. We will set the premise of loan prediction on this variable.

# #### Cleaning loan_status

# * 0 = (Charged off, Default)
# * 1 = (Fully -Paid) 
# * 2 = (current,late,in grace-period) 
# We will filter all the records with status = current/late/in-grace period. 'late payments' and 'in-grace period' status can turn up or down. After filtering we reduced the records to 319688.

# In[25]:

loandata['loan_status_clean'] = loandata['loan_status'].map({'Current': 2, 'Fully Paid': 1, 'Charged Off':0, 'Late (16-30 days)':2, 'In Grace Period': 2, 'Late (31-120 days)': 2, 'Default': 0})
loandata = loandata[loandata.loan_status_clean != 2]# Getting rid of current loan from the dataset  
loandata["loan_status_clean"] = loandata["loan_status_clean"].apply(lambda loan_status_clean: 0 if loan_status_clean == 0 else 1)
loandata.head(2)
# loandata['loan_new'] = loandata['loan_status'].map({'Current': 'X', 'Fully Paid': 'Y', 'Charged Off':'N', 'Late(31-120 days)':'X', 'In Grace Period': 'X', 'Late(16-30 days)': 'X', 'Default': 'N'})
# loandata = loandata[loandata.loan_new != 'X']


# In[26]:

pysql = lambda q: pdsql.sqldf(q, globals())

str1= """SELECT loan_status_clean, loan_status, count(*) as count_of_loan_issued from loandata group by loan_status_clean,loan_status """
df1 = pysql(str1)
# df1.set_index('loan_status',inplace = True)
df1.head(7)
# loandata.head()


# In[27]:

loandata.loan_status_clean.value_counts()


# Looks like  'Paid' and 'Charged/Defaulted' loans make up a ratio of approximately 3:1. We will label them 'Good' and 'Bad'. Below bar graph presents the distribution.

# In[28]:


get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


# In[29]:

colors = ['green','orange']
loandata.loan_status_clean.value_counts().plot(kind='bar',color = colors,alpha=.30,figsize = [16,4])
plt.xlabel("Good Loan = 1 Bad Loan = 0")
plt.ylabel("Cnt")
plt.title("Good Loan vs Bad Loan Count")
# plt.hist(list(fil_loandata['loan_status']))
plt.savefig("2013-2014-broader-bad-loan-def.png")


# In[30]:

loandata.shape


# In[31]:

loandata.info()


# ### Analysing "verification status"

# verification_status	Indicates if income was verified by LC, not verified, or if the income source was verified.
# 

# In[32]:

loandata.head()
loandata[ (loandata['verification_status'].isnull())]##checking if the column has null values


# In[33]:

verification_status_dist=pd.DataFrame(loandata['verification_status'].value_counts())
verification_status_dist.reset_index(inplace=True)
verification_status_dist.columns=['Status_Category','No. of applicants']
verification_status_dist


# In[34]:

import plotly
plotly.offline.init_notebook_mode()
trace=go.Pie(labels=verification_status_dist['Status_Category'],values=verification_status_dist['No. of applicants'])
iplot([trace])


# Alsmost 30% of the records seem to be 'Not-Verified'. It's appropriate to put' 'Verified' and 'Source Verified' records in one bucket and 'Not-verified' into the other.

# #### Cleaning Verification Status

# * Verified or Source Verified= 1
# * Not Verified = 0

# In[35]:

loandata['verification_status_clean'] = loandata['verification_status'].map({'Source Verified': 1, 'Verified': 1, 'Not Verified':0})
loandata["verification_status_clean"] = loandata["verification_status_clean"].apply(lambda verification_status_clean: 0 if verification_status_clean == 0 else 1)
loandata.head(2)


# In[36]:

loandata.verification_status_clean.value_counts()


# In[37]:

# t1 = t/t.ix["Ctotal","Rtotal"]
t1 = pd.crosstab(index=loandata["verification_status_clean"],columns=loandata["loan_status_clean"], margins = True)
t1.columns = ["Bad Loan","Good Loan","Rtotal"]
t1.index = ["Not-Verified","Verified","Ctotal"]
t1



# In[38]:

t2 = pd.crosstab(index=loandata["verification_status_clean"],columns=loandata["loan_status_clean"])
t2.columns = ["Bad Loan","Good Loan"]
t2.index = ["Not-Verified","Verified"]
t2.plot(kind='bar', stacked=True, color=['orange','green'], grid=False)


# ### Analysing "term"

# Term indicates number of payments on the loan. Values are in months and can be either 36 or 60. We will see if term has any impact in a loan getting paid defaulted.

# In[39]:

loandata[ (loandata['term'].isnull())]##checking if the column has null values


# In[40]:

term_dist=pd.DataFrame(loandata['term'].value_counts())
term_dist.reset_index(inplace=True)
term_dist.columns=['term','No. of applicants']
term_dist


# In[41]:

pysql = lambda q: pdsql.sqldf(q, globals())

str1= """SELECT loan_status_clean as defaulted_loan, term, grade,count(*) as count_of_loan_issued 
from loandata 
group by loan_status_clean,term,grade """
df1 = pysql(str1)
# df1.set_index('loan_status',inplace = True)
df1.head(14)
# loandata.head()


# ### Cleaning "term"

# 36 months = '1',
# 60 months = '0'

# In[42]:

loandata['term_clean'] = loandata['term'].map({' 36 months': 1, ' 60 months': 0})
loandata["term_clean"] = loandata["term_clean"].apply(lambda term_clean: 0 if term_clean == 0 else 1)
loandata.head(2)
# loandata.drop(['term_cln'],1, inplace=True)


# In[43]:

t1 = pd.crosstab(index=loandata["term_clean"],columns=loandata["loan_status_clean"], margins = True)
t1.columns = ["Bad Loan","Good Loan","Rtotal"]
t1.index = ["60 months","36 months","Ctotal"]
t0 = t1/t1.ix["Ctotal","Rtotal"]*100
t0
t1


# In[44]:

t2 = pd.crosstab(loandata['term_clean'], loandata['loan_status_clean'])
t2.columns = ["Bad Loan","Good Loan"]
t2.index = ["60 months","36 months"]
print ("Term vs Loan")

t2.plot(kind='bar', stacked=True, color=['orange','g'], grid=False)


# 18 % of the 36 months term loans have turned bad vs 36 % of loan with 60 months term.We will check it's predictive power in the latter section.

# ### Analysing "emp_length"

# It indicates employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. It's intuitive enough to assume that the loan payments would continue as long as individual continues to work but I would like to explore if number of years have anything to do with the loan defaults emphasizing more on the lower range(1-3 years) of emp_length.

# In[45]:

loandata[ (loandata['emp_length'].isnull())]##checking if the column has null values


# In[46]:

loandata["emp_length"].unique()


# In[47]:

###EMP_LENGTH
print(loandata.emp_length.value_counts())
loandata.emp_length.unique().shape


# Lets look at the "n/a" records against good/bad loan counts.

# In[48]:

pysql = lambda q: pdsql.sqldf(q, globals())

str1= """SELECT emp_length,loan_status_clean,  count(*) as count_of_loan_issued 
from loandata
where emp_length = 'n/a'
group by emp_length,loan_status_clean """
df1 = pysql(str1)
# df1.set_index('loan_status',inplace = True)
df1.head(17)
# loandata.head()


# Almost half of 'n/a' records resulted into "bad loans"

# In[49]:

# mean_emp_length_clean = loandata[loandata.emp_length.notnull()].emp_length.mean()
# print(mean_emp_length_clean)


# ### Cleaning Emp_length

# Steps involve:-
# filling n/a with NaN.
# eliminating NaN  by replacing it with 0.
# getting rid of < using regex.
# combining < 1 and 1 years together.

# In[50]:

print(loandata.emp_length.value_counts())


# In[51]:

loandata['emp_length'] = loandata.emp_length.str.replace('n/a','Not specified')


# In[52]:

# eliminating n/a from emp_length
 
# loandata.replace('n/a', np.nan,inplace=True) #replacing n/a with np.nan
# loandata.emp_length_clean.fillna(value = 0,inplace=True)# filling 0 for np.nan
# loandata['emp_length_clean'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True) # replacing < with empty space
# loandata['emp_length_clean'] = loandata.emp_length_clean.map(float)
# print(loandata.emp_length_clean.value_counts())
# loandata.emp_length_clean.unique().shape
# loandata.head(1)

loandata['emp_length_clean'] = loandata.emp_length.str.replace('+','')
loandata['emp_length_clean'] = loandata.emp_length_clean.str.replace('<','')
loandata['emp_length_clean'] = loandata.emp_length_clean.str.replace('years','')
loandata['emp_length_clean'] = loandata.emp_length_clean.str.replace('year','')
loandata['emp_length_clean'] = loandata.emp_length_clean.str.replace('Not specified','99')
loandata['emp_length_clean'] = loandata.emp_length_clean.str.replace(' 1','1')
print(loandata.emp_length_clean.value_counts())


# In[53]:

t1 = pd.crosstab(index=loandata["emp_length_clean"],columns=loandata["loan_status_clean"], margins = True)
t1.columns = ["Bad Loan%","Good Loan%","Rtotal%"]
t1.index = ['Not spec', '<1  year', '2 years', '3 years', '4 years', '5 years', '6 years', ' 7 years' ,'8 years', '9 years', '10+ years','Ctotal%']
t0 = t1/t1.ix["Ctotal%","Rtotal%"]*100
t0


# In[54]:

t2 = pd.crosstab(loandata['emp_length_clean'], loandata['loan_status_clean'])
t2.columns = ["Bad Loan","Good Loan"]
t2.index = ['Not spec', '<=1 year', '2 years', '3 years', '4 years', '5 years', '6 years'
            , ' 7 years' ,'8 years', '9 years', '10+ years']

t2.plot(kind='bar', stacked=True, color=['orange','green'], grid=False)


# It's interesting to see the % of bad loans against 1 year pill vs others . We will explore whether or not it holds predictive power while building our model.

# ### Analysing 'home_ownership'

# "home ownership" status provided by the borrower during registration or obtained from the credit report. Values are: RENT, OWN, MORTGAGE, OTHER

# In[55]:

loandata[ (loandata['home_ownership'].isnull())]##checking if the column has null values


# In[56]:

loandata.home_ownership.unique().tolist()


# In[57]:

loandata['home_ownership']=loandata['home_ownership'].apply(lambda x: 'OTHER' if (x == 'NONE' or x=='ANY') else x)


# In[58]:

home_ownership_dist=pd.DataFrame(loandata.home_ownership.value_counts()) # 1 omitted as it contains missing data
home_ownership_dist.reset_index(inplace=True)
home_ownership_dist.columns=['Home Ownership','Number of applicants']
print(home_ownership_dist)

# print(loandata_s.home_ownership.value_counts())
# loandata_s.home_ownership.unique().shape


# We have most number of application with ho_status 'Mortgage' status in this dataset and just one record with 'OTHER' which is odd. We will look at the loan_status for this record below.

# In[59]:

colors = ['purple', 'green', 'y', 'pink','orange','blue']
loandata.home_ownership.value_counts().plot(kind='bar',alpha=.20,legend=True, color = colors)
plt.title('Loan Distribution by Home Ownership', fontsize = 10)


# Lets look at the loan status for the "OTHER" category

# In[60]:

pysql = lambda q: pdsql.sqldf(q, globals())

str1= """SELECT home_ownership, loan_status_clean, Avg(loan_amnt) as avg_loan ,count(loan_amnt) as NumOfApp
from loandata 
where home_ownership = "OTHER"
and
(loan_status_clean = '0' or loan_status_clean = '1' )
group by home_ownership ,loan_status_clean
"""
df1 = pysql(str1)
df1.head(25)



# ### Cleaning home_ownership

# In[61]:

loandata = loandata[loandata.home_ownership != 'OTHER']# Eliminating OTHER"  
loandata.head(2)
print(loandata.home_ownership.value_counts())


# In[62]:

# loandata['home_ownership_clean'] = loandata['home_ownership'].map({'MORTGAGE': 2, 'RENT': 3, 'OWN':1, 'OTHER':4})
# loandata = loandata[loandata.home_ownership_clean != 4]# Eliminating OTHER"  

loandata['home_ownership_clean'] = loandata['home_ownership'].map({'MORTGAGE': 1, 'OWN': 1, 'RENT':0})
loandata["home_ownership_clean"] = loandata["home_ownership_clean"].apply(lambda home_ownership_clean: 0 if home_ownership_clean == 0 else 1)


# We will assign 
#  * OWN + MORTGAGE = '1' 
#  * 'RENT' = '0'

# In[63]:

print(loandata.home_ownership_clean.value_counts())


# In[64]:

get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


# In[65]:

t1 = pd.crosstab(index=loandata["home_ownership"],columns=loandata["loan_status_clean"], margins = True)
t1.columns = ["Bad Loan","Good Loan","Rtotal"]
t1.index = ["Mortgage", "Own","Rent", "Ctotal"]
t1
# t0 = t1/t1.ix["Ctotal","Rtotal"]*100
# t0


# In[66]:

pysql = lambda q: pdsql.sqldf(q, globals())

str1= """SELECT home_ownership, loan_status_clean, count(loan_status_clean) as NumOfApp
from loandata 
group by home_ownership ,loan_status_clean
"""
df1 = pysql(str1)
df1.head(25)



# In[67]:

t2 = pd.crosstab(loandata['home_ownership'], loandata['loan_status_clean'])
t2.columns = ["Bad Loan","Good Loan"]
# t2.index = ['MORTGAGE','Mortgage/Own']
t2.plot(kind='bar', stacked=True, color=['orange','green'], grid=False)


# * 20%(32146/159786) of the loan with home_ownership = "MORTGAGE" has defaulted.
# * 27%(35360/127259)of the loan with home_ownership = "RENT" has defaulted
# * 24%(8097/32642)of the loans with home_ownership = "OWN" has defaulted.
# * "Rent" status seems to have the biggest contribution to the bad loans as compared to "Mortgage,Own" categories. 

# ### Analysing "grade"

# This is LC assigned loan grade: There are 7 loan grades ranging from A:F, A being the finest and F being the lowest grade. Lets look at distribution of loan_amnt against grades.

# In[68]:

loandata[ (loandata['grade'].isnull())]##checking if the column has null values


# In[69]:

from ggplot import *
# ggplot(loandata, aes(loan_amnt, col = grade)) + geom_histogram(bins = 50) + facet_grid(grade)
gg = ggplot(loandata, aes('loan_amnt',col='grade')) + geom_histogram(binwidth=300)+facet_grid('grade')
print (gg)

#more loans have been allotted to the loan grade A,B,C,D compared to the lower grades.


# It appears more loans have been allotted to the loan grade A,B,C,D compared to the lower grades.

# ### Cleaning grade

# In[70]:

##Loan grade
loandata['grade_clean'] = loandata['grade'].map({'A':7,'B':6,'C':5,'D':4,'E':3,'F':2,'G':1})


# In[71]:

loandata.head()


# In[72]:

t1 = pd.crosstab(index=loandata["grade_clean"],columns=loandata["loan_status_clean"], margins = True)
t1.columns = ["Bad Loan","Good Loan","Rtotal"]
t1.index = ['1','2','3','4','5','6','7',"Ctotal"]

t1


# In[73]:

t2 = pd.crosstab(loandata['grade_clean'], loandata['loan_status_clean'])
t2.columns = ["Bad Loan","Good Loan"]
t2.index = ['1','2','3','4','5','6','7']
t2.plot(kind='bar', stacked=True, color=['orange','green'], grid=False)


# Loan grade 1,2 & 3 appear to have 50:50 Good and bad loans.

# <div class="span5 alert alert-info">
# <h3>Applying Machine Learning Algorithms</h3>
# <b>
# * We will be using following techniques.
# * Logistic Regression 
# * KNN
# * Random Forest
# * Naive Bayes
# * Decision Tree

# In[74]:

import statsmodels.api as sm
from sklearn import linear_model, datasets
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.cross_validation import train_test_split


# In[75]:

loandata.info()


# We will keep only selected columns and apply modelling techniques and explore their predictive power against loan status.
# We call this df loan_2

# In[76]:

loan_2 = loandata[['emp_length','emp_length_clean','loan_status_clean','home_ownership_clean','home_ownership','grade_clean','verification_status_clean','term_clean']]


# 
# <div class="span5 alert alert-info">
# 
# <h3>Logistic regression</h3>
# <b>
# </n>
# 
# We are going to use Logistic Regression which is an often used model in problems with binary target variables. Target variable in our case is Loan_status which is indeed a binary variable. It might not be the best approach, yet it definitely offers some insights in the data.We will analyse Home_Ownership, Years of employed , Term, Grade, Emp_length and Verification_status.
# 
# * LR with Term ,Verification status and grade.
# * LR with home ownership status.
# * LR with Emp length.
# 

# In[77]:

loan_2.shape


# #### home_ownership
# We will apply logistic regression to the 'home_ownership' and predict the loan gets paid off or defaulted given the lender's home ownership status.' We will be creating dummy variables for individual statuses.
# 
# 

# In[78]:

home_ownership = pd.get_dummies(loan_2.home_ownership)
loan_ho = loan_2.join(home_ownership)
loan_ho.head(1)


# In[79]:

X_Var_0 = ['MORTGAGE','OWN','RENT']
X_0 = loan_ho[X_Var_0]


# In[80]:

X_0 = X_0.values


# In[81]:

y_0 = loan_ho['loan_status_clean'].values


# In[82]:

clf = linear_model.LogisticRegression()


# In[83]:

model_0 = clf.fit(X_0,y_0)
print('intercept:', clf.intercept_)
print('coefficient:', clf.coef_[0])


# In[84]:

model_0.score(X_0,y_0)


# In[85]:

s= pd.DataFrame(list(zip(X_Var_0,model_0.coef_.T)))
s.columns = ["Status","Coef"]
# s.index = ['Ownership Status = Mortgage','Ownership Status = Own','Ownership Status = Rent']
s


# 
# #### Inference:- 
#  Home ownership status marked as "Rent" have only 0.14 log odds of paying back the loan where as 
# "Mortgage" has 0.46 chance of paying off and "Own" = 0.25 chance of paying back loan.
# 
# 

# #### EMP_LENGTH

# In[86]:

emp_len= pd.get_dummies(loan_2.emp_length)
loan_emplen = loan_2.join(emp_len)
loan_emplen.head(1)


# In[87]:

X_Var_1 = ['< 1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years','Not specified']
X_1 = loan_emplen[X_Var_1]


# In[88]:

y_1 = loan_emplen['loan_status_clean'].values


# In[89]:

model_1 = clf.fit(X_1,y_1)


# In[90]:

model_1.score(X_1, y_1)


# In[91]:

# pd.concat([pd.DataFrame(X_Var_1),pd.DataFrame(model_1.coef_.T)],axis = 1 )
s= pd.DataFrame(list(zip(X_Var_1,model_1.coef_.T)))
s.columns = ["Emp_length","Coef"]
# s.index = ['1','2','3','4','5','6','7','8','9','10']
s


# Inference:-
# 
# There does not seem to be striking variation in the coefficient values for the number of employment years between 2 to 9 years. They are hovering over the range of .01 to .05. However, applications with emp_length <= 1 year  and also those which dint have the employment_length specified have -ve coefficients. THey are more likely to be defaulting. To my understanding, anybody not specifying the employment could be because they might not be bound with employment at the time loan was funded or could be in some other business. Their annual inc and verification status could be analysed further to draw any co-relation yet this feature does have predictive power to a certain extent as much intuitive it appears(a person would be continuing to pay off as lons as he/she is employed)
# 

# #### Verification Status and Term

# In[92]:

X_Var_3 = ['verification_status_clean','term_clean', 'grade_clean']
X_3 = loan_2[X_Var_3]


# In[93]:

X_3 = X_3.values


# In[94]:

y_3 = loan_2['loan_status_clean'].values


# In[95]:

model_3 = clf.fit(X_3,y_3)


# In[96]:

model_3.score(X_3, y_3)


# In[97]:

# pd.concat([pd.DataFrame(X_Var_3),pd.DataFrame(model_3.coef_.T)],axis = 1 )
s= pd.DataFrame(list(zip(X_Var_3,model_3.coef_.T)))
s.columns = ["Status","Coef"]
# s.index = ['Verified Status']
s


# 
# <div class="span5 alert alert-info">
# 
# <h3>Inference</h3>
# <b>
# 
# * Verified Status indicates if income was verified by LC. There is a good chance of loan being defaulted if "Status" is not verified where as 1 unit increase in the term from 36 month to 60 months increases the log odds of loan being paid off by .43
# Similary, for every additional increase in the grade "G" to "F" or in our case "1" to "2" the log odds of the loan being paid off increases by .432 which makes quite intuitive sense. LC assigns a high grade to a loan that they think is stable and low grade to a loan which is faulty. Conclusively as the grade for a loan increases  the chance of the loan being paid off also increases by a coeff of .43.
# 

# <div class="span5 alert alert-info">
# <h3>Split, Train and Test</h3>
# <b>

#  Here we split the dataset into train:test in the ratio 7:3. And we will be using above variables to predict bad loans following below machine learning techniques.
# (1)    Logistic regression
# (2)    Random forest
# (3)    k-Nearest neighbor (k=13)
# (4)    Decision Tree

# In[98]:

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd


# In[99]:

features = ['verification_status_clean'] +['home_ownership_clean'] +['emp_length_clean']+['grade_clean']+['term_clean']
target = 'loan_status_clean'

# Now create an X variable (containing the features) and an y variable (containing only the target variable)
X = loan_2[features]
y = loan_2[target]


# In[100]:

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)


# #### Logistic Regression

# First, we try a basic Logistic Regression:
# * Split the data into a training and test (hold-out) set
# * Train on the training set, and test for accuracy on the testing set

# In[101]:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split the data into a training and test set.
X_train, X_test, y_train, y_test = train_test_split(X.values,y.values,test_size=0.33,random_state=5)

clf = LogisticRegression()
# Fit the model on the trainng data.
clf.fit(X_train, y_train)

print('intercept:', clf.intercept_)
print('coefficient:', clf.coef_[0])

# s.index = ['Ownership Status = Mortgage','Ownership Status = Own','Ownership Status = Rent']
print(s)

# Print the accuracy from the testing data.
print(accuracy_score(clf.predict(X_test), y_test))
print(classification_report(y_test, clf.predict(X_test)))


# The 0 classes (Defaulted/Charged Off loans) are predicted with .52 precision and .13 recall. Let's tune the model and try to generate the best value of C via grid search in the below steps:-

# #### Tuning the model

# The model has some hyperparameters we can tune for hopefully better performance. In Logistic Regression, the most important parameter to tune is the regularization parameter C. For tuning the parameters, we will use a mix of cross-validation and grid search. 

# In[102]:

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def cv_score(clf, x, y, score_func=accuracy_score):
    result = 0
    nfold = 5
    for train, test in KFold(nfold).split(x): # split data into train/test groups, 5 times
        clf.fit(x[train], y[train]) # fit
        result += score_func(clf.predict(x[test]), y[test]) # evaluate score function on held-out data
    return result / nfold # average


# In[103]:

clf = LogisticRegression()
score = cv_score(clf, X_test, y_test)
print(score)


# In[104]:

#the grid of parameters to search over
Cs = [0.001, 0.1, 1, 10, 100]
for c in Cs:
    score = []
    clf = LogisticRegression(C=c)
    c_score = cv_score(clf, X_train, y_train)
    score.append(c_score)
    print("for C = " + str(c) + " | Avg_score = " + str(c_score))
print ("Highest avg sore is: "+ str(max(score)))


# We get the lowest average score at C = 1.

# In[105]:

# when C= 1
clf = LogisticRegression(C=1)
clf.fit(X_train,y_train)
accuracy= accuracy_score(clf.predict(X_test), y_test)
print(accuracy)


# In[106]:

# Define the model
lr = LogisticRegression()

# Define the splitter for splitting the data in a train set and a test set
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=0)

# Loop through the splits (only one)
for train_indices, test_indices in splitter.split(X, y):
    # Select the train and test data
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    
# Fit the model
    lr_result= lr.fit(X_train, y_train)
    lr_pred = lr_result.predict(X_test)

# And finally show the results
print(classification_report(y_test, lr_pred))
print(accuracy_score((y_test), lr_pred))   


# #### Random Forest

# In[107]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
rf = RandomForestClassifier(n_estimators=10, min_samples_split=2)

rf_result=rf.fit(X_train,y_train)

rf_pred = rf_result.predict(X_test)
accuracy = accuracy_score(y_test, rf_pred)
accuracy


# #### KNN

# In[108]:

# ##K-nearest neighbor: let us try a range of k to see what might be the best k
# k_range=range(1,21)
# scores=[]
# for k in k_range:

#    knn = KNeighborsClassifier(n_neighbors=k)

#    knn_result=knn.fit(X_train,y_train)

#    knn_pred = knn.predict(X_test)

# #    scores.append(metrics.accuracy_score(y_test, knn_pred))    


# In[109]:

# plt.plot(k_range, scores)
# plt.xlabel('Value of k for KNN')
# plt.ylabel('Performance')
# plt.title('The effect of "k" in k-nearest neighbor')


# In[110]:

knn = KNeighborsClassifier(n_neighbors=21)
knn_result=knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, knn_pred)
accuracy


# #### Decision Tree

# In[111]:

from sklearn import tree


# In[112]:

dt = tree.DecisionTreeClassifier()
dt = clf.fit(X,y)
dt_result = dt.fit(X_train,y_train)
dt_pred = dt_result.predict_proba(X_test)


# In[113]:

dt.score(X_train,y_train)


# In[114]:

dt.score(X_test,y_test)


# #### Naive Bayes

# In[115]:

from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split


# In[116]:

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
nb = GaussianNB()
nb_result = nb.fit(X_train,y_train)
nb_pred = nb_result.predict(X_test)


# In[117]:


nb.score(X_train,y_train)


# In[118]:

nb.score(X_test,y_test)


# In[119]:

from sklearn import metrics
def measure_performance(X,y,nb,show_accuracy=True, show_classification_report=True, show_confusion_matrix=True):
    y_pred= dt.predict(X)   
    if show_accuracy:
        print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y_test,nb_pred)),"\n")

    if show_classification_report:
        print ("Classification report")
        print (metrics.classification_report(y_test,nb_pred),"\n")
        
    if show_confusion_matrix:
        print ("Confusion matrix")
        print (metrics.confusion_matrix(y_test,nb_pred),"\n")
        
measure_performance(X_train,y_train,dt,show_accuracy=True,show_classification_report=True,show_confusion_matrix=True)


# <div class="span5 alert alert-info">
# <h3>Model comparision with ROC</h3>
# <b>
# * AUC for each model and their performance when we set probability cutoff at 50% is summarised below

# In[120]:

# calculate the fpr and tpr for all thresholds of the classification
probs_lr = clf.predict_proba(X_test)
preds_lr = probs_lr[:,1]

probs_dt = dt.predict_proba(X_test)
preds_dt = probs_dt[:,1]

probs_nb = nb.predict_proba(X_test)
preds_nb = probs_nb[:,1]

probs_rf = rf.predict_proba(X_test)
preds_rf = probs_rf[:,1]

probs_knn = knn.predict_proba(X_test)
preds_knn = probs_knn[:,1]

fpr1, tpr1, threshold1 = metrics.roc_curve(y_test, preds_lr)
roc_auc_lr = metrics.auc(fpr1, tpr1)

fpr2, tpr2, threshold2 = metrics.roc_curve(y_test, preds_dt)
roc_auc_dt = metrics.auc(fpr2, tpr2)

fpr3, tpr3, threshold3 = metrics.roc_curve(y_test, preds_nb)
roc_auc_nb = metrics.auc(fpr3, tpr3)

fpr4, tpr4, threshold4 = metrics.roc_curve(y_test, preds_rf)
roc_auc_rf = metrics.auc(fpr4, tpr4)

fpr5, tpr5, threshold5 = metrics.roc_curve(y_test, preds_knn)
roc_auc_knn = metrics.auc(fpr5, tpr5)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr1,tpr1,label='Logistic Regression(AUC = %0.2f)' % roc_auc_lr)
# plt.plot(fpr2,tpr2,label='Decision Tree(AUC = %0.2f)' % roc_auc_dt)
plt.plot(fpr3,tpr3,label='Naive Bayes(AUC = %0.2f)' % roc_auc_nb)
plt.plot(fpr4,tpr4,label='Random Forest(AUC = %0.2f)' % roc_auc_rf)
plt.plot(fpr5,tpr5,label='kNN, n=21(AUC = %0.2f)' % roc_auc_knn)
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.plot([0, 1], [0, 1],'k--')
plt.xlabel('Specificity')
plt.ylabel('Sensitivity')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# <div class="span5 alert alert-info">
# <h3>Conclusion</h3>
# 
# <b>Although the overall prediction accuracy is good , yet the prediction accuracy for defaulter instances are not as good as expected. Out of all the classification algorithms used on the Lending Club dataset for the year 2014 -2015, Logistic regression and Random forest stand at .70 as the best overall accuracy. kNN  and Naive Bayes scores are weak as compared to other algorithms. 
# 
# Our input variables Term, Emp_length, home_ownership_status, verification_status seem to have a decent predictive strength against loan_stats(default/paid off).
