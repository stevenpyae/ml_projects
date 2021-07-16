# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 13:35:43 2021
50000 samples
Task: Identify which users will likely NOT enroll in the paid product, so that additional offers can be given to them.
Data: app usage data-> user's first day in the app. Limitation: users can enjoy 24-hour free trial of premium features. 

@author: Corbi
"""

#Importing the data
# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sn # Statistical data visualization
from dateutil import parser #to parse our datetime fields

dataset = pd.read_csv('appdata10.csv') 

### EDS ####
# dataset.head()
# dataset.describe() 
# Hour column cannot be found. cuz its String, First, convert it into string

## Data Cleaning
dataset['hour'] = dataset.hour.str.slice(1,3).astype(int)

### Plotting
dataset2 = dataset.copy().drop(columns = ['user', 'screen_list', 'enrolled_date', 'first_open','enrolled'])

### Histograms
plt.suptitle('Historgram of Numerical Columns', fontsize =20)

for i in range(1, dataset2.shape[1]+1): #dataset2.shape[1] is the number of columns, python doesnt include the last column, you +1 
    plt.subplot(3,3,i) #number of dimentions we want the image to be, 3by 3 plot, 
    f = plt.gca() # GCA to clean up everything.
    f.set_title(dataset2.columns.values[i-1]) #dataset2.columns.value[1-1]
    
    # how many bins its gonna use for  
    vals = np.size(dataset2.iloc[:, i-1].unique()) #dataset2.iloc[:, i-1] this is querying the entire column 
    plt.hist(dataset2.iloc[:, i -1], bins = vals, color = '#3F5D7D')
## Correlation with Response
## good for us to how each independent feature affects the response variable. To build a proper model 
dataset2.corrwith(dataset.enrolled).plot.bar(figsize = (20,10), 
                                            title = 'Correlation with Response variable', 
                                            fontsize = 15, rot = 45, grid= True)

#above means positively correlated, below means negatively correlated. 
# Correlation Matrix
# Plot that gives you correlation between each field relation. which field is linearly depending on each other.
# it will help in building the model because we don't want any field to be dependent,
# Assumption is that the features are independent 

sn.set(style="white", font_scale=2)

# Compute the correlation matrix
corr = dataset2.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Day of week, is correlated to all the feathres, if its 0.0 means zero to no correlation. Independent. 
# Age is bluish means its negatively correlated. 
# Gray - No correlation, Pink- Positively correlated, Bluish Green - Negatively correlated.

## Feature Engineering ###
#Fine tuning the response variable
# to have a particular date reange limit. Plot distribution between difference between the first open and enrollment date
# dataset.dtypes function
dataset['first_open']=[parser.parse(row_data) for row_data in dataset['first_open']] # convert the first_open date into date-time format
dataset["enrolled_date"] = [parser.parse(row_date) if isinstance(row_date, str) else row_date for row_date in dataset["enrolled_date"]] # convert the first_open date into date-time format
#only apply the formulat to things that has values inside. because enrolled_date can be None.

dataset['difference'] = (dataset.enrolled_date - dataset.first_open).astype('timedelta64[h]') #convert it to hours

plt.hist(dataset['difference'].dropna(), color = '#3F5D7D')
plt.title("Distribution of Time-Since-Enrolled")
plt.show()

plt.hist(dataset['difference'].dropna(), color = '#3F5D7D', range =[0,100])
plt.title("Distribution of Time-Since-Enrolled")
plt.show()

dataset.loc[dataset.difference> 48,'enrolled'] = 0 #every person who ever enroll is 0, If they enrolled after 2 days, we categorize them as not enrolled

dataset = dataset.drop(columns = ['difference', 'enrolled_date', 'first_open'])


## Feature Engineering - Screens List, comma separated, convert it to the data that the software can read. 
top_screens = pd.read_csv('top_screens.csv').top_screens.values
#list of popular screens, columns for the popular screens, the rest, create another column for account for this

dataset['screen_list'] = dataset.screen_list.astype(str) + ',' 

for sc in top_screens:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int) # boolean true or false.
    dataset['screen_list'] = dataset.screen_list.str.replace(sc+',', '')
    
#Final column called other
dataset["Other"] = dataset.screen_list.str.count(',') #find how many left
dataset = dataset.drop(columns = ['screen_list'])

# Funnels - group of screens that belong to the same set
# Group the similar screens into one funnel, they become one column
savings_screens = ["Saving1",
                    "Saving2",
                    "Saving2Amount",
                    "Saving4",
                    "Saving5",
                    "Saving6",
                    "Saving7",
                    "Saving8",
                    "Saving9",
                    "Saving10"]
dataset["SavingCount"] = dataset[savings_screens].sum(axis=1) #counts all the columns that has the names, and count them 
dataset = dataset.drop(columns=savings_screens)

cm_screens = ["Credit1",
               "Credit2",
               "Credit3",
               "Credit3Container",
               "Credit3Dashboard"]
dataset["CMCount"] = dataset[cm_screens].sum(axis=1)
dataset = dataset.drop(columns=cm_screens)

cc_screens = ["CC1",
                "CC1Category",
                "CC3"]
dataset["CCCount"] = dataset[cc_screens].sum(axis=1)
dataset = dataset.drop(columns=cc_screens)

loan_screens = ["Loan",
               "Loan2",
               "Loan3",
               "Loan4"]
dataset["LoansCount"] = dataset[loan_screens].sum(axis=1)
dataset = dataset.drop(columns=loan_screens)
# We group them so that each individual column is unique and not correlated with the other.

dataset.to_csv('new_appdata10.csv', index = False)
