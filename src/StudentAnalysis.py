#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tabulate 
from tabulate import tabulate


# In[2]:


df=pd.read_csv("FinalDatasetStudent.csv")
print(df.head())


# # ABOUT DATASET

# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# # Cleaning

# In[6]:


#Drop columns not required.


# In[7]:


df = df.drop("Unnamed: 4", axis=1)
print(df.head())


# In[8]:


#wklystudyHours 10-may is actually 10-5 hours
#now changing it
df["WklyStudyHours"]=df["WklyStudyHours"].str.replace("10-May","5-10")
df.head()


# In[9]:


#Remove null values from columns
df=df.dropna()
print(df.head())


# In[10]:


df.isnull().sum()


# In[11]:


unique_count = df['EthnicGroup'].nunique()


# In[12]:


for column in df.columns:
    unique_count = df[column].nunique()
    print(f"Number of unique values in '{column}': {unique_count}")


# # Visualization
# 

# ### Gender

# In[13]:


#Gender distribution
#Analysis: Female > male
plt.figure(figsize=(5,5))
ax=sns.countplot(data=df,x="Gender")
ax.bar_label(ax.containers[0])
plt.show()


# ### StudentScores Analysis: MathScore
# 

# In[14]:


#Max MathScore
selected_columns = ['ID','Gender','ParentEduc','TestPrep','ParentMarrStatus','PracticeSport','WklyStudyHours','MathScore','MathGrade']
max_math_score_row = df[df['MathScore'] == df['MathScore'].max()][selected_columns]
colalign = ["center"] * len(max_math_score_row.columns)
print(tabulate(max_math_score_row, headers='keys', tablefmt='pretty', showindex=False, colalign=colalign))


# In[15]:


#Min MathScore
selected_columns = ['ID','Gender','ParentEduc','TestPrep','ParentMarrStatus','PracticeSport','WklyStudyHours','MathScore','MathGrade']
min_math_score_row = df[df['MathScore'] == df['MathScore'].min()][selected_columns]
colalign = ["center"] * len(min_math_score_row.columns)
print(tabulate(min_math_score_row, headers='keys', tablefmt='pretty', showindex=False, colalign=colalign))


# In[16]:


# Calculate the mean MathScore
mean_math_score = df['MathScore'].mean()

mean_math_score_row = pd.DataFrame({'MeanMathScore': [mean_math_score]})
print(tabulate(mean_math_score_row, headers='keys', tablefmt='pretty', showindex=False))


# In[17]:


mean_math_score = df['MathScore'].mean()
threshold = 5
selected_rows = df[(df['MathScore'] >= mean_math_score - threshold) & (df['MathScore'] <= mean_math_score + threshold)]

selected_columns = ['ID', 'Gender', 'EthnicGroup', 'ParentEduc', 'TestPrep', 'ParentMarrStatus', 'PracticeSport', 'WklyStudyHours', 'MathScore', 'MathGrade']
print(tabulate(selected_rows[selected_columns], headers='keys', tablefmt='pretty', showindex=False))


# In[18]:


#Student Failed in Maths
failing_students = df[df['MathScore'] < 55]
total_failing_students = len(failing_students)

print("Total number of failing students:", total_failing_students)


# In[19]:


#Student passed in Maths
passing_students = df[df['MathScore'] >=55]
total_passing_students = len(passing_students)

print("Total number of passed students:", total_passing_students)


# In[20]:


#box plot for MathScore
plt.boxplot(df['MathScore'], vert=False, showmeans=True, meanline=True, labels=['MathScore'])
plt.title('Box Plot of MathScores')
plt.xlabel('MathScore')
plt.show()


# In[21]:


# Data for the pie chart
labels = ['Pass', 'Fail']
sizes = [total_passing_students, total_failing_students]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
plt.title('Distribution of Math Pass/Fail')
plt.show()


# ### StudentScore Analysis:ReadingScores

# In[22]:


#Max ReadingScore
selected_columns = ['ID','Gender','ParentEduc','TestPrep','ParentMarrStatus','PracticeSport','WklyStudyHours','ReadingScore']
max_Reading_score_row = df[df['ReadingScore'] == df['ReadingScore'].max()][selected_columns]
colalign = ["center"] * len(max_Reading_score_row.columns)
print(tabulate(max_Reading_score_row, headers='keys', tablefmt='pretty', showindex=False, colalign=colalign))


# In[23]:


#Min ReadingScore
selected_columns = ['ID','Gender','ParentEduc','TestPrep','ParentMarrStatus','PracticeSport','WklyStudyHours','ReadingScore','ReadingGrade']
min_Reading_score_row = df[df['ReadingScore'] == df['ReadingScore'].min()][selected_columns]
colalign = ["center"] * len(min_Reading_score_row.columns)
print(tabulate(min_Reading_score_row, headers='keys', tablefmt='pretty', showindex=False, colalign=colalign))


# In[24]:


# Calculate the mean ReadingScore
mean_reading_score = df['ReadingScore'].mean()
mean_reading_score_row = pd.DataFrame({'MeanReadingScore': [mean_reading_score]})
print(tabulate(mean_reading_score_row, headers='keys', tablefmt='pretty', showindex=False))


# In[25]:


mean_reading_score = df['ReadingScore'].mean()
df['ReadingScoreDifference'] = abs(df['ReadingScore'] - mean_reading_score)

threshold = 5 
average_reading_score_students = df[df['ReadingScoreDifference'] <= threshold]
selected_columns = ['ID', 'Gender', 'ParentEduc', 'TestPrep', 'ParentMarrStatus', 'PracticeSport', 'WklyStudyHours', 'ReadingScore', 'ReadingGrade']

print(tabulate(average_reading_score_students[selected_columns], headers='keys', tablefmt='pretty', showindex=False))


# In[26]:


#Number of student pass
passed_reading_count = df[df['ReadingGrade'] != 'F']['ID'].count()
print(f"Number of students who passed in Reading: {passed_reading_count}")


# In[27]:


#Number of student fail
fail_reading_count = df[df['ReadingGrade'] =='F']['ID'].count()
print(f"Number of students who failed in Reading: {fail_reading_count}")


# In[28]:


# Create a box plot for ReadingScore
plt.boxplot(df['ReadingScore'], vert=False, showmeans=True, meanline=True, labels=['ReadingScore'])
plt.title('Box Plot of ReadingScores')
plt.xlabel('ReadingScore')
plt.show()


# In[ ]:





# In[29]:


# Data for the pie chart
labels = ['Pass', 'Fail']
sizes = [passed_reading_count, fail_reading_count]

# Create a pie chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
plt.title('Distribution of Reading Pass/Fail')
plt.show()


# ### StudentScore Analysis:WritingScores

# In[30]:


#Max WritingScore
selected_columns = ['ID','Gender','ParentEduc','TestPrep','ParentMarrStatus','PracticeSport','WklyStudyHours','ReadingScore']
max_Writing_score_row = df[df['WritingScore'] == df['WritingScore'].max()][selected_columns]
colalign = ["center"] * len(max_Writing_score_row.columns)
print(tabulate(max_Writing_score_row, headers='keys', tablefmt='pretty', showindex=False, colalign=colalign))


# In[31]:


#Min WritingScore
selected_columns = ['ID','Gender','ParentEduc','TestPrep','ParentMarrStatus','PracticeSport','WklyStudyHours','ReadingScore','ReadingGrade']
min_Writing_score_row = df[df['WritingScore'] == df['WritingScore'].min()][selected_columns]
colalign = ["center"] * len(min_Writing_score_row.columns)
print(tabulate(min_Writing_score_row, headers='keys', tablefmt='pretty', showindex=False, colalign=colalign))


# In[32]:


# Calculate the mean WritingScore
mean_writing_score = df['WritingScore'].mean()
mean_writing_score_row = pd.DataFrame({'MeanWritingScore': [mean_writing_score]})
print(tabulate(mean_writing_score_row, headers='keys', tablefmt='pretty', showindex=False))


# In[33]:


mean_writing_score = df['WritingScore'].mean()
df['WritingScoreDifference'] = abs(df['WritingScore'] - mean_writing_score)

threshold = 5 
average_writing_score_students = df[df['WritingScoreDifference'] <= threshold]
selected_columns = ['ID', 'Gender', 'ParentEduc', 'TestPrep', 'ParentMarrStatus', 'PracticeSport', 'WklyStudyHours', 'WritingScore', 'WritingGrade']

print(tabulate(average_writing_score_students[selected_columns], headers='keys', tablefmt='pretty', showindex=False))


# In[34]:


#Number of student pass
passed_writing_count = df[df['WritingGrade'] != 'F']['ID'].count()
print(f"Number of students who passed in Writing: {passed_writing_count}")


# In[35]:


#Number of student failed
fail_writing_count = df[df['WritingGrade']== 'F']['ID'].count()
print(f"Number of students who failed in Writing: {fail_writing_count}")


# In[36]:


#box plot for WritingScore
plt.boxplot(df['WritingScore'], vert=False, showmeans=True, meanline=True, labels=['WritingScore'])
plt.title('Box Plot of WritingScores')
plt.xlabel('WritingScore')
plt.show()


# In[ ]:





# In[37]:


# Data for the pie chart
labels = ['Pass', 'Fail']
sizes = [passed_writing_count, fail_writing_count]

# Create a pie chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
plt.title('Distribution of Writing Pass/Fail')
plt.show()


# # OverAll Visualization

# ### w.r.t ParentEduc

# #####  Results: since difference is 11 btw mean high and low so it can be deduce that parent educ effects marks.

# In[38]:


gb=df.groupby("ParentEduc").agg({"MathScore":'mean',"ReadingScore":'mean',"WritingScore":'mean'})
print(gb)


# In[39]:


sns.heatmap(gb,annot=True)
plt.title("Relationship Between ParentEduc and Student's Scores")


# In[40]:


failed_students = df[df['MathGrade'] == 'F']
failed_by_parent_educ = failed_students['ParentEduc'].value_counts().reset_index()

failed_by_parent_educ.columns = ['ParentEduc', 'Number of Failed Students']
print(failed_by_parent_educ)
print("\n")

passed_students = df[df['MathGrade'] != 'F']
passed_by_parent_educ = passed_students['ParentEduc'].value_counts().reset_index()

passed_by_parent_educ.columns = ['ParentEduc', 'Number of Passed Students']
print(passed_by_parent_educ)


# In[41]:


failed_students = df[df['ReadingGrade'] == 'F']

failed_by_parent_educ = failed_students['ParentEduc'].value_counts().reset_index()

failed_by_parent_educ.columns = ['ParentEduc', 'Number of Failed Students']
print(failed_by_parent_educ)

print("\n")


passed_students = df[df['ReadingGrade'] != 'F']
passed_by_parent_educ = passed_students['ParentEduc'].value_counts().reset_index()

passed_by_parent_educ.columns = ['ParentEduc', 'Number of Passed Students']
print(passed_by_parent_educ)


# In[42]:


failed_students = df[df['WritingGrade'] == 'F']

failed_by_parent_educ = failed_students['ParentEduc'].value_counts().reset_index()

failed_by_parent_educ.columns = ['ParentEduc', 'Number of Failed Students']
print(failed_by_parent_educ)

print("\n")

passed_students = df[df['WritingGrade'] != 'F']
passed_by_parent_educ = passed_students['ParentEduc'].value_counts().reset_index()

passed_by_parent_educ.columns = ['ParentEduc', 'Number of Passed Students']
print(passed_by_parent_educ)


# ### w.r.t ParentMaritalStatus

# ##### Results: Negligible impact on Students Scores

# In[43]:


gb1=df.groupby("ParentMarrStatus").agg({"MathScore":'mean',"ReadingScore":'mean',"WritingScore":'mean'})
print(gb1)


# In[44]:


sns.heatmap(gb1,annot=True)
plt.title("Relationship Between Parent Marital Status and Student's Scores")


# In[45]:


failed_math_students = df[df['MathGrade'] == 'F']
failed_reading_students = df[df['ReadingGrade'] == 'F']
failed_writing_students = df[df['WritingGrade'] == 'F']

failed_math_by_parent_status = failed_math_students['ParentMarrStatus'].value_counts().reset_index()
failed_reading_by_parent_status = failed_reading_students['ParentMarrStatus'].value_counts().reset_index()
failed_writing_by_parent_status = failed_writing_students['ParentMarrStatus'].value_counts().reset_index()

print("Number of Failed Math Students by Parent Marital Status:")
print(failed_math_by_parent_status)

print("\nNumber of Failed Reading Students by Parent Marital Status:")
print(failed_reading_by_parent_status)

print("\nNumber of Failed Writing Students by Parent Marital Status:")
print(failed_writing_by_parent_status)


# In[46]:


passing_math_students = df[df['MathGrade'] != 'F']
passing_reading_students = df[df['ReadingGrade'] != 'F']
passing_writing_students = df[df['WritingGrade'] != 'F']

passing_math_by_parent_status = passing_math_students['ParentMarrStatus'].value_counts().reset_index()
passing_reading_by_parent_status = passing_reading_students['ParentMarrStatus'].value_counts().reset_index()
passing_writing_by_parent_status = passing_writing_students['ParentMarrStatus'].value_counts().reset_index()

print("Number of Passing Math Students by Parent Marital Status:")
print(passing_math_by_parent_status)

print("\nNumber of Passing Reading Students by Parent Marital Status:")
print(passing_reading_by_parent_status)

print("\nNumber of Passing Writing Students by Parent Marital Status:")
print(passing_writing_by_parent_status)


# ### w.r.t TestPrep
# 
# #### Results: The difference in result is of 10 hence , TestPrep has major effects on Student Scores

# In[47]:


gb3=df.groupby("TestPrep").agg({"MathScore":'mean',"ReadingScore":'mean',"WritingScore":'mean'})
print(gb3)


# In[48]:


sns.heatmap(gb3,annot=True)
plt.title("Relationship Between TestPrep Status and Student's Scores")


# In[49]:


# students who have 'none' in TestPrep and failed
no_test_prep_students = df[df['TestPrep'] == 'none']

failed_math_no_test_prep = no_test_prep_students[no_test_prep_students['MathGrade'] == 'F']

failed_reading_no_test_prep = no_test_prep_students[no_test_prep_students['ReadingGrade'] == 'F']

failed_writing_no_test_prep = no_test_prep_students[no_test_prep_students['WritingGrade'] == 'F']

failed_all_courses_no_test_prep = set(failed_math_no_test_prep['ID']) & set(failed_reading_no_test_prep['ID']) & set(failed_writing_no_test_prep['ID'])

print("Total number of students with no test preparation:", len(no_test_prep_students))
print("Total number of students who failed in Math among those with no test preparation:", len(failed_math_no_test_prep))
print("Total number of students who failed in Reading among those with no test preparation:", len(failed_reading_no_test_prep))
print("Total number of students who failed in Writing among those with no test preparation:", len(failed_writing_no_test_prep))


# In[50]:


# students who have 'none' in TestPrep and passed
none_test_prep_students = df[df['TestPrep'] == 'none']

passed_math_none_test_prep = none_test_prep_students[none_test_prep_students['MathGrade'] != 'F']

passed_reading_none_test_prep = none_test_prep_students[none_test_prep_students['ReadingGrade'] != 'F']

passed_writing_none_test_prep = none_test_prep_students[none_test_prep_students['WritingGrade'] != 'F']

passed_all_courses_none_test_prep = set(passed_math_none_test_prep['ID']) & set(passed_reading_none_test_prep['ID']) & set(passed_writing_none_test_prep['ID'])

print("Total number of students with 'none' in TestPrep:", len(none_test_prep_students))
print("Total number of students who passed in Math among those with 'none' in TestPrep:", len(passed_math_none_test_prep))
print("Total number of students who passed in Reading among those with 'none' in TestPrep:", len(passed_reading_none_test_prep))
print("Total number of students who passed in Writing among those with 'none' in TestPrep:", len(passed_writing_none_test_prep))
print("Total number of students who passed in every individual course among those with 'none' in TestPrep:", len(passed_all_courses_none_test_prep))


# In[51]:


none_test_prep_passed = len(passed_all_courses_none_test_prep)
none_test_prep_failed = len(none_test_prep_students) - none_test_prep_passed

labels = ['Passed', 'Failed']
sizes = [none_test_prep_passed, none_test_prep_failed]
colors = ['#66b3ff', '#99ff99']  # Blue for Passed, Green for Failed

plt.figure(figsize=(4, 4))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Distribution of students with "none" in TestPrep (Passed vs Failed)')
plt.show()


# In[52]:


# students who have 'completed' in TestPrep and failed
completed_test_prep_students = df[df['TestPrep'] == 'completed']

failed_math_completed_test_prep = completed_test_prep_students[completed_test_prep_students['MathGrade'] == 'F']

failed_reading_completed_test_prep = completed_test_prep_students[completed_test_prep_students['ReadingGrade'] == 'F']

failed_writing_completed_test_prep = completed_test_prep_students[completed_test_prep_students['WritingGrade'] == 'F']

failed_all_courses_completed_test_prep = set(failed_math_completed_test_prep['ID']) & set(failed_reading_completed_test_prep['ID']) & set(failed_writing_completed_test_prep['ID'])

print("Total number of students with completed test preparation:", len(completed_test_prep_students))
print("Total number of students who failed in Math among those with completed test preparation:", len(failed_math_completed_test_prep))
print("Total number of students who failed in Reading among those with completed test preparation:", len(failed_reading_completed_test_prep))
print("Total number of students who failed in Writing among those with completed test preparation:", len(failed_writing_completed_test_prep))
print("Total number of students who failed in every individual course among those with completed test preparation:", len(failed_all_courses_completed_test_prep))


# In[53]:


# students who have 'completed' in TestPrep and passed
completed_test_prep_students = df[df['TestPrep'] == 'completed']

passed_math_completed_test_prep = completed_test_prep_students[completed_test_prep_students['MathGrade'] != 'F']

passed_reading_completed_test_prep = completed_test_prep_students[completed_test_prep_students['ReadingGrade'] != 'F']

passed_writing_completed_test_prep = completed_test_prep_students[completed_test_prep_students['WritingGrade'] != 'F']

passed_all_courses_completed_test_prep = set(passed_math_completed_test_prep['ID']) & set(passed_reading_completed_test_prep['ID']) & set(passed_writing_completed_test_prep['ID'])

print("Total number of students with completed test preparation:", len(completed_test_prep_students))
print("Total number of students who passed in Math among those with completed test preparation:", len(passed_math_completed_test_prep))
print("Total number of students who passed in Reading among those with completed test preparation:", len(passed_reading_completed_test_prep))
print("Total number of students who passed in Writing among those with completed test preparation:", len(passed_writing_completed_test_prep))
print("Total number of students who passed in every individual course among those with completed test preparation:", len(passed_all_courses_completed_test_prep))


# In[54]:


completed_test_prep_passed = len(passed_all_courses_completed_test_prep)
completed_test_prep_failed = len(completed_test_prep_students) - completed_test_prep_passed

labels = ['Passed', 'Failed']
sizes = [completed_test_prep_passed, completed_test_prep_failed]
colors = ['#66b3ff', '#99ff99']  # Blue for Passed, Green for Failed

plt.figure(figsize=(4,4))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Distribution of students with "completed" in TestPrep (Passed vs Failed)')
plt.show()


# In[55]:


# None Highest and Lowest marks

#rows where TestPrep is 'none'
none_test_prep_students = df[df['TestPrep'] == 'none']

lowest_marks_row = none_test_prep_students.loc[none_test_prep_students[['MathScore', 'ReadingScore', 'WritingScore']].idxmin()]
lowest_marks_row = lowest_marks_row[['ID', 'MathScore', 'ReadingScore', 'WritingScore']]

highest_marks_row = none_test_prep_students.loc[none_test_prep_students[['MathScore', 'ReadingScore', 'WritingScore']].idxmax()]
highest_marks_row = highest_marks_row[['ID', 'MathScore', 'ReadingScore', 'WritingScore']]

print("Student with TestPrep as 'none' and lowest marks:")
print(lowest_marks_row)

print("\nStudent with TestPrep as 'none' and highest marks:")
print(highest_marks_row)


# In[56]:


#Completed Highest and Lowest

#rows where TestPrep is 'completed'
completed_test_prep_students = df[df['TestPrep'] == 'completed']

lowest_marks_row_completed = completed_test_prep_students.loc[completed_test_prep_students[['MathScore', 'ReadingScore', 'WritingScore']].idxmin()]
lowest_marks_row_completed = lowest_marks_row_completed[['ID', 'MathScore', 'ReadingScore', 'WritingScore']]

highest_marks_row_completed = completed_test_prep_students.loc[completed_test_prep_students[['MathScore', 'ReadingScore', 'WritingScore']].idxmax()]
highest_marks_row_completed = highest_marks_row_completed[['ID', 'MathScore', 'ReadingScore', 'WritingScore']]

print("Student with TestPrep as 'completed' and lowest marks:")
print(lowest_marks_row_completed)

print("\nStudent with TestPrep as 'completed' and highest marks:")
print(highest_marks_row_completed)


# ### w.r.t WeeklyHours
# 
# #### Result: Difference is not much, very minor effect on StudentScores can be seen.

# In[57]:


study_hours_counts = df['WklyStudyHours'].value_counts()

labels = study_hours_counts.index
sizes = study_hours_counts.values
colors = ['#ff9999', '#66b3ff', '#99ff99']
explode = (0.1, 0, 0)  # explode the 1st slice

plt.figure(figsize=(4, 4))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode, startangle=90)
plt.title('Distribution of Weekly Study Hours')
plt.show()


# In[58]:


gb2=df.groupby("WklyStudyHours").agg({"MathScore":'mean',"ReadingScore":'mean',"WritingScore":'mean'})
print(gb2)


# In[59]:


sns.heatmap(gb2,annot=True)
plt.title("Relationship Between WklyStudyHours Status and Student's Scores")


# 
# ### w.r.t Practice Sports
# 
# #### Results: The difeerence is scores is of 1 or 2 it is showing some effects on scores , however, scores are not really affected by it.

# In[60]:


gb4=df.groupby("PracticeSport").agg({"MathScore":'mean',"ReadingScore":'mean',"WritingScore":'mean'})
print(gb4)


# In[61]:


sns.heatmap(gb4,annot=True)
plt.title("Relationship Between PracticeSport Status and Student's Scores")


# In[62]:


PracticeSportcounts = df['PracticeSport'].value_counts()

labels = PracticeSportcounts.index
sizes = PracticeSportcounts.values
colors = ['#ff9999', '#66b3ff', '#99ff99']
explode = (0.1, 0, 0)  # explode the 1st slice

plt.figure(figsize=(4, 4))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode, startangle=90)
plt.title('Distribution of PracticeSport')
plt.show()


# # Machine Learning Algorithm:

# In[63]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# In[64]:


df=pd.read_csv("FinalDatasetStudent.csv")
df.head()


# In[65]:


df = df.drop("Unnamed: 4", axis=1)
print(df.head())


# In[66]:


#wklystudyHours 10-may is actually 10-5 hours
#now changing it
df["WklyStudyHours"]=df["WklyStudyHours"].str.replace("10-May","5-10")
df.head()


# In[67]:


#Remove null values from columns
df=df.dropna()
print(df.head())


# In[68]:


df.isnull().sum()


# In[93]:


df = df.drop(['WklyStudyHours', 'PracticeSport'], axis=1)


# In[105]:


features = df.drop(['OverallGrade', 'MathGrade', 'ReadingGrade', 'WritingGrade'], axis=1)
target = df['OverallGrade']


# In[106]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# In[110]:


model = RandomForestRegressor(random_state=42)


# In[111]:


# Assuming 'A', 'B', 'C', etc. are grades
grade_mapping = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}

# Map grades to numerical values in the target variable
y_train = y_train.map(grade_mapping)
y_test = y_test.map(grade_mapping)
model.fit(X_train, y_train)


# In[112]:


y_pred = model.predict(X_test)


# In[113]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# In[116]:


feature_importances = model.feature_importances_
feature_names = features.columns


# In[126]:


fig, ax = plt.subplots(figsize=(10, 6))
plt.bar(feature_names, feature_importances)
plt.xlabel('Features', rotation=45, ha='right') 
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.tight_layout() 
plt.show()


# In[121]:


def convert_to_grades(predictions):
    return [round(p) for p in predictions]


# In[122]:


new_data = df.head()
new_predictions = model.predict(new_data.drop(['OverallGrade', 'MathGrade', 'ReadingGrade', 'WritingGrade'], axis=1))
predicted_grades = convert_to_grades(new_predictions)


# In[123]:


print("Predicted Overall Grades:")
print(predicted_grades)


# In[130]:


all_predictions = model.predict(features)


# In[132]:


new_data = df.tail()
new_predictions = model.predict(new_data.drop(['OverallGrade', 'MathGrade', 'ReadingGrade', 'WritingGrade'], axis=1))
predicted_grades = convert_to_grades(new_predictions)


# In[133]:


print("Predicted Overall Grades:")
print(predicted_grades)


# In[134]:


print("Predicted Overall Grades for the entire dataset:")
print(all_predictions)


# In[ ]:





# In[ ]:





# In[ ]:




