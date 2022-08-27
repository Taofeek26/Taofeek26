import numpy as np
import pandas as pd

#Setting the recent season match
yrBefore = np.arange(1900,2023)
yrAfter = np.arange(1901,2024)
yrBefore_list = []
yrAfter_list = []

for s in yrBefore:
    a = str(s)
    yrBefore_list.append(a)

for j in yrAfter:
    b = str(j)
    yrAfter_list.append(b)

season_list = []
for f in range (len(yrBefore)):
    season = yrBefore_list[f] + '/' + yrAfter_list[f]
    season_list.append(season)


#Getting Table from online
df_bt = pd.read_html("https://www.soccerbase.com/teams/team.sd?team_id=2898&team2_id=376&teamTabs=h2h")

#Picking Table From Source
sdf= df_bt[2]
startingYear = sdf.columns[0]
if startingYear in season_list:
    x = startingYear
else:
    print ('No past record of the teams')
    
y = x
r = x + '.1'
n = x + '.2'
m = x + '.7'
l = x + '.8'
p = x + '.9'

new_df = sdf[sdf[r].apply(lambda x: x[4])!= '/']

new_df.drop(y, axis = 1, inplace = True)

new_df.set_index(r,inplace= True)

new_df.drop([n, m,l,p], axis = 1, inplace = True)

new_df.columns = ['Home', 'Scores', 'Away', 'Result']
new_df.index.names = ['Date']


new_df['ScoresH'] = new_df['Scores'].apply(lambda x: x[0])
new_df['ScoresA'] = new_df['Scores'].apply(lambda x: x[4])

new_df['ScoresH'] = new_df['ScoresH'].apply(lambda x: int(x))
new_df['ScoresA'] = new_df['ScoresA'].apply(lambda x: int(x))
new_df['ResultN'] = new_df['ScoresH'] - new_df['ScoresA']

new_df['Result'][new_df['ResultN']>0]=new_df['Home']
new_df['Result'][new_df['ResultN']<0]=new_df['Away']
new_df['Result'][new_df['ResultN']==0]='Draw'


new_df['Result']= new_df['Result'] + ' Wins'

Result = pd.get_dummies(new_df['Result'])
Home = pd.get_dummies(new_df['Home'])
Away = pd.get_dummies(new_df['Away'])

new_df.drop(['Home','Scores', 'Away'], axis = 1,inplace = True)

ddf= pd.concat([new_df,Result,Home,Away],axis = 1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
for i in Result:
    x = i
    print(x.upper())
    X_train, X_test, y_train, y_test = train_test_split(ddf.drop([x,'Result'],axis=1), 
                                                    ddf[x], test_size=0.30, 
                                                    random_state=101)

    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    predictions = logmodel.predict(X_test)
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,predictions))
