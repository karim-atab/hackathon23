import pandas as pd 

dp2 = pd.read_csv('convo.csv')
col = ['ConvoID','CallerID','Script']
dp2 = dp2[col]
dp2.columns = ['ConvoID','CallerID','Script']
print(dp2)
test = {'Script':'sum'}
dp3 = dp2.groupby(['ConvoID','CallerID']).aggregate(test)
dp3.to_csv('test.csv', encoding='utf-8')
print(dp3)