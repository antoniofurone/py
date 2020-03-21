import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))


url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
df = pd.read_csv(url)
print(df)
#df = df.loc[:,['data','totale_casi']]
df = df.loc[:,['data','totale_attualmente_positivi']]
FMT = '%Y-%m-%d %H:%M:%S'
date = df['data']
df['data'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01 00:00:00", FMT)).days  )
print(df)

x = list(df.iloc[:,0])
y = list(df.iloc[:,1])
fit = curve_fit(logistic_model,x,y,p0=[2,100,20000])

a=fit[0][0] 
print("infection speed:",a)
b=fit[0][1] 
print("day with max infections:",b)
c=fit[0][2]
print("total infections at the end:",c)

sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))
#infection end day
print("infection end day:",sol)

pred_x = list(range(max(x),sol))

# Real data
plt.scatter(x,y,label="Real data",color="red")
# Predicted logistic curve
plt.plot(x+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x+pred_x], label="Logistic model" ) 

plt.legend()
plt.xlabel("Days")
plt.title("Total Cases")

plt.show()