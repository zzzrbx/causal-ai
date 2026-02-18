## collections_email


```python
import pandas as pd
import numpy as np

np.random.seed(24)
n = 5000
email = np.random.binomial(1, 0.5, n)

credit_limit = np.random.gamma(6, 200, n)
risk_score = np.random.beta(credit_limit, credit_limit.mean(), n)

opened = np.random.normal(5 + 0.001*credit_limit - 4*risk_score, 2)
opened = (opened > 4).astype(float) * email


agreement = np.random.normal(30 +(-0.003*credit_limit - 10*risk_score), 7) * 2 * opened
agreement = (agreement > 40).astype(float)

payments = (np.random.normal(500 + 0.16*credit_limit - 40*risk_score + 11*agreement + email, 75).astype(int) // 10) * 10

data = pd.DataFrame(dict(payments=payments,
                         email=email,
                         opened=opened,
                         agreement=agreement,
                         credit_limit=credit_limit,
                         risk_score=risk_score))

data.to_csv("collections_email.csv", index=False)
```

## hospital_treatment


```python
import pandas as pd
import numpy as np

np.random.seed(24)
n = 80

hospital = np.random.binomial(1, 0.5, n)

treatment = np.where(hospital.astype(bool),
                     np.random.binomial(1, 0.9, n),
                     np.random.binomial(1, 0.1, n))

severity = np.where(hospital.astype(bool), 
                    np.random.normal(20, 5, n),
                    np.random.normal(10, 5, n))

days = np.random.normal(15 + -5*treatment + 2*severity, 7).astype(int)

hospital = pd.DataFrame(dict(hospital=hospital,
                             treatment=treatment,
                             severity=severity,
                             days=days))

hospital.to_csv("hospital_treatment.csv", index=False)
```

## app engagement push


```python
import pandas as pd
import numpy as np

np.random.seed(24)
n = 10000

push_assigned = np.random.binomial(1, 0.5, n)

income = np.random.gamma(6, 200, n)

push_delivered = np.random.normal(5 + 0.3+income, 500)
push_delivered = ((push_delivered > 800) & (push_assigned == 1)).astype(int)

in_app_purchase = (np.random.normal(100 + 20*push_delivered + 0.5*income, 75).astype(int) // 10)

data = pd.DataFrame(dict(in_app_purchase=in_app_purchase,
                         push_assigned=push_assigned,
                         push_delivered=push_delivered))

data.to_csv("app_engagement_push.csv", index=False)
```

## Drug Impact


```python
import numpy as np
import pandas as pd

def make_confounded_data(N):

    def get_severity(df):
        return ((np.random.beta(1, 3, size=df.shape[0]) * (df["age"] < 30)) +
                (np.random.beta(3, 1.5, size=df.shape[0]) * (df["age"] >= 30)))

    def get_treatment(df):
        return ((.33 * df["sex"] +
                1.5 * df["severity"] +  df["severity"] ** 2 +
                0.15 * np.random.normal(size=df.shape[0])) > 1.5).astype(int)

    def get_recovery(df):
        return ((2 +
                0.5 * df["sex"] +
                0.03 * df["age"] + 0.03 * ((df["age"] * 0.1) ** 2) +
                df["severity"] + np.log(df["severity"]) +
                df["sex"] * df["severity"] -
                df["medication"]) * 10).astype(int)

    np.random.seed(1111)
    sexes = np.random.randint(0, 2, size=N)
    ages = np.random.gamma(8, scale=4, size=N)
    meds = np.random.beta(1, 1, size=N)

    # dados com designação aleatória
    df_rnd = pd.DataFrame(dict(sex=sexes, age=ages, medication=meds))
    df_rnd['severity'] = get_severity(df_rnd)
    df_rnd['recovery'] = get_recovery(df_rnd)

    features = ['sex', 'age', 'severity', 'medication', 'recovery']
    df_rnd = df_rnd[features]  # to enforce column order

    # dados observacionais
    df_obs = df_rnd.copy()
    df_obs['medication'] = get_treatment(df_obs)
    df_obs['recovery'] = get_recovery(df_obs)

    # dados contrafactuais data
    df_ctf = df_obs.copy()
    df_ctf['medication'] = ((df_ctf['medication'] == 1) ^ 1).astype(float)
    df_ctf['recovery'] = get_recovery(df_ctf)

    return df_rnd, df_obs, df_ctf

np.random.seed(1234)
df_rnd, df_obs, df_ctf = make_confounded_data(20000)

df_obs.to_csv("medicine_impact_recovery.csv", index=False)
```

## Bilboard Mkt


```python
import pandas as pd
import numpy as np
np.random.seed(123)
POAMay = np.random.gamma(7,10, 500) * np.random.binomial(1, .7, 500)
POAJul = np.random.gamma(7,15, 800) * np.random.binomial(1, .8, 800)
FLMay = np.random.gamma(10,20, 1300) * np.random.binomial(1, .85, 1300)
FLJul = np.random.gamma(11,21, 2000) * np.random.binomial(1, .9, 2000)

data = pd.concat([
    pd.DataFrame(dict(deposits = POAMay.astype(int), poa=1, jul=0)),
    pd.DataFrame(dict(deposits = POAJul.astype(int), poa=1, jul=1)),
    pd.DataFrame(dict(deposits = FLMay.astype(int), poa=0, jul=0)),
    pd.DataFrame(dict(deposits = FLJul.astype(int), poa=0, jul=1))
])
data.to_csv("billboard_impact.csv", index=False)
```

## Customer Lifecycle


```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from toolz import merge
from sklearn.preprocessing import LabelEncoder

np.random.seed(12)

n = 10000
t = 30

age = 18 + np.random.poisson(10, n)
income = 500+np.random.exponential(2000, size=n).astype(int)
region = np.random.choice(np.random.lognormal(4, size=50), size=n)

freq = np.random.lognormal((1 + age/(18+10)).astype(int))
churn = np.random.poisson((income-500)/2000 + 22, n)

ones = np.ones((n, t))
alive = (np.cumsum(ones, axis=1) <= churn.reshape(n, 1)).astype(int)
buy = np.random.binomial(1, ((1/(freq+1)).reshape(n, 1) * ones))

cacq = -1*abs(np.random.normal(region, 2, size=n).astype(int))
transactions = np.random.lognormal(((income.mean() - 500) / 1000), size=(n, t)).astype(int) * buy * alive

data = pd.DataFrame(merge({"customer_id": range(n), "cacq":cacq},
                          {f"day_{day}": trans 
                           for day, trans in enumerate(transactions.T)}))

encoced = {value:index for index, value in
           enumerate(np.random.permutation(np.unique(region)))}

customer_features = pd.DataFrame(dict(customer_id=range(n), 
                                      region=region,
                                      income=income,
                                      age=age)).replace({"region":encoced}).astype(int)

print((data.drop(columns=["customer_id"]).sum(axis=1) > 0).mean()) # proportion of profitable customers
print((alive).mean(axis=0)) # alive customer per days

data.to_csv("./causal-inference-for-the-brave-and-true/data/customer_transactions.csv", index=False)
customer_features.to_csv("./causal-inference-for-the-brave-and-true/data/customer_features.csv", index=False)
```

    0.3721
    [1.     1.     1.     1.     1.     1.     1.     0.9999 0.9994 0.9984
     0.9966 0.994  0.9886 0.9791 0.9663 0.944  0.9128 0.8726 0.8205 0.7603
     0.6932 0.6138 0.5295 0.4424 0.3618 0.2919 0.2308 0.1769 0.1286 0.0942]


## Price and Sales


```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

np.random.seed(5)

def price_elast(price, temp, weekday, cost):
    return  -4 + 0.2*price + 0.05*temp + 2*np.isin(weekday, [1,7]) + 0.3 * cost

def sales(price, temp, weekday, cost):
    elast = -abs(price_elast(price, temp, weekday, cost))
    output = np.random.normal(200 + 20*np.isin(weekday, [1,7]) + 1.3 * temp +
                              5*elast * price, 5).astype(int)
    
    return output


n_rnd = 5000

temp = np.random.normal(24, 4, n_rnd).round(1)
weekday = np.random.choice(list(range(1, 8)), n_rnd)
cost = np.random.choice([0.3, 0.5, 1.0, 1.5], n_rnd)
price_rnd = np.random.choice(list(range(3, 11)), n_rnd)

price_df_rnd = pd.DataFrame(dict(temp=temp, weekday=weekday, cost=cost,
                                 price=price_rnd, sales=sales(price_rnd, temp, weekday, cost)))

n = 10000
temp = np.random.normal(24, 4, n).round(1)
weekday = np.random.choice(list(range(1, 8)), n)
cost = np.random.choice([0.3, 0.5, 1.0, 1.5], n)
price = np.random.normal(5 + cost + np.isin(weekday, [1,7])).round(1)

price_df = pd.DataFrame(dict(temp=temp, weekday=weekday, cost=cost,
                             price=price, sales=sales(price, temp, weekday, cost)))

price_df_rnd.to_csv("./causal-inference-for-the-brave-and-true/data/ice_cream_sales_rnd.csv", index=False)
price_df.to_csv("./causal-inference-for-the-brave-and-true/data/ice_cream_sales.csv", index=False)
```

## Marketing Email


```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler((0, 1))

np.random.seed(12321)

n_rnd=5000

age = 18 + np.random.normal(24, 4, n_rnd).round(1)
income = 500 + np.random.gamma(1, age * 100, n_rnd).round(2)
insurance = np.random.gamma(30/age, age*1000, n_rnd).round(2)
invested = np.random.gamma(age/10, income/2, n_rnd).round(2)

em1_ps = income.min()/(income + 10)
em2_ps = invested/(invested.max())
em3_ps = np.where(age > 40, scaler.fit_transform(-income.reshape(-1,1)).ravel(), 0)

em1 = np.random.binomial(1, em1_ps)
em2 = np.random.binomial(1, em2_ps)
em3 = np.random.binomial(1, em3_ps)

elast_em1 = scaler.fit_transform((-3*age + 0.005*invested).reshape(-1,1)).ravel()
elast_em2 = scaler.fit_transform((age + income*0.005).reshape(-1,1)).ravel()
elast_em3 = scaler.fit_transform((-insurance).reshape(-1,1)).ravel()

buy = scaler.fit_transform((1 + 0.4*age - invested/10000).reshape(-1,1)).ravel()
buy += elast_em1*em1 + elast_em2*em2 + elast_em3*em3
buy = scaler.fit_transform(buy.reshape(-1,1)).ravel()
buy = np.random.binomial(1, buy).round(2)

df = pd.DataFrame(dict(age=age, income=income, insurance=insurance, invested=invested,
                       em1_ps=em1_ps, em2_ps=em2_ps, em3_ps=em3_ps,
                       em1=em1, em2=em2, em3=em3,
                       converted=buy))

df.to_csv("./causal-inference-for-the-brave-and-true/data/invest_email.csv", index=False)
```


```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler((0.001, 0.999))

np.random.seed(12321)

n_rnd=15000

age = 18 + np.random.normal(24, 4, n_rnd).round(1)
income = 500 + np.random.gamma(1, age * 100, n_rnd).round(2)
insurance = np.random.gamma(30/age, age*1000, n_rnd).round(2)
invested = np.random.gamma(age/10, income/2, n_rnd).round(2)

em1 = np.random.binomial(1, 0.5, n_rnd)
em2 = np.random.binomial(1, 0.2, n_rnd)
em3 = np.random.binomial(1, 0.9, n_rnd)

elast_em1 = scaler.fit_transform((-3*age + 0.005*invested).reshape(-1,1)).ravel()
elast_em2 = scaler.fit_transform((age + income*0.005).reshape(-1,1)).ravel()
elast_em3 = scaler.fit_transform((-insurance).reshape(-1,1)).ravel()

buy = (200*elast_em1*em1 + 100*elast_em2*em2 + 10*elast_em3*em3 
       + 1.5*age + 0.0005*invested - 0.0001*income)

buy = scaler.fit_transform(buy.reshape(-1,1)).ravel()

buy = np.random.binomial(1, buy)

df = pd.DataFrame(dict(age=age, income=income, insurance=insurance, invested=invested,
                       em1=em1, em2=em2, em3=em3,
                       converted=buy))

df.to_csv("./causal-inference-for-the-brave-and-true/data/invest_email_rnd.csv", index=False)
```


```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler((0.001, 0.999))

np.random.seed(12321)

n_rnd=15000

age = 18 + np.random.normal(24, 4, n_rnd).round(1)
income = 500 + np.random.gamma(1, age * 100, n_rnd).round(2)
insurance = np.random.gamma(30/age, age*1000, n_rnd).round(2)
invested = np.random.gamma(age/10, income/2, n_rnd).round(2)

em1_ps = income.min()/(income + 10)
em2_ps = invested/(invested.max())
em3_ps = np.where(age > 40, scaler.fit_transform(-income.reshape(-1,1)).ravel(), 0)


em1 = np.random.binomial(1, em1_ps)
em2 = np.random.binomial(1, em2_ps)
em3 = np.random.binomial(1, em3_ps)

elast_em1 = scaler.fit_transform((-3*age + 0.005*invested).reshape(-1,1)).ravel()
elast_em2 = scaler.fit_transform((age + income*0.005).reshape(-1,1)).ravel()
elast_em3 = scaler.fit_transform((-insurance).reshape(-1,1)).ravel()

buy = (200*elast_em1*em1 + 100*elast_em2*em2 + 10*elast_em3*em3 
       + 1.5*age + 0.0005*invested - 0.0001*income)

buy = scaler.fit_transform(buy.reshape(-1,1)).ravel()

buy = np.random.binomial(1, buy)

df = pd.DataFrame(dict(age=age, income=income, insurance=insurance, invested=invested,
                       em1=em1, em2=em2, em3=em3,
                       converted=buy))

df.to_csv("./causal-inference-for-the-brave-and-true/data/invest_email_biased.csv", index=False)
```


```python

```
