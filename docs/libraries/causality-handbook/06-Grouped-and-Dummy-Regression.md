# 06 - Grouped and Dummy Regression


## Regression With Grouped Data

Not all data points are created equal. If we look again at our ENEM dataset, we trust the scores of big schools much more than the scores from small schools. This is not to say that big schools are better or anything. It is just due to the fact that their big size imply less variance.


```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import style
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf

style.use("fivethirtyeight")
```


```python
np.random.seed(876)
enem = pd.read_csv("./data/enem_scores.csv").sample(200)
plt.figure(figsize=(8,4))
sns.scatterplot(y="avg_score", x="number_of_students", data=enem)
sns.scatterplot(y="avg_score", x="number_of_students", s=100, label="Trustworthy",
                data=enem.query(f"number_of_students=={enem.number_of_students.max()}"))
sns.scatterplot(y="avg_score", x="number_of_students", s=100, label="Not so Much",
                data=enem.query(f"avg_score=={enem.avg_score.max()}"))
plt.title("ENEM Score by Number of Students in the School");
```


    
![png](06-Grouped-and-Dummy-Regression_files/06-Grouped-and-Dummy-Regression_2_0.png)
    


In the data above, intuitively, points to the left should have less impact in my model than points to the right. In essence, points to the right are actually lots of other data points grouped into a single one. If we could unbundle them and run a linear regression on the ungrouped data, they would indeed contribute much more to the model estimation than an unbundled point in the left. 

This phenomenon of having a region of low variance and another of high variance is called **heteroskedasticity**. Put it simply, heteroskedasticity is when the variance is not constant across all values of the features. In the case above, we can see that the variance decreases as the feature sample size increases. To give another example of where we have heteroskedasticity, if you plot wage by age, you will see that there is higher wage variance for the old than for the young. But, by far, the most common reason for variance to differ is grouped data.

Grouped data like the one above are extremely common in data analysis. One reason for that is confidentiality. Governments and firms can't give away personal data because that would violate data privacy requirements they have to follow. If they need to export data to an outside researcher, they can only do it by means of grouping the data. This way, individuals get grouped together and are no longer uniquely identifiable.

Fortunately for us, regression can handle those kinds of data pretty well. To understand how, let's first take some ungrouped data like the one we had on wage and education. It contains one line per worker, so we know the wage for each individual in this dataset and also how many years of education he or she has.


```python
wage = pd.read_csv("./data/wage.csv")[["wage", "lhwage", "educ", "IQ"]]

wage.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>wage</th>
      <th>lhwage</th>
      <th>educ</th>
      <th>IQ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>769</td>
      <td>2.956212</td>
      <td>12</td>
      <td>93</td>
    </tr>
    <tr>
      <th>1</th>
      <td>808</td>
      <td>2.782539</td>
      <td>18</td>
      <td>119</td>
    </tr>
    <tr>
      <th>2</th>
      <td>825</td>
      <td>3.026504</td>
      <td>14</td>
      <td>108</td>
    </tr>
    <tr>
      <th>3</th>
      <td>650</td>
      <td>2.788093</td>
      <td>12</td>
      <td>96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>562</td>
      <td>2.642622</td>
      <td>11</td>
      <td>74</td>
    </tr>
  </tbody>
</table>
</div>



If we run a regression model to figure out how education is associated with log hourly wages, we get the following result.


```python
model_1 = smf.ols('lhwage ~ educ', data=wage).fit()
model_1.summary().tables[1]
```




<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    2.2954</td> <td>    0.089</td> <td>   25.754</td> <td> 0.000</td> <td>    2.121</td> <td>    2.470</td>
</tr>
<tr>
  <th>educ</th>      <td>    0.0529</td> <td>    0.007</td> <td>    8.107</td> <td> 0.000</td> <td>    0.040</td> <td>    0.066</td>
</tr>
</table>



Now, let's pretend for a moment that this data was under some confidentiality constraint. The provider of it was not able to give individualised data. So we ask him instead to group everyone by years of education and give us only the mean log hourly wage and the number of individuals in each group. This leaves us with only 10 data points.


```python
group_wage = (wage
              .assign(count=1)
              .groupby("educ")
              .agg({"lhwage":"mean", "count":"count"})
              .reset_index())

group_wage
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>educ</th>
      <th>lhwage</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>2.856475</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>2.786911</td>
      <td>35</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11</td>
      <td>2.855997</td>
      <td>43</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>2.922168</td>
      <td>393</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>3.021182</td>
      <td>85</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14</td>
      <td>3.042352</td>
      <td>77</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15</td>
      <td>3.090766</td>
      <td>45</td>
    </tr>
    <tr>
      <th>7</th>
      <td>16</td>
      <td>3.176184</td>
      <td>150</td>
    </tr>
    <tr>
      <th>8</th>
      <td>17</td>
      <td>3.246566</td>
      <td>40</td>
    </tr>
    <tr>
      <th>9</th>
      <td>18</td>
      <td>3.144257</td>
      <td>57</td>
    </tr>
  </tbody>
</table>
</div>



Fear not! Regression doesn't need big data to work! What we can do is provide weights to our linear regression model. This way, it will consider groups with higher sample size more than the small groups. Notice how I've replaced the `smf.ols` with `smf.wls`, for weighted least squares. It's hard to notice, but it will make all the difference.


```python
model_2 = smf.wls('lhwage ~ educ', data=group_wage, weights=group_wage["count"]).fit()
model_2.summary().tables[1]
```




<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    2.2954</td> <td>    0.078</td> <td>   29.327</td> <td> 0.000</td> <td>    2.115</td> <td>    2.476</td>
</tr>
<tr>
  <th>educ</th>      <td>    0.0529</td> <td>    0.006</td> <td>    9.231</td> <td> 0.000</td> <td>    0.040</td> <td>    0.066</td>
</tr>
</table>



Notice how the parameter estimate of `educ` in the grouped model is very close to the one in the ungrouped data (actually, they are the same in this case). Also, even with only 10 data points, we've managed to get a statistically significant coefficient. That's because, although we have fewer points, grouping also lowers the variance by a lot. Also notice how the standard error is a bit smaller and the t statistics is a bit larger. That's because some information about the variance is lost, so we have to be more conservative. Once we group the data, we don't know how large the variance is within each group. Compare the results above with what we would have with the non weighted model below.


```python
model_3 = smf.ols('lhwage ~ educ', data=group_wage).fit()
model_3.summary().tables[1]
```




<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    2.3650</td> <td>    0.082</td> <td>   28.988</td> <td> 0.000</td> <td>    2.177</td> <td>    2.553</td>
</tr>
<tr>
  <th>educ</th>      <td>    0.0481</td> <td>    0.006</td> <td>    8.136</td> <td> 0.000</td> <td>    0.034</td> <td>    0.062</td>
</tr>
</table>



The parameter estimate is smaller. What is happening here is that the regression is placing equal weight for all points. If we plot the model along the grouped points, we see that the non weighted model is giving more importance to  small points in the lower left than it should. As a consequence, the line has a lower slope.


```python
sns.scatterplot(x="educ", y = "lhwage", size="count", legend=False, data=group_wage, sizes=(40, 400))
plt.plot(wage["educ"], model_2.predict(wage["educ"]), c="C1", label = "Weighted")
plt.plot(wage["educ"], model_3.predict(wage["educ"]), c="C2", label = "Non Weighted")
plt.xlabel("Years of Education")
plt.ylabel("Log Hourly Wage")
plt.legend();
```


    
![png](06-Grouped-and-Dummy-Regression_files/06-Grouped-and-Dummy-Regression_14_0.png)
    


The bottom line is that regression is this marvellous tool that works both with individual or aggregated data, but you have to use weights in this last case. To use weighted regression you need mean statistics. Not sums, not standard deviations, not medians, but means! For both the covariates and the dependent variable. Just keep in mind that the result of weighted regression with grouped data won't match exactly that of regression in ungrouped data, but it will be pretty similar. 

![img](./data/img/dummy/heterosk.png)

I'll finish with a final example using additional covariates in a grouped data model.


```python
group_wage = (wage
              .assign(count=1)
              .groupby("educ")
              .agg({"lhwage":"mean", "IQ":"mean", "count":"count"})
              .reset_index())

model_4 = smf.wls('lhwage ~ educ + IQ', data=group_wage, weights=group_wage["count"]).fit()
print("Number of observations:", model_4.nobs)
model_4.summary().tables[1]
```

    Number of observations: 10.0





<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    1.8821</td> <td>    0.324</td> <td>    5.800</td> <td> 0.001</td> <td>    1.115</td> <td>    2.649</td>
</tr>
<tr>
  <th>educ</th>      <td>    0.0257</td> <td>    0.021</td> <td>    1.198</td> <td> 0.270</td> <td>   -0.025</td> <td>    0.077</td>
</tr>
<tr>
  <th>IQ</th>        <td>    0.0077</td> <td>    0.006</td> <td>    1.309</td> <td> 0.232</td> <td>   -0.006</td> <td>    0.022</td>
</tr>
</table>



In this example, we've included IQ as a feature, besides the previously added years of education. The mechanics is pretty much the same: get the means and count, regress the mean and use the count as weights. 

## Regression for Dummies

Dummy variables are categorical variables we've encoded as binary columns. For example, suppose you have a gender variable that you wish to include in your model. This variable is encoded into 3 categories: male, female and other genders. 

|gender|
|------|
|male  |
|female|
|female|
|other |
|male  |

Since our model only accepts numerical values, we need to convert this category to a number. In linear regression, we use dummies for that. We encode each variable as a 0/1 column, denoting the presence of a category. We also leave one of the categories out as the base category. This is necessary since the last category is a linear combination of the others. Put it differently, we can know the last category if someone gives us information on the others. In our example, if someone is neither female nor other genders, we can infer that the person's category is male.

|gender|female|other|
|------|:-----|:----|
|male  |0|0|
|female|1|0|
|female|1|0|
|other |0|1|
|male  |0|0|

We've already dealt with a simple form of dummy regression when dealing with A/B testing. More generally, when we are dealing with a binary treatment, we represent it as a dummy variable. In this case, **the regression coefficient for that dummy is the increment for the intercept in the regression line**, or the difference in means between the treated and untreated.

To make this more concrete, consider the problem of estimating the effect of graduating 12th grade on hourly wage (and let's ignore confounding just for now). In the code below, we've created a treatment dummy variable `T` indicating if years of education is greater than 12. 


```python
wage = (pd.read_csv("./data/wage.csv")
        .assign(hwage=lambda d: d["wage"] / d["hours"])
        .assign(T=lambda d: (d["educ"] > 12).astype(int)))

wage[["hwage", "IQ", "T"]].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hwage</th>
      <th>IQ</th>
      <th>T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19.225</td>
      <td>93</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16.160</td>
      <td>119</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20.625</td>
      <td>108</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.250</td>
      <td>96</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.050</td>
      <td>74</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The dummy works as a kind of switch. In our example, if the dummy is on, the predicted value is the intercept plus the dummy coefficient. If the dummy is off, the predicted value is just the intercept. 


```python
smf.ols('hwage ~ T', data=wage).fit().summary().tables[1]
```




<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   19.9405</td> <td>    0.436</td> <td>   45.685</td> <td> 0.000</td> <td>   19.084</td> <td>   20.797</td>
</tr>
<tr>
  <th>T</th>         <td>    4.9044</td> <td>    0.626</td> <td>    7.830</td> <td> 0.000</td> <td>    3.675</td> <td>    6.134</td>
</tr>
</table>



In this case, when the person hasn't completed 12th grade (dummy off), the average income is 19.9. When he or she has completed 12th grade (dummy on), the predicted value or the average income is 24.8449 (19.9405 + 4.9044). Hence, the dummy coefficient captures the difference in means, which is 4.9044 in our case.

More formally, when the independent variable is binary, as is often the case with treatment indicators, regression captures the ATE perfectly. That is because regression is a linear approximation to the conditional expectation function (CEF) $E[Y|X]$ and, in this particular case, the CEF IS linear. Namely, we can define $E[Y_i|T_i=0]=\alpha$ and $E[Y_i|T_i=1] = \alpha + \beta$, which leads to the following CEF

$
E[Y_i|T_i] =  E[Y_i|T_i=0] + \beta T_i = \alpha + \beta T_i
$

and $\beta$ is the difference in means or the ATE in the case of random data

$
\beta = [Y_i|T_i=1] - [Y_i|T_i=0]
$

If we use additional variables, the dummy coefficient becomes the **conditional** difference in means. For instance, let's say we add IQ to the previous model. Now, the dummy coefficient tells us how much increase we should expect from graduating 12th grade **while holding IQ fixed**. If we plot the prediction, we will see two parallel lines. The jump from one line to the next says the amount we should expect for completing 12th grade. They also say that the effect is constant. No matter your IQ, everyone benefits the same from graduating 12th grade. 


```python
m = smf.ols('hwage ~ T+IQ', data=wage).fit()
plt_df = wage.assign(y_hat = m.fittedvalues)

plt.plot(plt_df.query("T==1")["IQ"], plt_df.query("T==1")["y_hat"], c="C1", label="T=1")
plt.plot(plt_df.query("T==0")["IQ"], plt_df.query("T==0")["y_hat"], c="C2", label="T=0")
plt.title(f"E[T=1|IQ] - E[T=0|IQ] = {round(m.params['T'], 2)}")
plt.ylabel("Wage")
plt.xlabel("IQ")
plt.legend();
```


    
![png](06-Grouped-and-Dummy-Regression_files/06-Grouped-and-Dummy-Regression_22_0.png)
    


If we put this model into an equation, we can see why:

$
wage_i = \beta_0 + \beta_1T_i + \beta_2 IQ_i + e_i
$

Here, $\beta_1$ is the conditional difference in means and it is a constant value, 3.16 in our case. We can make this model more flexible by adding an interaction term.

$
wage_i = \beta_0 + \beta_1T_i + \beta_2 IQ_i + \beta_3 IQ_i * T_i  + e_i
$

Things are getting a little bit more complex, so let's see what each parameter means in this model. First, the intercept $\beta_0$. This bad boy doesn't have a particularly interesting interpretation. It's the expected wage when the treatment is zero (the person hasn't graduated from 12th grade) AND the IQ is zero. Since we don't expect IQ to be zero for anyone this parameter is not very meaningful. As for $\beta_1$, we have a similar situation. This parameter is how much increase in wage we should expect from completing 12th grade **when IQ is zero**. Once again, since IQ is never zero, it doesn't have a particularly interesting meaning. Now, $\beta_2$ is a bit more interesting. It tells us how much IQ increases wages **for the non-treated**. So, in our case, it is something like 0.11. This means that for each 1 extra IQ point, the person that has not completed 12th grade should expect to gain an extra 11 cents per hour. Finally, the most interesting parameter is $\beta_3$. It tells us how much IQ increases the effect of graduating 12th grade. In our case, this parameter is 0.024, which means that for each extra IQ point, graduating 12th grade gives 2 extra cents. This might not seem much, but compare someone with 60IQ and with 140IQ. The first one will get an increase of 1.44 in wage (60 * 0.024), while the person with 140 IQ will gain an extra 3.36 dollars (140 * 0.024) when graduating from 12th grade.

In simple modeling jargon, this interaction term allows the treatment effect to change by levels of the features (only IQ, in this example). The result is that if we plot the prediction lines, we will see that they are no longer parallel and that those that graduate 12th grade (T=1) have a higher slope on IQ, higher IQ benefit more from graduating than lower IQ. This is sometimes referenced as effect modification or heterogeneous treatment effect. 


```python
m = smf.ols('hwage ~ T*IQ', data=wage).fit()
plt_df = wage.assign(y_hat = m.fittedvalues)

plt.plot(plt_df.query("T==1")["IQ"], plt_df.query("T==1")["y_hat"], c="C1", label="T=1")
plt.plot(plt_df.query("T==0")["IQ"], plt_df.query("T==0")["y_hat"], c="C2", label="T=0")
plt.title(f"E[T=1|IQ] - E[T=0|IQ] = {round(m.params['T'], 2)}")
plt.ylabel("Wage")
plt.xlabel("IQ")
plt.legend();
```


    
![png](06-Grouped-and-Dummy-Regression_files/06-Grouped-and-Dummy-Regression_24_0.png)
    


Finally, let's look at the case where all the variables in our model are dummies. To do so, we will discretize IQ into 4 bins and treat years of education as a category.


```python
wage_ed_bins = (wage
                .assign(IQ_bins = lambda d: pd.qcut(d["IQ"], q=4, labels=range(4)))
                [["hwage", "educ", "IQ_bins"]])

wage_ed_bins.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hwage</th>
      <th>educ</th>
      <th>IQ_bins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19.225</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16.160</td>
      <td>18</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20.625</td>
      <td>14</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.250</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.050</td>
      <td>11</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Treating education as a category, we no longer restrict the effect of education to a single parameter. Instead, we allow each year of education to have its own distinct impact. By doing so, we gain flexibility, since the effect of education is no longer parametric. This model simply computes the mean wage for each year of education.


```python
model_dummy = smf.ols('hwage ~ C(educ)', data=wage).fit()
model_dummy.summary().tables[1]
```




<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td>   18.5600</td> <td>    3.011</td> <td>    6.164</td> <td> 0.000</td> <td>   12.651</td> <td>   24.469</td>
</tr>
<tr>
  <th>C(educ)[T.10]</th> <td>   -0.7874</td> <td>    3.414</td> <td>   -0.231</td> <td> 0.818</td> <td>   -7.488</td> <td>    5.913</td>
</tr>
<tr>
  <th>C(educ)[T.11]</th> <td>    0.1084</td> <td>    3.343</td> <td>    0.032</td> <td> 0.974</td> <td>   -6.452</td> <td>    6.669</td>
</tr>
<tr>
  <th>C(educ)[T.12]</th> <td>    1.7479</td> <td>    3.049</td> <td>    0.573</td> <td> 0.567</td> <td>   -4.236</td> <td>    7.732</td>
</tr>
<tr>
  <th>C(educ)[T.13]</th> <td>    4.3290</td> <td>    3.183</td> <td>    1.360</td> <td> 0.174</td> <td>   -1.918</td> <td>   10.576</td>
</tr>
<tr>
  <th>C(educ)[T.14]</th> <td>    4.0888</td> <td>    3.200</td> <td>    1.278</td> <td> 0.202</td> <td>   -2.192</td> <td>   10.370</td>
</tr>
<tr>
  <th>C(educ)[T.15]</th> <td>    6.3013</td> <td>    3.329</td> <td>    1.893</td> <td> 0.059</td> <td>   -0.231</td> <td>   12.834</td>
</tr>
<tr>
  <th>C(educ)[T.16]</th> <td>    7.2225</td> <td>    3.110</td> <td>    2.323</td> <td> 0.020</td> <td>    1.120</td> <td>   13.325</td>
</tr>
<tr>
  <th>C(educ)[T.17]</th> <td>    9.5905</td> <td>    3.366</td> <td>    2.849</td> <td> 0.004</td> <td>    2.984</td> <td>   16.197</td>
</tr>
<tr>
  <th>C(educ)[T.18]</th> <td>    7.3681</td> <td>    3.264</td> <td>    2.257</td> <td> 0.024</td> <td>    0.962</td> <td>   13.775</td>
</tr>
</table>




```python
plt.scatter(wage["educ"], wage["hwage"])
plt.plot(wage["educ"].sort_values(), model_dummy.predict(wage["educ"].sort_values()), c="C1")
plt.xlabel("Years of Education")
plt.ylabel("Hourly Wage");
```


    
![png](06-Grouped-and-Dummy-Regression_files/06-Grouped-and-Dummy-Regression_29_0.png)
    


First of all, notice how this removes any assumption about the functional form of how education affects wages. We don't need to worry about logs anymore. In essence, this model is completely non-parametric. All it does is compute sample averages of wage for each year of education. This can be seen in the plot above, where the fitted line doesn't have a particular form. Instead, is the interpolation of the sample means for each year of education. We can also see that by reconstructing one parameter, for instance, that of 17 years of education. For this model, it's `9.5905`. Below, we can see how it is just the difference between the baseline years of education (9) and the individuals with 17 years

$
\beta_{17} = E[Y|T=17]-E[Y|T=9]
$

The trade-off is that we lose statistical significance when we allow such flexibility. Notice how big the p-values are for some years.


```python
t1 = wage.query("educ==17")["hwage"]
t0 = wage.query("educ==9")["hwage"]
print("E[Y|T=9]:", t0.mean())
print("E[Y|T=17]-E[Y|T=9]:", t1.mean() - t0.mean())
```

    E[Y|T=9]: 18.56
    E[Y|T=17]-E[Y|T=9]: 9.590472362353516


If we include more dummy covariates in the model, the parameter on education become a weighted average of the effect on each dummy group:

$
E\{ \ (E[Y_i|T=1, Group_i] - E[Y_i|T=0, Group_i])w(Group_i) \ \}
$

$w(Group_i)$ is not exactly, but is proportional to the variance of the treatment in the group $Var(T_i|Group_i)$. One natural question that arises from this is why not use the full nonparametric estimator, where the group weight is the sample size? This indeed is a valid estimator, but it is not what regression does. By using the treatment variance, regression is placing more weight on groups where the treatment varies a lot. This makes intuitive sense. If the treatment was almost constant (say 1 treated and everyone else untreated), it doesn't matter its sample size. It wouldn't provide much information about the treatment effect.


```python
model_dummy_2 = smf.ols('hwage ~ C(educ) + C(IQ_bins)', data=wage_ed_bins).fit()
model_dummy_2.summary().tables[1]
```




<table class="simpletable">
<tr>
         <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>       <td>   18.4178</td> <td>    2.991</td> <td>    6.158</td> <td> 0.000</td> <td>   12.548</td> <td>   24.288</td>
</tr>
<tr>
  <th>C(educ)[T.10]</th>   <td>   -1.2149</td> <td>    3.392</td> <td>   -0.358</td> <td> 0.720</td> <td>   -7.872</td> <td>    5.442</td>
</tr>
<tr>
  <th>C(educ)[T.11]</th>   <td>   -0.4687</td> <td>    3.332</td> <td>   -0.141</td> <td> 0.888</td> <td>   -7.008</td> <td>    6.070</td>
</tr>
<tr>
  <th>C(educ)[T.12]</th>   <td>    0.3400</td> <td>    3.059</td> <td>    0.111</td> <td> 0.912</td> <td>   -5.664</td> <td>    6.344</td>
</tr>
<tr>
  <th>C(educ)[T.13]</th>   <td>    2.4103</td> <td>    3.206</td> <td>    0.752</td> <td> 0.452</td> <td>   -3.882</td> <td>    8.702</td>
</tr>
<tr>
  <th>C(educ)[T.14]</th>   <td>    1.8040</td> <td>    3.238</td> <td>    0.557</td> <td> 0.578</td> <td>   -4.551</td> <td>    8.159</td>
</tr>
<tr>
  <th>C(educ)[T.15]</th>   <td>    3.8599</td> <td>    3.369</td> <td>    1.146</td> <td> 0.252</td> <td>   -2.752</td> <td>   10.472</td>
</tr>
<tr>
  <th>C(educ)[T.16]</th>   <td>    4.4060</td> <td>    3.171</td> <td>    1.390</td> <td> 0.165</td> <td>   -1.817</td> <td>   10.629</td>
</tr>
<tr>
  <th>C(educ)[T.17]</th>   <td>    6.7470</td> <td>    3.422</td> <td>    1.971</td> <td> 0.049</td> <td>    0.030</td> <td>   13.464</td>
</tr>
<tr>
  <th>C(educ)[T.18]</th>   <td>    4.3463</td> <td>    3.332</td> <td>    1.304</td> <td> 0.192</td> <td>   -2.194</td> <td>   10.886</td>
</tr>
<tr>
  <th>C(IQ_bins)[T.1]</th> <td>    1.4216</td> <td>    0.898</td> <td>    1.584</td> <td> 0.114</td> <td>   -0.340</td> <td>    3.183</td>
</tr>
<tr>
  <th>C(IQ_bins)[T.2]</th> <td>    2.9717</td> <td>    0.930</td> <td>    3.195</td> <td> 0.001</td> <td>    1.146</td> <td>    4.797</td>
</tr>
<tr>
  <th>C(IQ_bins)[T.3]</th> <td>    3.7879</td> <td>    1.022</td> <td>    3.708</td> <td> 0.000</td> <td>    1.783</td> <td>    5.793</td>
</tr>
</table>



![img](./data/img/dummy/you_little_shit.png)

## Key Ideas

We started this section by looking at how some data points are more important than others. Namely, those with higher sample size and lower variance should be given more weight when estimating a linear model. Then, we looked at how linear regression can even handle grouped anonymised data with elegance, provided we use sample weights in our model.

Next, we moved to dummy regression. We saw how it can be made a non parametric model that places no assumptions whatsoever on the functional form of how the treatment impacts the outcome. We then explored the intuition behind dummy regression.

## References

I like to think of this entire book as a tribute to Joshua Angrist, Alberto Abadie and Christopher Walters for their amazing Econometrics class. Most of the ideas here are taken from their classes at the American Economic Association. Watching them is what is keeping me sane during this tough year of 2020.
* [Cross-Section Econometrics](https://www.aeaweb.org/conference/cont-ed/2017-webcasts)
* [Mastering Mostly Harmless Econometrics](https://www.aeaweb.org/conference/cont-ed/2020-webcasts)

I'll also like to reference the amazing books from Angrist. They have shown me that Econometrics, or 'Metrics as they call it, is not only extremely useful but also profoundly fun.

* [Mostly Harmless Econometrics](https://www.mostlyharmlesseconometrics.com/)
* [Mastering 'Metrics](https://www.masteringmetrics.com/)

My final reference is Miguel Hernan and Jamie Robins' book. It has been my trustworthy companion in the most thorny causal questions I had to answer.

* [Causal Inference Book](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)

![img](./data/img/poetry.png)

## Contribute

Causal Inference for the Brave and True is an open-source material on causal inference, the statistics of science. It uses only free software, based in Python. Its goal is to be accessible monetarily and intellectually.
If you found this book valuable and you want to support it, please go to [Patreon](https://www.patreon.com/causal_inference_for_the_brave_and_true). If you are not ready to contribute financially, you can also help by fixing typos, suggesting edits or giving feedback on passages you didn't understand. Just go to the book's repository and [open an issue](https://github.com/matheusfacure/python-causality-handbook/issues). Finally, if you liked this content, please share it with others who might find it useful and give it a [star on GitHub](https://github.com/matheusfacure/python-causality-handbook/stargazers).




```python

```
