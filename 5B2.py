#1
import statsmodels.formula.api as smf

# formula: response ~ predictor + predictor
est = smf.ols(formula='sales ~ I(np.square(np.log(TV)))*I(radio)', data=df_adv).fit()
print((est.summary2()))

#2
import statsmodels.formula.api as smf

# formula: response ~ predictor + predictor
est = smf.ols(formula='sales ~ I(np.square(np.log(TV)))*I(radio)*np.square(I(newspaper))', data=df_adv).fit()
print((est.summary2()))

#3
est = smf.ols(formula='sales ~ I(radio) + I(TV) + I(np.log(np.power(TV, 2))) + I(radio):I(TV) ', data=df_adv).fit()

print((est.summary2()))