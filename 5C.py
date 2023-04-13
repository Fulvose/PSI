model = ols("PRICE ~ CRIM + I(ZN)+I(RAD)+I(MEDV)", bos).fit()
# Print the summary
print((model.summary()))