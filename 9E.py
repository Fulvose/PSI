train_data["Survived"].value_counts()
train_data.isnull().sum()
X = train_data.drop(['Survived'], axis=1)
y = train_data['Survived'].values
np.unique(y)
y[ y == ' <=50K'] = 0
y[ y == ' >50K'] = 1

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("X.shape: {} y.shape: {}".format(X.shape, y.shape))

y_train = X.values
X_train = X

train_data["Sex"].value_counts()
train_data["Ticket"].value_counts()
train_data["Embarked"].value_counts()

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
    ])
num_pipeline.fit_transform(X_train)

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

        from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])