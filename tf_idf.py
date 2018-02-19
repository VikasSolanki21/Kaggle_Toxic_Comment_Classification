import numpy as np # linear algebra
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack

train = pd.read_csv("filtered_train1.csv")
test = pd.read_csv("filtered_test1.csv")
df = train.iloc[:,:2].append(test,ignore_index=True)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
COMMENT = 'filtered_text'


train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)
df[COMMENT].fillna("unknown", inplace=True)

class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self
print ('Before training')

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    ngram_range=(1, 2),max_features=50000).fit(df[COMMENT])
train_word_features = word_vectorizer.transform(train[COMMENT])
test_word_features = word_vectorizer.transform(test[COMMENT])


vec = TfidfVectorizer(ngram_range=(1,5),analyzer='char',max_features=50000,
               min_df=1, max_df=1.0, strip_accents='unicode', sublinear_tf=1).fit(df[COMMENT])

train_char_features = vec.transform(train[COMMENT])
test_char_features = vec.transform(test[COMMENT])

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])
print ('Training completed')

s=0
for lab in label_cols:
    model = NbSvmClassifier(C=4, dual=True)#.fit(trnd_term_doc, traind[lab])

    cv_loss = np.mean(cross_val_score(model, train_features, train[lab], cv=3, scoring='roc_auc'))
    #y_pred = model.predict_proba(val_term_doc)[:,1]
    print ('Score for class {} is {}'.format(lab,cv_loss))
    s+=cv_loss

print (s/len(label_cols))

#Final Predictions

preds = np.zeros((len(test), len(label_cols)))

for i,j in enumerate(label_cols):
    
    print('fit', j)
    model = NbSvmClassifier(C=4, dual=True).fit(train_features, train[j])
    preds[:,i] = model.predict_proba(test_features)[:,1]
subm = pd.read_csv('sample_submission.csv')
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('nbsvm_ensemble_best.csv',index=False)


