import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

train = pd.read_csv('filtered_train(copy).csv')
test = pd.read_csv('test_stacked.csv')
features = ['toxic-tf', 'severe_toxic-tf', 'obscene-tf', 'threat-tf', 'insult-tf',
       'identity_hate-tf', 'toxic-nn.1', 'severe_toxic-nn.1', 'obscene-nn.1',
       'threat-nn.1', 'insult-nn.1', 'identity_hate-nn.1']
target = ['toxic',
       'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
X_train = train[features].values
#y_train = train[target].values
X_test = test[features].values
preds = np.zeros((len(test), len(target)))

for i,j in enumerate(target):
    
    print('fit', j)
    model = xgb.XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05).fit(X_train, train[j])
    print (model.feature_importances_)
    preds[:,i] = model.predict_proba(X_test)[:,1]

subm = pd.read_csv('sample_submission.csv')
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = target)], axis=1)
print (submission.head(2))

submission.to_csv('stacked.csv',index=False)


#s=0
#for lab in target:
 #   features= [lab+'-tf',lab+'-nn.1']
  #  X_train = train[features].values
   # y_train = train[lab].values
    #model = xgb.XGBClassifier(max_depth=1, n_estimators=400, learning_rate=0.05)
   
  #  kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
   # cv_loss = (cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc'))
    #mean = np.mean(cv_loss)
    #std = np.std(cv_loss)
    #print (('{}+-{}').format(mean,std))
    #s+=mean
#print (s/len(target))

