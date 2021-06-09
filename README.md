- 👋 Hi, I’m @mehrjerd
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---
mehrjerd/mehrjerd is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->

**********************************************************
Hybrid  Methods
**********************************************************
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 12:58:41 2021

@author: pc8
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:08:24 2020

@author: Maxvel
"""

from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import scale,normalize,minmax_scale
from sklearn.metrics import accuracy_score
from time import time 
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


#%%
"random Selection Import Data"
ivfdata2=pd.read_excel('F://wrapperfilter2.xlsx')

ivftarget1=pd.read_excel('F://ivfpretarget.xlsx')

#print("ivf data and shape  :")
#print(ivfdata)
#print(ivfdata.shape)

#print("ivf target1 :")
#print(ivftarget1.shape)

#%%

ivfdatarand=ivfdata2.rename(columns={'Addiction':'Addiction','Day 3 FSH':'FSH','Duration of Infertility':'Duration','Endometrium Thickness':'Endometrium','Female Age':'FAge',
'No of Viable Occytes(MII)':'MII','Sperm Motility%':'Motility'})


#print(ivfdata)

ivfdata_feature_namesrand=['Addiction','FSH','Duration','Endometrium','FAge','MII',
'Motility']


ivfdata_dfranf=pd.DataFrame(ivfdatarand,columns=ivfdata_feature_namesrand)
#%%
"Find missing values"

ivfdatarand.replace(' ',np.nan,inplace=True)




#print('ivftarget1(output) shape is: ')
#print(ivftarget1.shape)
xrand=ivfdatarand
yrand=ivftarget1



#%%



"Test And Train"

x_trainrand,x_testrand,y_trainrand,y_testrand=train_test_split(xrand,yrand,test_size=0.3,random_state=42,stratify=yrand)


#%%
"Kbest Output Import Data fvalue"

ivfdata=pd.read_excel('F://ivffilterembedwrapperkbest.xlsx')
ivftarget1=pd.read_excel('F://ivfpretarget.xlsx')



#%%

ivfdata=ivfdata.rename(columns={'AFC':'AFC','Duration of Infertility':'Duration',
'Endometrium Thickness':'Endometrium','Male Age':'MAge','Day 3 FSH':'FSH','Female Age':'FAge',
'BMI':'BMI','No of Folli':'Folli','No of Frozen Embryos Pack':'Embryos','No of Occytes Collected':'Occytes',
'No of Viable Occytes(MII)':'MII',
'Sperm Count *106':'Count','Sperm Motility%':'Motility','Sperm Morphology%':'Morphology',
'Total Gonadotropin Dose':'Gonadotropin'})


ivfdata_feature_names=['AFC','Duration','Endometrium','Mage','FSH','FAge',
'BMI','Folli','Embryos','Occytes','MII','Count','Motility','Morphology',
'Gonadotropin']



ivfdata_df=pd.DataFrame(ivfdata,columns=ivfdata_feature_names)

#%%


#%%
"Find missing values"

ivfdata.replace(' ',np.nan,inplace=True)


x=ivfdata
y=ivftarget1

x1=ivfdata

x2=ivfdata
x3=ivfdata
#%%

"Test And Train"

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)

#%%

"Random Forest"
"Random Forest Classifier"
from sklearn.ensemble import RandomForestClassifier


rf_modelrand=RandomForestClassifier(n_estimators=200,max_features='sqrt', min_samples_split=5 , criterion='entropy',random_state=1234)
clf =RandomForestClassifier(n_estimators=200,max_features='sqrt', min_samples_split=5 , criterion='entropy',random_state=1234)

#%%

rf_modelrand.fit(x_trainrand,y_trainrand)

y_predrand=rf_modelrand.predict(x_testrand)

rf_predictionrand=rf_modelrand.predict_proba(x_testrand)





#%%
" Sequential Forward Selection "


sfs1=sfs(clf,
         k_features=7,
         forward=True,
         floating=False,
         verbose=2,
         scoring='accuracy',
         cv=4)

sfs1=sfs1.fit(x,y)




#print('features names selected  SFS: \n' , sfs1.k_feature_names_,'\n')

#%%

"Forward"
#
#print("Data Frame of feature selected SFS")
#print(pd.DataFrame.from_dict(sfs1.get_metric_dict()).T)



" Sequential Backward Selection "
sbs=sfs(clf,
         k_features=7,
         forward=False,
         floating=False,
         verbose=2,
         scoring='accuracy',
         cv=4)

sbs=sbs.fit(x1,y)



print('features names selected SBS : \n' , sbs.k_feature_names_,'\n')

#%%

"Backward"

print("Data Frame of feature selected SBS")
print(pd.DataFrame.from_dict(sbs.get_metric_dict()).T)


" Sequential floating Forward Selection "


sffs=sfs(clf,
         k_features=7,
         forward=True,
         floating=True,
         verbose=2,
         scoring='accuracy',
         cv=4)

sffs=sffs.fit(x2,y)


#print('features names selected SFFS : \n' , sffs.k_feature_names_,'\n')

#"Float Forward"

#print("Data Frame of feature selected SFSS")
#print(pd.DataFrame.from_dict(sffs.get_metric_dict()).T)
" Sequential floating Backward Selection "

sfbs=sfs(clf,
         k_features=7,
         forward=False,
         floating=True,
         verbose=2,
         scoring='accuracy',
         cv=4)

sfbs=sfbs.fit(x3,y)




#%%
" Build full model with selected features  "
" Feature selected in Forward  "


xnewsfs=sfs1.transform(x)

xnewsbs=sbs.transform(x)

xnewsffs=sffs.transform(x)

xnewsfbs=sfbs.transform(x)



"Test And Train"
x_trainsfs,x_testsfs,y_trainsfs,y_testsfs=train_test_split(xnewsfs,y,test_size=0.3,random_state=42,stratify=y)
x_trainsbs,x_testsbs,y_trainsbs,y_testsbs=train_test_split(xnewsbs,y,test_size=0.3,random_state=42,stratify=y)

x_trainsffs,x_testsffs,y_trainsffs,y_testsffs=train_test_split(xnewsffs,y,test_size=0.3,random_state=42,stratify=y)
x_trainsfbs,x_testsfbs,y_trainsfbs,y_testsfbs=train_test_split(xnewsfbs,y,test_size=0.3,random_state=42,stratify=y)

#%%


clf.fit(x_trainsfs,y_trainsfs)
y_predsfs=clf.predict(x_testsfs)
rf_predictionsfs=clf.predict_proba(x_testsfs)



clf.fit(x_trainsbs,y_trainsbs)
y_predsbs=clf.predict(x_testsbs)
rf_predictionsbs=clf.predict_proba(x_testsbs)


clf.fit(x_trainsffs,y_trainsffs)
y_predsffs=clf.predict(x_testsffs)
rf_predictionsffs=clf.predict_proba(x_testsffs)


clf.fit(x_trainsfbs,y_trainsfbs)
y_predsfbs=clf.predict(x_testsfbs)
rf_predictionsfbs=clf.predict_proba(x_testsfbs)


#%%
"Roc Curve " 
print('Roc Curve')
import matplotlib.lines as mlines

fig, ax = plt.subplots()

y_pred_probrand=rf_modelrand.predict_proba(x_testrand)[:,1]
fprrand,tprrand, thresholdsrand=roc_curve(y_testrand,y_pred_probrand)

y_pred_probsfs=clf.predict_proba(x_testsfs)[:,1]
fprsfs,tprsfs, thresholdssfs=roc_curve(y_testsfs,y_pred_probsfs)


y_pred_probsbs=clf.predict_proba(x_testsbs)[:,1]
fprsbs,tprsbs, thresholdssbs=roc_curve(y_testsbs,y_pred_probsbs)


y_pred_probsffs=clf.predict_proba(x_testsffs)[:,1]
fprsffs,tprsffs, thresholdssffs=roc_curve(y_testsffs,y_pred_probsffs)

y_pred_probsfbs=clf.predict_proba(x_testsfbs)[:,1]
fprsfbs,tprsfbs, thresholdssfbs=roc_curve(y_testsfbs,y_pred_probsfbs)




fig.suptitle('ROC Plot for Wrapper Methodes (IVF/ICSI)')
plt.plot([0,1],[0,1],'k--')

plt.plot(fprrand,tprrand,linewidth=1,label='Rand(AUC=0.759)')
plt.plot(fprsfs,tprsfs,linewidth=1,label='SFS(AUC=0.507)')
plt.plot(fprsbs,tprsbs,linewidth=1,label='SBS(AUC=0.565)')
plt.plot(fprsffs,tprsffs,linewidth=1,label='SFFS(AUC=0.507)')
plt.plot(fprsfbs,tprsfbs,linewidth=1,label='SFBS(AUC=0.744)')


plt.grid(True)
ax.set(facecolor = "lightgray")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()




aucsfbs=roc_auc_score(y_testsfbs,y_pred_probsfbs)
print('auc sfbs')
print(aucsfbs)




"Matthews Corrolation Cofficient (MCC)"

from sklearn .metrics import matthews_corrcoef
print('MCC sfbs : ' )
mccsfbs=matthews_corrcoef(y_predsfbs,y_testsfbs)
print(mccsfbs)

"Classification Report"
print('Classification Report')

print(classification_report(y_testsfbs,y_predsfbs))
print('features names selected sfbs: \n' , sfbs.k_feature_names_,'\n')

print(classification_report(y_testsffs,y_predsffs))
print('features names selected sffs: \n' , sffs.k_feature_names_,'\n')

print(classification_report(y_testsfs,y_predsfs))
print('features names selected sfs: \n' , sfs1.k_feature_names_,'\n')

print(classification_report(y_testsbs,y_predsbs))
print('features names selected sbs : \n' , sbs.k_feature_names_,'\n')

"Area Under the roc Curves:AUC"
print('AUC')
aucrand=roc_auc_score(y_testrand,y_pred_probrand)
print('auc rand')
print(aucrand)


aucsfs=roc_auc_score(y_testsfs,y_pred_probsfs)
print('auc sfs')
print(aucsfs)


aucsbs=roc_auc_score(y_testsbs,y_pred_probsbs)
print('auc sbs')
print(aucsbs)

aucsffs=roc_auc_score(y_testsffs,y_pred_probsffs)
print('auc sffs')
print(aucsffs)



aucsfbs=roc_auc_score(y_testsfbs,y_pred_probsfbs)
print('auc sfbs')
print(aucsfbs)


"Calculate Accuracy"
accrand=accuracy_score(y_testrand,y_predrand)
print('accuracy rand')
print(accrand)

accsfs=accuracy_score(y_testsfs,y_predsfs)
print('accuracy sfs')
print(accsfs)

accsbs=accuracy_score(y_testsbs,y_predsbs)
print('accuracy sbs')
print(accsbs)

accsffs=accuracy_score(y_testsffs,y_predsffs)
print('accuracy sffs')
print(accsffs)



"Matthews Corrolation Cofficient (MCC)"

from sklearn .metrics import matthews_corrcoef
print('MCC meature : ' )
mccrand=matthews_corrcoef(y_predrand,y_testrand)
mccsfs=matthews_corrcoef(y_predsfs,y_testsfs)
mccsbs=matthews_corrcoef(y_predsbs,y_testsbs)
mccsffs=matthews_corrcoef(y_predsffs,y_testsffs)


print('mcc rand')
print(mccrand)


print('mcc sfs')
print(mccsfs)

print('mcc sbs')
print(mccsbs)

print('mcc sffs')
print(mccsffs)



"Confusion Matrix"

"Classification Report"
print('Classification Report')

print(classification_report(y_testrand,y_predrand))

print(classification_report(y_testsfs,y_predsfs))

print(classification_report(y_testsbs,y_predsbs))
print(classification_report(y_testsffs,y_predsffs))
***********************************************************************

"The best Hybrid Method with SHAP value plot for selected features"
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 08:24:14 2021

@author: pc8
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 12:58:41 2021

@author: pc8
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:08:24 2020

@author: Maxvel
"""

from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import scale,normalize,minmax_scale
from sklearn.metrics import accuracy_score
from time import time 
from mlxtend.feature_selection import SequentialFeatureSelector as sfs



#%%

ivfdata=pd.read_excel('F://ivffilterembedwrapperkbest.xlsx')
ivftarget1=pd.read_excel('F://ivfpretarget.xlsx')

#print("ivf data and shape  :")
#print(ivfdata)
#print(ivfdata.shape)

#print("ivf target1 :")
#print(ivftarget1.shape)

#%%

ivfdata=ivfdata.rename(columns={'AFC':'AFC','Duration of Infertility':'Duration',
'Endometrium Thickness':'Endometrium','Male Age':'MAge','Day 3 FSH':'FSH','Female Age':'FAge',
'BMI':'BMI','No of Folli':'Folli','No of Frozen Embryos Pack':'Embryos','No of Occytes Collected':'Occytes',
'No of Viable Occytes(MII)':'MII',
'Sperm Count *106':'Count','Sperm Motility%':'Motility','Sperm Morphology%':'Morphology',
'Total Gonadotropin Dose':'Gonadotropin'})


#print(ivfdata)

ivfdata_feature_names=['AFC','Duration','Endometrium','Mage','FSH','FAge',
'BMI','Folli','Embryos','Occytes',
'MII',
'Count','Motility','Morphology',
'Gonadotropin']

ivfdata_df=pd.DataFrame(ivfdata,columns=ivfdata_feature_names)

#%%


#%%
"Find missing values"

ivfdata.replace(' ',np.nan,inplace=True)


x=ivfdata
y=ivftarget1
#%%

"Test And Train"

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)






#%%

"Random Forest"
"Random Forest Classifier"
from sklearn.ensemble import RandomForestClassifier


rf_modelrand=RandomForestClassifier(n_estimators=200,max_features='sqrt', min_samples_split=5 , criterion='entropy',random_state=1234)
clf =RandomForestClassifier(n_estimators=200,max_features='sqrt', min_samples_split=5 , criterion='entropy',random_state=1234)



rf_model=RandomForestClassifier(n_estimators=200,max_features='auto', min_samples_split=5 , criterion='entropy',random_state=1234)
rf_model.fit(x_train,y_train)
y_predr=rf_model.predict(x_test)
rf_prediction=rf_model.predict_proba(x_test)
#%
"Random Forest"
y_pred_probr=rf_model.predict_proba(x_test)[:,1]
fprr,tprr, thresholdsr=roc_curve(y_test,y_pred_probr)
aucr=roc_auc_score(y_test,y_pred_probr)

#%%
# from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
# import matplotlib.pyplot as plt

# " Sequential Forward Selection "


# sfs1=sfs(clf,
#          k_features=15,
#          forward=True,
#          floating=False,
#          verbose=2,
#          scoring='accuracy',
#          cv=4)

# sfs1=sfs1.fit(x,y)





# #%%


# print("Data Frame of feature selected SFS")
# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)
# pd.set_option('display.width',None)
# pd.set_option('display.max_colwidth',-1)
# print(pd.DataFrame.from_dict(sfs1.get_metric_dict()).T)
# # pd.set_option('display.max_rows',None)
# # pd.set_option('display.max_columns',None)
# # pd.set_option('display.width',None)
# # pd.set_option('display.max_colwidth',-1)


# #%%

# plt.plot('k--')
# fig1, ax=plot_sfs(sfs1.get_metric_dict(), kind='std_dev', figsize=(6,4), color='darkgreen', bcolor='steelblue',marker='o',alpha=0.2)
# #plt.ylim([0.6,1])

# plt.title('Sequential Forward Selection (w. StdDev)')
# plt.grid()
# ax.set(facecolor = "lightgray")

# plt.show()

# print('features names selected  SFS: \n' , sfs1.k_feature_names_,'\n')


#%%

"SHAP value on Random Forest model "

import shap
from sklearn.ensemble import RandomForestRegressor
# print('MSE RF', mean_squared_error(y_test, y_predr)**(0.5))
"Local Interpretation using SHAP (for prediction at id number 1889)"
# rfm = RandomForestRegressor(n_estimators=1000, max_depth=10)
# rfm.fit(x_train, y_train)

# y_predrr = rfm.predict(x_test)


explainer = shap.TreeExplainer(rf_model)
# explainer = shap.KernelExplainer(y_predr,x_test)
# rf_sv=np.array(explainer.shap_values(x_test))
# rf_ev=np.array(explainer.expected_value)
# rf_pred=rf_sv[1,0,:].sum()+rf_ev[1]
# assert np.isclose(rf_pred,rf_model.predict_proba(x_test)[0,1])

shap_values = explainer.shap_values(x_test)
sh=np.array(explainer.shap_values(x_test))
print('shap_values',sh.shape)
# i = 1888
# shap.force_plot(explainer.expected_value, shap_values[i], features=x_train.loc[i], feature_names=x_train.columns)
plt.title("Random Forest Model")

shap.summary_plot(shap_values[0],x_test)

# shap.plots.beeswarm(shap_values)


# rf_shap_values = shap.KernelExplainer(rf_model.predict,x_test)
# shap.summary_plot(rf_shap_values, x_test)

#%%
*******************************************************************************************8
