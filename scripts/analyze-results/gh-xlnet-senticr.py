# This file is used to compare the predictions from XLNet and SentiCR
# on GitHub dataset
# Created by happygirlzt

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

data_folder=Path('your_github_predictions_folder')

### Concatenate predictions from XLNet and SentiCR into one dataframe
#xlnet = pd.read_csv(data_folder/'XLNet_github_predictions.csv')
#print(xlnet.shape)

#cr = pd.read_csv(data_folder/'senticr_github_predictions.csv')
#print(cr.head())
#print(cr.shape)


# final_df=pd.DataFrame(columns=['Text', 'True_label', 'XLNet_predicted', 'SentiCR_predicted'])
# final_df['Text'] = pd.Series(xlnet['Text'])
# final_df['True_label']=pd.Series(xlnet['True_label'])
# final_df['xlnet_predicted']=pd.Series(xlnet['xlnet_predicted'])
# final_df['SentiCR_predicted']=pd.Series(cr['SentiCR_predicted'])
#print(final_df.head())

#final_df.to_csv(data_folder/'xlnet-senticr.csv', header=True)

### Read the df
final_df=pd.read_csv(data_folder/'xlnet-senticr.csv')
# xlnet true predictions
xlnet_true = final_df.loc[final_df['True_label'] == final_df['XLNet_predicted']]
print(xlnet_true.head())

# xlnet true, while cr false
# xlnet true predictions
xlnet_true_cr_false = final_df.loc[
    (final_df['True_label'] == final_df['XLNet_predicted']) & 
    (final_df['True_label'] != final_df['SentiCR_predicted'])
    ]

#print(xlnet_true_cr_false.head())
print(xlnet_true_cr_false.shape)
print(xlnet_true.shape)

# cr true predictions
cr_true = final_df.loc[
    final_df['True_label'] == final_df['SentiCR_predicted']
    ]
print(cr_true.head())

# xlnet false, while cr true
cr_true_xlnet_false=final_df.loc[
    (final_df['True_label'] != final_df['XLNet_predicted']) &
    (final_df['True_label'] == final_df['SentiCR_predicted'])
    ]

print(cr_true_xlnet_false.shape[0])
print(cr_true.shape[0])

# both true
both_true=final_df.loc[
    (final_df['True_label'] == final_df['SentiCR_predicted']) & 
    (final_df['True_label'] == final_df['XLNet_predicted'])
    ]
#both_true.head()
print('both true {}'.format(both_true.shape[0]))