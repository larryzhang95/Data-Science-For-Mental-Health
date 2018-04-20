import feature_embeddings
from model import build_svm_model, getAllData,getEmbedding, saveToPickle, getFromPickle
import pandas as pd
"""
Usage Requirements:
- Must have own label set
- Must have audio directory with wavFiles to extract data from
- Must have pca_param_np,ckpt file available
"""
audioDir = './Bipolar/Bipolar_Disorder'
#csv_file = './Bipolar_data.csv'

def main():
    label = abNormalData()
    #ppc_batch = getAllData(audioDir)
    #saveToPickle(ppc_batch,'data.p')
    ppc_batch = getFromPickle('data.p')  #Used to save embedding data to pickle file so you dont have to rerun getAllData
    data_embedded = getEmbedding(ppc_batch,1)
    accuracy = build_svm_model(data_embedded,label)

#Bipolar Labelling Function(s)
def getBipolarLabels(csv):
    label_df = pd.read_csv(csv)
    label_df = label_df[['Notes','Symptoms']]

    label_df.Symptoms.str.lower()
    label_df['Type'] = label_df.apply(bipolarCase,axis=1)
    label_df = label_df.drop(['Notes'],axis=1)
    return label_df
def bipolarCase(row):
    if "II" in row.Notes:
        return 'BipolarII'
    else:
        return 'BipolarI'
#get usable Y value
def getY(csv):
    df = getBipolarLabels(csv)
    df = df.drop(['Type'],axis=1)
    df['label'] = df.apply(labelFunc,axis=1)
    df = df.drop(['Symptoms'], axis=1)
    df = df.drop([18])
    return df

def labelFunc(row):
    if row.Symptoms == 'neutral':
        return 0
    elif row.Symptoms == 'depressed':
        return 1
    else:
        return 2
def abNormalData():
    df = getY('./Bipolar/Bipolar_data.csv')
    df = df.replace(to_replace=2,value=1)
    return df
def depVsManic(df):
    df = getY('Bipolar_data.csv')
    return df[df.label != 0]
#Bipolar Labelling Function(s)

if __name__ == "__main__":
    main()
