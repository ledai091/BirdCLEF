import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# read function
def read_data():
    df = pd.read_csv('data/birdclef-2023/train_metadata.csv')
    df['filepath'] = df['filename'].apply(lambda x: 'data/birdclef-2023/train_audio/' + x)
    return df

# balance data
def balance_df(df):
    sample_count = max(df.primary_label.value_counts())
    
    balanced_df = []
    augmentation_proba = {}
    
    for i, label in enumerate(df.primary_label.unique()):
        selected_ids = np.random.choice(df[df['primary_label'] == label].index, sample_count)
        balanced_df.append(df.loc[selected_ids])
        augmentation_proba[label] = 1 - (len(df[df['primary_label'] == label]) / sample_count)
    balanced_df = pd.concat(balanced_df)
    return balanced_df, augmentation_proba


# balance test
def balance_test(df):
    sample_count = 10
    
    balanced_df = []
    augmentation_proba = {}
    
    for i, label in enumerate(df.primary_label.unique()):
        selected_ids = np.random.choice(df[df['primary_label'] == label].index, sample_count) \
            if len(df[df['primary_label'] == label]) < sample_count \
            else df[df['primary_label'] == label].index
        balanced_df.append(df.loc[selected_ids])
        augmentation_proba[label] = 1 - (len(df[df['primary_label'] == label]) / sample__count) \
            if len(df[df['primary_label'] == label]) < sample__count else 0
    balanced_df = pd.concat(balanced_df, axis=0)
    
    return balanced_df, augmentation_proba


def stratified_train_test_split(df, test_size=0.3):
    train_data, test_data = [], []
    
    for i, label in enumerate(df.primary_label.unique()):
        if len(df[df.primary_label == label]) == 1:
            train = df[df.primary_label == label]
            test = df[df.primary_label == label]
        else:
            train, test = train_test_split(df[df.primary_label == label], test_size=test_size)
        train_data.append(train)
        test_data.append(test)
        
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)
    
    return train_data, test_data