import datatable as dt
import numpy as np

def split_sequences(sequences, n_steps, limits = None):
    if sequences.shape[0] == n_steps:
        return np.array([sequences])
    X = list()
    ## Add on rows of zeros at start
    #to_add = np.zeros((n_steps-1, sequences[0].size))
    #sequences = np.vstack((to_add,sequences))
    if limits == None:
        limits = [0, len(sequences)]
    for i in range(limits[0],limits[1]):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x = sequences[i:end_ix, :]
        X.append(seq_x)
    #print(sequences)
    #print(limits)
    #print(np.array(X))
    return np.array(X)

class Data:
    def __init__(self, df):
        '''
        Class for loading the data frame.
        Arguments:
            df: dataframe
        '''
        self.df = df
        self.nans = {}
        if df.isnull().values.any():
            for col in [x for x in self.df.columns if 'feature' in x]:
                self.nans[col] = self.df[col].median()
                self.df[col] = self.df[col].replace(np.NaN, self.nans[col])
        self.train_x, self.train_y, self.test_x, self.test_y, self.train_weight, self.test_weight = self.train_test()
        del self.df

    @classmethod
    def from_csv(cls, short=False, path='input/train.csv'):
        '''
        Class method for loading directly from csv
        Arguments:
            short: reduce the size of the data for testing purposes. Removes most of the rows and most of the features.
            path:  path to input csv file.
        '''
        if short:
            df = dt.fread(path,max_nrows=500,fill=True).to_pandas()
            #df.drop(['feature_'+str(i) for i in range(2,130)]+['resp_'+str(i) for i in range(1,5)],axis=1, inplace=True)
        else:
            df = dt.fread(path,fill=True).to_pandas()
        return cls(df)

    @classmethod
    def for_kaggle_predict(cls, df, fill_nans):
        '''
        Static method for just getting sequenced data from an input dataframe.
        Arguments:
            df: dataframe
            fill_nans: dictionary with value to fill nans for each column
        '''
        #for col in df.columns:
        #    if 'feature' not in col: continue
        #    df[col] = df[col].replace(np.NaN, fill_nans[col])
        df.fillna(fill_nans,inplace=True)    
        dat = cls(df)
        dat.train_x = np.vstack((dat.train_x,dat.test_x))
        return dat 


    def n_features(self):
        #features = [x for x in self.df.columns if 'feature' in x]
        #return len(features)
        return len(self.train_x[0])

    def features_as_np(self):
        return np.array([self.df[x].values.astype(float) for x in self.df.columns if 'feature' in x]).transpose()

    def split_sequences(self, seq_len):
        raise RuntimeError('Obsolete function. Splitting the entire dataframe rather than just a batch is very wasteful of RAM.')
        newdf = self.df[[x for x in self.df.columns if 'feature' in x] + ['resp']]
        vals = newdf.values.astype(float)
        return split_sequences(vals, seq_len)

    def train_test(self, train_frac = 0.7):
        train_n = int(train_frac*len(self.df))
        test_n  = len(self.df) - train_n

        features = self.features_as_np()
        train_x = features[:train_n,:]
        test_x  = features[train_n:,:]

        if 'resp' in self.df.columns:
            y = self.df['resp'].values.astype(float)
            train_y = y[:train_n]
            test_y  = y[train_n:]
        else:
            train_y = []
            test_y = []

        weight = self.df['weight'].values.astype(float)
        train_weight = weight[:train_n]
        test_weight  = weight[train_n:]

        return train_x, train_y, test_x, test_y, train_weight, test_weight

def find_nans():
    x = Data(short=False)
    print (x.train_x.size)
    print (np.sum(np.isnan(x.train_x)))
    exit(1)
    if np.isnan(x.train_x).any():
        print('nan found in features')
        for i in np.argwhere(np.isnan(x.train_x)):
            print (i)

if __name__ == '__main__':
    find_nans()
    exit(1)
    x = Data(short=True)
    #print(x.df)
    #x.add_zero_rows(5)
    #print(x.df)
    #exit(1)
    #print(split_sequences(x.features_as_np(),4))
    arr = np.array(range(100))
    arr = arr.reshape(10,10)
    print(arr)
    print(split_sequences(arr,3))
    print(split_sequences(arr,3,[2,6]))
    exit(1)
    x = Data()
    print(x.df)
    #print('===============')
    #print (x.features_as_np())
    print('==================')
    for n,i in enumerate(x.train_test()):
        #if n in [0,1]: continue
        print('\n')
        print(i)
        if n in [0,2]: 
            print ([len(j) for j in i])
        else: 
            print (len(i))

'''
df = dt.fread('input/train.csv').to_pandas()
#print('\n'.join(list(df.columns)))
#print(df.describe())
#print(df.head())


keep_features = ['feature_'+str(i) for i in [60,61,65,66]]
#df = df[keep_features]
#print(df.head())


import itertools
import matplotlib.pyplot as plt
plots = itertools.combinations(keep_features, 2)
for plot in plots:
    plt.scatter(df[plot[0]], df[plot[1]])
    plt.savefig(plot[0]+'_'+ plot[1]+'.png')
    plt.clf()
'''
