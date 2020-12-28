import datatable as dt
import numpy as np

def create_inout_sequences(input_vector, seq_len):
    out_seq = []
    L = len(input_vector)
    for i in range(L-seq_len):
        x = input_vector[i:i+seq_len+1]
        out_seq.append(x)
    #print(input_vector)
    #print(out_seq)
    return out_seq

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

class Data:
    def __init__(self, short=False,path='input/train.csv'):
        '''
        Class for loading the data for this competition.
        Arguments:
            short: reduce the size of the data for testing purposes. Removes most of the rows and most of the features.
            path:  path to input csv file.
        '''
        if short: self.df = df = dt.fread(path,max_nrows=50).to_pandas()
        else: self.df = dt.fread(path).to_pandas()
        #print(self.df.columns)
        if short: self.df.drop(['feature_'+str(i) for i in range(2,130)]+['resp_'+str(i) for i in range(1,5)],axis=1, inplace=True)

    def n_features(self):
        features = [x for x in self.df.columns if 'feature' in x]
        return len(features)

    def features_as_np(self):
        return np.array([self.df[x].values.astype(float) for x in self.df.columns if 'feature' in x])

    def split_sequences(self, seq_len):
        raise RuntimeError('Obsolete function. Splitting the entire dataframe rather than just a batch is very wasteful of RAM.')
        newdf = self.df[[x for x in self.df.columns if 'feature' in x] + ['resp']]
        vals = newdf.values.astype(float)
        return split_sequences(vals, seq_len)

    def train_test(self, train_frac = 0.7, seq_len=5):
        train_n = int(train_frac*len(self.df))
        test_n  = len(self.df) - train_n

        features = self.features_as_np()
        y = self.df['resp'].values.astype(float)

        train_x = features[:train_n,:]
        train_y = y[:train_n]
        test_x  = features[train_n:,:]
        test_y  = y[train_n:]

        return train_x, train_y, test_x, test_y

    def train_test_old(self, train_frac = 0.7, seq_len=5):
        x = self.features_as_np()
        y = self.df.resp
        train_n = int(train_frac*len(self.df))
        test_n  = len(self.df) - train_n

        train_x = []
        test_x  = []
        for i in range(len(x)):
            x[i] = np.concatenate([np.zeros(seq_len), x[i]])
            #print(x[i])
            train_x.append(create_inout_sequences(x[i][:train_n+seq_len],seq_len))
            test_x .append(create_inout_sequences(x[i][train_n:],seq_len))

        train_y = y[:train_n].values.astype(float)
        test_y  = y[train_n:].values.astype(float)

        return train_x, train_y, test_x, test_y
        

if __name__ == '__main__':
    x = Data()
    print(x.split_sequences())
    arr = np.array(range(50))
    arr = arr.reshape(5,10)
    print(arr)
    print(split_sequences(arr,3))
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
