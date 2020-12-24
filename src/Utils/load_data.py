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

class Data:
    def __init__(self, short=True):
        if short: self.df = df = dt.fread('input/train.csv',max_nrows=50).to_pandas()
        else: self.df = dt.fread('input/train.csv').to_pandas()
        self.df.drop(['feature_'+str(i) for i in range(2,130)]+['resp_'+str(i) for i in range(1,5)],axis=1, inplace=True)

    def n_features(self):
        features = [x for x in self.df.columns if 'feature' in x]
        return len(features)

    def features_as_np(self):
        return [self.df[x].values.astype(float) for x in self.df.columns if 'feature' in x]

    def train_test(self, train_frac = 0.7, seq_len=5):
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
