import datatable as dt
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

