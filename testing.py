# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import cla
from percetron import Perceptron

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv ( url, header=None )
y = df.iloc[ 0:100, 4 ].values
y = np.where ( y == 'Iris-setosa', 1, -1 )
X = df.iloc[ 0:100, [ 0, 2 ] ].values
plt.scatter ( X[ :50, 0 ], X[ :50, 1 ], color='red', marker='o', label=u'щетинстый' )
plt.scatter ( X[ 50:100, 0 ], X[ 50:100, 1 ], color='blue', marker='x', label=u'разноцветный' )
""""plt.xlabel(u'длина чашелистника')
plt.ylabel(u'длина лепестка')
plt.legend(loc='upper left')
plt.show()"""
ppn = Perceptron ( eta=0.1, n_iter=10 )
ppn.fit ( X, y )
plt.plot ( range ( 1, len ( ppn.errors_ ) + 1 ), ppn.errors_, marker='o' )
# plt.xlabel ( u'Эпох' )
# plt.ylabel ( u'Число случаев ошибочной класификации' )
# plt.show ()


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap ( colors[ :len ( np.unique( y ) ) ] )
    x1_min, x1_max = X[ :, 0 ].min() - 1, X[ :, 0 ].max() + 1
    x2_min, x2_max = X[ :, 1 ].min() - 1, X[ :, 1 ].max() + 1
    xx1, xx2 = np.meshgrid ( np.arange ( x1_min, x1_max, resolution ), np.arange( x2_min, x2_max, resolution ) )
    print xx1, xx2
    Z = classifier.predict( np.array([ xx1.ravel(), xx2.ravel()]).T )
    print Z
    Z = Z.reshape ( xx1.shape )

    plt.contourf ( xx1, xx2, Z, alpha=0.4, cmap=cmap )
    plt.xlim ( xx1.min(), xx1.max() )
    plt.ylim ( xx2.min(), xx2.max() )
    for idx, cl in enumerate ( np.unique ( y ) ):
        plt.scatter ( x=X[ y == cl, 0 ], y=X[ y == cl, 1 ], alpha=0.8, c=cmap ( idx ), marker=markers[ idx ], label=cl )


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel(u'Длина чашелистника')
plt.ylabel(u'длина лепестка')
plt.legend(loc='upper left')
plt.show()

"""rrewerwrwercssdfsdfs"""
