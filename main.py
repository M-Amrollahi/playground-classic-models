import streamlit as st
from dataset import *
from model import cls_model
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

picked_data = st.radio('Dataset', ['Moon', 'Spiral', 'Circle'])

data = None
if picked_data=="Moon":
    data = f_create_moon()
elif picked_data=="Spiral":
    data = f_create_spiral()
else:
    data = f_create_circle()

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(data[:,0], data[:,1],c=data[:,2])

st.pyplot(fig=fig)


run_clicked = st.button('Run Models')
#if run_clicked == False:
#    st.stop()

np.random.shuffle(data)


#size_test = int(0.1 * data.shape[0])
#x_train, x_test, y_train, y_test = data[:size_test,:2], data[size_test:,:2], data[:size_test,-1], data[size_test:,-1]


model = cls_model()

df_result = model.f_scores(data)

size_subp = ceil(len(model)**.5)

fig, ax = plt.subplots(size_subp, size_subp, figsize=(10, 10))
fig.subplots_adjust(wspace=.7,hspace=.5)

x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)

XY = np.stack((X,Y),axis=2)
XY = XY.reshape(x.shape[0]*y.shape[0], 2)

for i, (Z) in enumerate(model.f_iter_predict_range(XY)):
    Z = Z.reshape(x.shape[0], y.shape[0])
    ax[i//size_subp, i%size_subp].contourf(X,Y,Z, cmap='bwr')
    #ax[i//size_subp, i%size_subp].set_title(f"{key} F1: {df_result.loc[key]['F1 Score']:.2f}" , fontsize=7)

st.pyplot(fig=fig)

