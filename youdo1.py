import numpy as np
import streamlit as st
from sklearn.datasets import fetch_california_housing
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

cal_housing = fetch_california_housing()
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
y = cal_housing.target
st.dataframe(X)
df = pd.DataFrame(
    dict(MedInc=X['MedInc'], Price=cal_housing.target))
x = np.array(df["MedInc"])

st.write("I wanted to L-infinity norm because only the largest element has any effect. Hence, we can minimize the maximum error. I changed the L2 normalitaziton to L-infinity.")
st.write("Having infinity power is not possible so I used 100 for the power.")



st.latex(
r"L(\beta_0, \beta_1) = \sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i))^2 } + \lambda (\beta_0^2 + \beta_1^2)")

lam1 = st.slider("Regularization Multiplier for L2 (lambda)", 0.001, 10., value=0.1)
loss, b0, b1 = [], [], []

for i, _b0 in enumerate(np.linspace(-100, 100, 50)):
    if i == 30:
        for _b1 in np.linspace(-100, 100, 50):
            b0.append(_b0)
            b1.append(_b1)

            loss.append(np.power((y - _b1 * x - _b0), 2).sum() + lam1*(_b0**100+_b1**100))

l = pd.DataFrame(dict(b0=b0, b1=b1, loss=loss))

fig = px.scatter(l, x="b1", y="loss")
st.plotly_chart(fig, use_container_width=True)

for i, _b0 in enumerate(np.linspace(-100, 100, 500)):
    if i == 30:
        for _b1 in np.linspace(-100, 100, 500):
            b0.append(_b0)
            b1.append(_b1)

            loss.append(np.power((y - _b1 * x - _b0), 2).sum() + lam1*(_b0**100+_b1**100))

l = pd.DataFrame(dict(b0=b0, b1=b1, loss=loss))

fig = px.scatter(l, x="b1", y="loss")
st.plotly_chart(fig, use_container_width=True)

st.write("There are two charts to show the convexity of the loss function. In the first one I divided -100,100 interval to 50 part and understanding the convexity is hard here. Therefore, I also divided this interval to 500 parts and draw a chart.")

def model2(x, y, lam, alpha=0.000001) -> np.ndarray:
    print("starting sgd")
    beta = np.random.random(2)

    for i in range(10000):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        g_b0 = -2 * (y - y_pred).sum() + 100 * lam * (beta[0])**99
        g_b1 = -2 * (x * (y - y_pred)).sum() + 100 * lam * (beta[1])**99

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < 0.000001:
            print(f"I do early stoping at iteration {i}")
            break

    return beta

beta2 = model2(x, y, lam1)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='data points'))
fig.add_trace(go.Scatter(x=x, y=beta2[0] + beta2[1] * x, mode='lines', name='regression + L2'))

st.plotly_chart(fig, use_container_width=True)
st.write("We do session model and graph")
image = Image.open('newplot.png')
st.image(image, caption='We do session')