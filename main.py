import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a * np.log(x + b) + c


def edit_pd(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df.columns = ['X', 'Y']
    xData = df['X']
    yData = df['Y']

    return xData, yData


def model(xData, yData):
    # max değerini bulur

    # başlangıç parametreleri olmazsa hata verebiliyor
    initialParameters = np.array([1.0, 1.0, 1.0])
    # verilen data ve başlangıç değerleri için parametre uydurmaya yarar
    fittedParameters, pcov = curve_fit(func, xData, yData, initialParameters, maxfev=5000)

    return fittedParameters


def plot_graph(xData, yData, modelPredictions):
    # Çizim için ayar
    fig = px.line(labels={'x': 'X Data', 'y': 'Y Data'}, title='Grafik')

    # Mevcut data
    fig.add_trace(go.Scatter(x=xData, y=yData, mode='lines', name='Data'))
    fig.add_trace(go.Scatter(x=xData, y=modelPredictions, mode='lines', name='Model'))

    # Grafik ekran ayarı
    fig.update_layout(width=1024, height=720)

    # ekrana çizdirme
    st.plotly_chart(fig)
    return xData, yData


if __name__ == '__main__':
    # Streamlit uygulama başlığı
    st.title("Curve fit")
    uploaded_file = st.file_uploader("Excel dosyanızı yükleyin", type=["xlsx", "xls"])
    # Dosya yükleme bileşeni

    if uploaded_file is not None:
        xData, yData = edit_pd(uploaded_file)
        xData_copy = xData.copy()
        yData_copy = yData.copy()

        set_n = xData_copy.min()
        set_c = yData_copy.max()
        if xData is not None and yData is not None:
            set_n = st.select_slider(
                'Seçilen Nokta:',
                options=xData_copy,
                value=xData_copy[xData_copy[xData_copy == set_n].index[0]])
            set_c = st.select_slider(
                'Seçilen Nokta:',
                options=yData_copy,
                value=yData_copy[yData_copy[yData_copy == set_c].index[0]])
            # selected_area = xData[xData[xData == set_n].index[0]:(xData[yData == set_c].index[0]) + 1]
            try:
                fittedParameters = model(
                    xData_copy[xData_copy[xData_copy == set_n].index[0]:yData_copy[yData_copy == set_c].index[0] + 1],
                    yData_copy[xData_copy[xData_copy == set_n].index[0]:yData_copy[yData_copy == set_c].index[0] + 1])
                model_predictions = func(xData, *fittedParameters)
                plot_graph(xData, yData, model_predictions)

            except Exception as e:
                st.write(f"Hata oluştu: {e}")
