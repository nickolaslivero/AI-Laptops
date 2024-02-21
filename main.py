import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("laptops.csv")

data_numeric = data[['RAM', 'Storage', 'Screen', 'Final Price']]

# Imputar valores ausentes nas colunas numéricas
imputer = SimpleImputer(strategy='mean')
data_filled = imputer.fit_transform(data_numeric)

# Normalizaçao
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_filled)

# KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(data_normalized)

def suggest_laptops(price, ram, storage, screen):
    # Selecionar laptops no cluster correspondente ao preço
    cluster_label = kmeans.predict([[ram, storage, screen, price]])[0]
    suggested_laptops = data[kmeans.labels_ == cluster_label]
    return suggested_laptops

# Front-end
st.title('Sistema de Recomendação de Laptops')

min_price = int(data['Final Price'].min())
max_price = int(data['Final Price'].max())
step = 100
price = st.slider('Preço', min_value=min_price, max_value=max_price, step=step, value=(min_price + max_price) // 2)
ram = st.slider('RAM', min_value=int(data['RAM'].min()), max_value=int(data['RAM'].max()), step=1, value=int(data['RAM'].mean()))
storage = st.slider('Armazenamento', min_value=int(data['Storage'].min()), max_value=int(data['Storage'].max()), step=100, value=int(data['Storage'].mean()))
screen = st.slider('Tamanho da Tela', min_value=float(data['Screen'].min()), max_value=float(data['Screen'].max()), step=0.1, value=float(data['Screen'].mean()))

suggestions = suggest_laptops(price, ram, storage, screen)

st.subheader('Sugestões de laptops com base nos parâmetros selecionados:')
st.write(suggestions)

# Gráfico
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Final Price', y='Storage', data=data, hue=kmeans.labels_, palette='viridis')
plt.title('Clusterização de Laptops (Preço vs Armazenamento)')
plt.xlabel('Preço')
plt.ylabel('Armazenamento')
plt.legend(title='Cluster')
st.pyplot(plt)

# Grafico
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Final Price', y='RAM', data=data, hue=kmeans.labels_, palette='viridis')
plt.title('Clusterização de Laptops (Preço vs RAM)')
plt.xlabel('Preço')
plt.ylabel('RAM')
plt.legend(title='Cluster')
st.pyplot(plt)
