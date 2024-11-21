import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import streamlit as st
import joblib

# Memuat model yang telah disimpan
knn_model = joblib.load('knn_model.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')

# Membuat scaler baru
# Tentukan rentang yang sama seperti saat pelatihan
scaler = MinMaxScaler(feature_range=(0, 1))

# Judul aplikasi
st.title("Prediksi Klasifikasi Gagal Jantung")

# Input data oleh pengguna sesuai dengan fitur yang ada
st.subheader("Masukkan data untuk prediksi")

# Fitur input dari pengguna
time = st.number_input("Time (Survival Time in Days)", min_value=0, value=100)
serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, value=1.2)
ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=30)

# Membuat array data baru untuk prediksi
X_new = np.array([[time, serum_creatinine, ejection_fraction]])

# Menentukan nilai minimum dan maksimum dari fitur (dari data pelatihan)
# Ganti dengan nilai asli dari data pelatihan Anda
scaler.fit([
    [4, 0.5, 14],    # Contoh minimum dari data pelatihan
    [285, 9.4, 80] # Contoh maksimum dari data pelatihan
])

# Normalisasi input baru menggunakan scaler yang sama
X_new_scaled = scaler.transform(X_new)

# Tombol untuk melakukan prediksi
if st.button('Prediksi'):
    # Prediksi menggunakan KNN
    knn_pred = knn_model.predict(X_new_scaled)
    knn_result = 'Will Die' if knn_pred[0] == 1 else 'Not Gonna Die'

    # Prediksi menggunakan Logistic Regression
    lr_pred = lr_model.predict(X_new_scaled)
    lr_result = 'Will Die' if lr_pred[0] == 1 else 'Not Gonna Die'

    # Prediksi menggunakan Decision Tree
    dt_pred = dt_model.predict(X_new_scaled)
    dt_result = 'Will Die' if dt_pred[0] == 1 else 'Not Gonna Die'

    # Menampilkan hasil prediksi
    st.subheader("Hasil Prediksi:")
    st.write(f"**KNN Prediction:** {knn_result} ")
    st.write(f"**Logistic Regression Prediction:** {lr_result} ")
    st.write(f"**Decision Tree Prediction:** {dt_result} ")
    # Tampilkan hasil
    st.write("Data sebelum normalisasi:")
    st.write(X_new)

    st.write("\nData setelah normalisasi:")
    st.write(X_new_scaled)





st.title("Clustering App KMEANS & AGGLOMERATIVE CLUSTERING")
st.write("Upload a CSV file to perform clustering using K-Means and Agglomerative Clustering.")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Membaca dataset
    data = pd.read_csv(uploaded_file, sep="\t", engine="python")  # Pemisah tab # Handle karakter khusus seperti tab atau spasi
    st.write("Dataset Overview:")
    st.dataframe(data.head())

    # Tampilkan tipe data
    st.write("Dataset Column Types:")
    st.write(data.dtypes)

    # Konversi semua kolom ke numerik jika memungkinkan
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Pilih kolom numerik
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns

    if numeric_columns.empty:
        st.error("No valid numeric columns found in the dataset after processing. Please check the dataset.")
    else:
        st.success(f"Found {len(numeric_columns)} numeric columns: {', '.join(numeric_columns)}")

        # Proses hanya kolom numerik
        numeric_data = data[numeric_columns]
        st.write("Processing Numeric Data:")
        st.dataframe(numeric_data.head())

        # Mengisi nilai NaN (jika ada)
        numeric_data = numeric_data.fillna(0)

        # Scaling data
        st.write("Scaling the Dataset...")
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(numeric_data)
        scaled_data = pd.DataFrame(data_scaled, columns=numeric_columns)
        st.dataframe(scaled_data.head())

        # PCA transformation
        st.write("Performing PCA (n_components=3)...")
        pca = PCA(n_components=3, random_state=42)
        pca_transformed = pca.fit_transform(scaled_data)
        pca_data = pd.DataFrame(pca_transformed, columns=["col1", "col2", "col3"])
        st.write("PCA Transformed Data:")
        st.dataframe(pca_data.head())

        # 3D Scatter Plot
        st.write("3D Scatter Plot of PCA Components:")
        fig = go.Figure(data=[go.Scatter3d(
            x=pca_data["col1"],
            y=pca_data["col2"],
            z=pca_data["col3"],
            mode='markers',
            marker=dict(size=5, color="#682F2F", opacity=0.8)
        )])
        fig.update_layout(
            title="3D Projection of PCA Components",
            scene=dict(
                xaxis_title="col1",
                yaxis_title="col2",
                zaxis_title="col3",
            )
        )
        st.plotly_chart(fig)

    # Elbow Method for KMeans
    st.write("Determining optimal number of clusters using Elbow Method:")
    st.write("Elbow Visualization (K-Means):")
    fig, ax = plt.subplots(figsize=(12, 6))
    elbow = KElbowVisualizer(KMeans(), k=10, ax=ax, timings=False, locate_elbow=True)
    elbow.fit(pca_data)
    st.pyplot(fig)

    # K-Means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    pca_data["KMeans_Clusters"] = kmeans.fit_predict(pca_data)
    data["KMeans_Clusters"] = pca_data["KMeans_Clusters"]
    st.write("K-Means Clusters:")
    st.dataframe(data[["KMeans_Clusters"]].value_counts())

    # Agglomerative Clustering
    agg_cluster = AgglomerativeClustering(n_clusters=4)
    pca_data["Agglomerative_Clusters"] = agg_cluster.fit_predict(pca_data)
    data["Agglomerative_Clusters"] = pca_data["Agglomerative_Clusters"]
    st.write("Agglomerative Clustering Clusters:")
    st.dataframe(data[["Agglomerative_Clusters"]].value_counts())

    # Visualize K-Means Clusters
    st.write("K-Means Cluster Distribution:")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x=data["KMeans_Clusters"], ax=ax, palette="viridis")
    ax.set_title("K-Means Clusters")
    st.pyplot(fig)

    # Visualize Agglomerative Clusters
    st.write("Agglomerative Clustering Distribution:")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x=data["Agglomerative_Clusters"], ax=ax, palette="plasma")
    ax.set_title("Agglomerative Clusters")
    st.pyplot(fig)

        # Silhouette Score for K-Means
    kmeans_silhouette = silhouette_score(pca_data[["col1", "col2", "col3"]], pca_data["KMeans_Clusters"])
    st.write(f"Silhouette Score (K-Means): {kmeans_silhouette:.3f}")

    # Silhouette Score for Agglomerative Clustering
    agglomerative_silhouette = silhouette_score(pca_data[["col1", "col2", "col3"]], pca_data["Agglomerative_Clusters"])
    st.write(f"Silhouette Score (Agglomerative Clustering): {agglomerative_silhouette:.3f}")

    from sklearn.metrics import davies_bouldin_score

# Davies-Bouldin Index for K-Means
    kmeans_db_index = davies_bouldin_score(pca_data[["col1", "col2", "col3"]], pca_data["KMeans_Clusters"])
    st.write(f"Davies-Bouldin Index (K-Means): {kmeans_db_index:.3f}")

    # Davies-Bouldin Index for Agglomerative Clustering
    agglomerative_db_index = davies_bouldin_score(pca_data[["col1", "col2", "col3"]], pca_data["Agglomerative_Clusters"])
    st.write(f"Davies-Bouldin Index (Agglomerative Clustering): {agglomerative_db_index:.3f}")


    # Silhouette Plot for K-Means
    fig, ax = plt.subplots(figsize=(10, 6))
    sample_silhouette_values = silhouette_samples(pca_data[["col1", "col2", "col3"]], pca_data["KMeans_Clusters"])
    y_lower = 10
    for i in range(4):  # 4 clusters
        ith_cluster_silhouette_values = sample_silhouette_values[pca_data["KMeans_Clusters"] == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / 4)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # 10 for spacing

    ax.set_title("Silhouette Plot for K-Means")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster")
    st.pyplot(fig)

    # Create "Spent" column by summing specific columns
    if {"MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"}.issubset(data.columns):
        data["Spent"] = (
            data["MntWines"] +
            data["MntFruits"] +
            data["MntMeatProducts"] +
            data["MntFishProducts"] +
            data["MntSweetProducts"] +
            data["MntGoldProds"]
        )
        st.success("Column 'Spent' successfully created as the sum of relevant columns.")
    else:
        st.warning("One or more required columns ('MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds') are missing. 'Spent' column cannot be created.")

    # Visualizing Clusters on the basis of "Income" and "Spending" (K-Means)
    if "Income" in data.columns and "Spent" in data.columns:
        st.write("Cluster Characteristics on Income and Spending (K-Means):")
        fig, axes = plt.subplots(figsize=(20, 8))
        sns.scatterplot(
            x=data["Spent"],
            y=data["Income"],
            hue=data["KMeans_Clusters"],
            palette=["#B9C0C9", "#682F2F", "#9F8A78", "#F3AB60"],
            sizes=60,
            alpha=1,
            edgecolor="#1c1c1c",
            linewidth=1
        )
        axes.set_title("\nIncome-Spending Basis Clustering Profile (K-Means)\n", fontsize=25)
        axes.set_ylabel("Income", fontsize=20)
        axes.set_xlabel("Spending", fontsize=20)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)

        # Visualizing Clusters on the basis of "Income" and "Spending" (Agglomerative Clustering)
        st.write("Cluster Characteristics on Income and Spending (Agglomerative Clustering):")
        fig, axes = plt.subplots(figsize=(20, 8))
        sns.scatterplot(
            x=data["Spent"],
            y=data["Income"],
            hue=data["Agglomerative_Clusters"],
            palette=["#B9C0C9", "#682F2F", "#9F8A78", "#F3AB60"],
            sizes=60,
            alpha=1,
            edgecolor="#1c1c1c",
            linewidth=1
        )
        axes.set_title("\nIncome-Spending Basis Clustering Profile (Agglomerative Clustering)\n", fontsize=25)
        axes.set_ylabel("Income", fontsize=20)
        axes.set_xlabel("Spending", fontsize=20)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
    else:
        st.warning("Columns 'Income' and 'Spent' are required for this visualization.")

    # Spending-Based Clustering Profile (K-Means)
    st.write("### Spending-Based Clustering Profile (K-Means)")
    if "Spent" in data.columns and "KMeans_Clusters" in data.columns:
        fig, axes = plt.subplots(figsize=(30, 10))
        sns.boxenplot(
            x=data["KMeans_Clusters"],
            y=data["Spent"],
            palette=["#B9C0C9", "#682F2F", "#9F8A78", "#F3AB60"]
        )
        # Optionally, you can add swarmplot for detailed data points:
        # sns.swarmplot(x=data["KMeans_Clusters"], y=data["Spent"], color="#B9C0C9", marker="o", size=10, alpha=0.6, linewidth=0, edgecolor="white")

        axes.set_title("\nSpending Based Clustering Profile (K-Means)\n", fontsize=25)
        axes.set_ylabel("Spending", fontsize=20)
        axes.set_xlabel("\nCluster", fontsize=20)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
    else:
        st.warning("Columns 'Spent' and 'KMeans_Clusters' are required for this visualization.")

    # Spending-Based Clustering Profile (Agglomerative Clustering)
    st.write("### Spending-Based Clustering Profile (Agglomerative Clustering)")
    if "Spent" in data.columns and "Agglomerative_Clusters" in data.columns:
        fig, axes = plt.subplots(figsize=(30, 10))
        sns.boxenplot(
            x=data["Agglomerative_Clusters"],
            y=data["Spent"],
            palette=["#B9C0C9", "#682F2F", "#9F8A78", "#F3AB60"]
        )
        # Optionally, you can add swarmplot for detailed data points:
        # sns.swarmplot(x=data["Agglomerative_Clusters"], y=data["Spent"], color="#B9C0C9", marker="o", size=10, alpha=0.6, linewidth=0, edgecolor="white")

        axes.set_title("\nSpending Based Clustering Profile (Agglomerative Clustering)\n", fontsize=25)
        axes.set_ylabel("Spending", fontsize=20)
        axes.set_xlabel("\nCluster", fontsize=20)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
    else:
        st.warning("Columns 'Spent' and 'Agglomerative_Clusters' are required for this visualization.")

else:
    st.info("Please upload a CSV file to proceed.")