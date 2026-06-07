import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================

sample_df = df.sample(
    n=5000,
    random_state=42
)

sample_df.to_csv(
    "creditcard_sample.csv",
    index=False
)

# ==================================================
# LOAD MODEL
# ==================================================

model_all = joblib.load("fraud_model_all.pkl")
model_sel = joblib.load("fraud_model_selected.pkl")

# ==================================================
# FITUR
# ==================================================

all_features = [
    'Time','V1','V2','V3','V4','V5','V6','V7',
    'V8','V9','V10','V11','V12','V13','V14',
    'V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26',
    'V27','V28','Amount'
]

selected_features = [
    'V12','V14','V3','V4','V11',
    'V16','V10','V9','V7','V2'
]

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("💳 Fraud Detection")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "Home",
        "Visualisasi Dataset",
        "Prediksi Fraud",
        "Perbandingan Model",
        "Tentang Project"
    ]
)

# ==================================================
# HOME
# ==================================================

if menu == "Home":

    st.title("💳 Fraud Detection System")

    st.markdown("""
    ### Deteksi Fraud Transaksi Digital

    Project UAS Machine Learning yang merupakan
    pengembangan dari project UTS dengan menambahkan:

    - Feature Selection
    - Perbandingan Model
    - Analisis Kinerja Model
    - Deployment Model
    """)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Jumlah Data", "284,807")

    with col2:
        st.metric("Jumlah Fraud", "492")

    with col3:
        st.metric("Model A", "30 Fitur")

    with col4:
        st.metric("Model B", "10 Fitur")

    st.divider()

    st.subheader("Metode")

    st.info("""
    Algoritma yang digunakan adalah Decision Tree
    dengan teknik SMOTE dan Random Undersampling.

    UTS menggunakan seluruh fitur dataset.

    UAS menggunakan Feature Selection
    untuk memilih 10 fitur terbaik.
    """)

# ==================================================
# VISUALISASI
# ==================================================

elif menu == "Visualisasi Dataset":

    st.title("📊 Visualisasi Dataset")

    tab1, tab2, tab3 = st.tabs([
        "Distribusi Fraud",
        "Heatmap",
        "Feature Selection"
    ])

    # --------------------------------

    with tab1:

        st.subheader("Distribusi Fraud")

        fig, ax = plt.subplots(figsize=(6,4))

        sns.countplot(
            x="Class",
            data=df,
            ax=ax
        )

        ax.set_title("Distribusi Fraud")

        st.pyplot(fig)

        st.warning("""
        Dataset memiliki ketidakseimbangan kelas
        sehingga digunakan teknik SMOTE dan
        Random Undersampling.
        """)

    # --------------------------------

    with tab2:

        st.subheader("Heatmap Korelasi")

        corr = df.corr()

        fig, ax = plt.subplots(
            figsize=(12,8)
        )

        sns.heatmap(
            corr,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

    # --------------------------------

    with tab3:

        st.subheader("10 Fitur Terpilih")

        feature_df = pd.DataFrame({
            "Fitur Terpilih": selected_features
        })

        st.dataframe(
            feature_df,
            use_container_width=True
        )

        st.success("""
        Fitur dipilih berdasarkan
        korelasi absolut tertinggi
        terhadap label fraud.
        """)

# ==================================================
# PREDIKSI
# ==================================================

elif menu == "Prediksi Fraud":

    st.title("🔍 Prediksi Fraud")

    st.markdown("""
    Upload file CSV dengan format yang sama
    seperti dataset creditcard.csv.
    """)

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        data_pred = pd.read_csv(uploaded_file)

        st.subheader("Preview Data")

        st.dataframe(
            data_pred.head()
        )

        # Validasi

        missing = [
            col for col in all_features
            if col not in data_pred.columns
        ]

        if len(missing) > 0:

            st.error(
                f"Kolom tidak ditemukan: {missing}"
            )

        else:

            st.success(
                "Format file valid."
            )

            if st.button(
                "Jalankan Prediksi"
            ):

                # Model A

                pred_all = model_all.predict(
                    data_pred[all_features]
                )

                # Model B

                pred_sel = model_sel.predict(
                    data_pred[selected_features]
                )

                result = data_pred.copy()

                result["Prediksi_Model_A"] = pred_all
                result["Prediksi_Model_B"] = pred_sel

                st.subheader(
                    "Hasil Prediksi"
                )

                st.dataframe(
                    result.head(20)
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Fraud Model A",
                        int(pred_all.sum())
                    )

                with col2:

                    st.metric(
                        "Fraud Model B",
                        int(pred_sel.sum())
                    )

                csv = result.to_csv(
                    index=False
                )

                st.download_button(
                    "📥 Download Hasil",
                    csv,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv"
                )

# ==================================================
# PERBANDINGAN
# ==================================================

elif menu == "Perbandingan Model":

    st.title("⚖️ Perbandingan Model")

    comparison = pd.DataFrame({

        "Metric":[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Model A":[
            0.99,
            0.93,
            0.82,
            0.87
        ],

        "Model B":[
            0.99,
            0.95,
            0.85,
            0.89
        ]

    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.bar_chart(
        comparison.set_index(
            "Metric"
        )
    )

# ==================================================
# ABOUT
# ==================================================

elif menu == "Tentang Project":

    st.title("📚 Tentang Project")

    st.markdown("""
    ## Judul

    Deteksi Fraud Transaksi Digital
    Menggunakan Algoritma Decision Tree
    dengan Teknik SMOTE dan Undersampling

    ---

    ## Pengembangan UTS → UAS

    ### UTS
    - Menggunakan seluruh fitur
    - Decision Tree
    - SMOTE
    - Random Undersampling

    ### UAS
    - Feature Selection
    - Perbandingan Model
    - Analisis Kinerja
    - Deployment Model

    ---

    ## Feature Selection

    Jumlah fitur berhasil dikurangi
    dari 30 fitur menjadi 10 fitur.

    ---

    ## Fitur Terpilih

    - V12
    - V14
    - V3
    - V4
    - V11
    - V16
    - V10
    - V9
    - V7
    - V2
    """)