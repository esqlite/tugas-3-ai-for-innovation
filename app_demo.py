import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

# Judul Dashboard
st.title("Dashboard Prediksi Permohonan Informasi")

# Penjelasan singkat mengenai proyek
st.markdown("""
Dashboard ini dirancang untuk memprediksi dan menganalisis kategori permohonan informasi publik berdasarkan data historis. 
Menggunakan berbagai model machine learning, dashboard ini dapat memberikan wawasan mengenai pola permohonan di masa depan, 
serta memberikan analisis visual terkait tren permohonan dan waktu penyelesaian.

Anda bisa mengunggah dataset Anda sendiri atau menggunakan data sampel yang tersedia untuk eksplorasi lebih lanjut.
            
Tugas Mata Kuliah: AI for Innovation
""")

# Bagian Upload Data
st.sidebar.header("Upload Dataset Anda")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

# Tambahkan link ke data sampel di bawah uploader
st.sidebar.markdown('[Unduh sample data](http://unpad7.siat-dev.com/tiket_permohonan_with_dates_120_rows_prep.csv)')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Konversi kolom `status` ke kategori 'closed' atau 'open'
    data['status'] = data['status'].apply(lambda x: 'closed' if x == 2 else 'open')

    # Konversi kolom `churn` dari 'Yes'/'No' menjadi 1/0
    data['churn'] = data['churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Konversi kolom tanggal
    data['created_at'] = pd.to_datetime(data['created_at'])
    data['updated_at'] = pd.to_datetime(data['updated_at'])
    
    st.write("Data yang diupload:")
    st.write(data)

    # Memastikan kolom yang diperlukan ada
    required_columns = ['applicant_category', 'ticket_category', 'completion_time', 'churn', 'created_at', 'updated_at', 'status']
    if not all(col in data.columns for col in required_columns):
        st.error(f"File CSV harus mengandung kolom: {', '.join(required_columns)}")
    else:
        # Pilih fitur dan target
        features = st.sidebar.multiselect("Pilih Fitur", [col for col in data.columns if col not in []])
        target = st.sidebar.selectbox("Pilih Target", [col for col in data.columns if col != 'ticket_id' and col != 'full_name'])

        if features and target:
            if target in features:
                st.error("Kolom target tidak boleh ada dalam fitur.")
            else:
                # Pilih Tahun
                st.sidebar.header("Pilih Tahun")
                available_years = ["Semua tahun"] + sorted(data['created_at'].dt.year.unique().tolist(), reverse=True)
                selected_year = st.sidebar.selectbox("Pilih Tahun", available_years)

                # Filter data berdasarkan tahun
                if selected_year != "Semua tahun":
                    data = data[data['created_at'].dt.year == int(selected_year)]

                # Normalisasi fitur
                le = LabelEncoder()
                for col in features:
                    if data[col].dtype == 'object':
                        data[col] = le.fit_transform(data[col])
                
                scaler = StandardScaler()
                data[features] = scaler.fit_transform(data[features])

                # Hitung waktu penyelesaian dalam jam
                data['completion_time'] = (data['updated_at'] - data['created_at']).dt.total_seconds() / 3600

                # Split data
                X = data[features]
                y = data[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Memilih Model
                st.sidebar.header("Pilih Model")
                model_choice = st.sidebar.selectbox("Pilih Model", [
                    "Random Forest", 
                    "SVM", 
                    "Logistic Regression", 
                    "K-Means", 
                    "Decision Tree", 
                    "Gradient Boosting", 
                    "Naive Bayes", 
                    "Neural Networks"
                ])

                if model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100)
                elif model_choice == "SVM":
                    model = SVC()
                elif model_choice == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif model_choice == "K-Means":
                    model = KMeans(n_clusters=2)
                elif model_choice == "Decision Tree":
                    model = DecisionTreeClassifier()
                elif model_choice == "Gradient Boosting":
                    model = GradientBoostingClassifier(n_estimators=100)
                elif model_choice == "Naive Bayes":
                    model = GaussianNB()
                elif model_choice == "Neural Networks":
                    model = MLPClassifier(max_iter=2000)

                if model_choice in ["Random Forest", "SVM", "Logistic Regression", "Decision Tree", "Gradient Boosting", "Naive Bayes", "Neural Networks"]:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    precision = precision_score(y_test, y_pred, average='weighted')

                    # Menghitung metrik evaluasi setelah model memprediksi
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    precision = precision_score(y_test, y_pred, average='weighted')

                    # Menampilkan Metrik Evaluasi dalam Format Grid
                    col1, col2, col3, col4 = st.columns(4)

                    # Akurasi
                    with col1:
                        st.markdown("**Akurasi Model**")
                        st.write(f"{accuracy:.2f}")

                    # F1 Score
                    with col2:
                        st.markdown("**F1 Score**")
                        st.write(f"{f1:.2f}")

                    # Recall
                    with col3:
                        st.markdown("**Recall**")
                        st.write(f"{recall:.2f}")

                    # Precision
                    with col4:
                        st.markdown("**Precision**")
                        st.write(f"{precision:.2f}")

                    # Interpretasi Hasil
                    st.markdown("**Interpretasi Dinamis Hasil Metrik Evaluasi**")

                    if accuracy > 0.9:
                        st.write("- **Akurasi Model Sangat Tinggi**: Model ini memiliki akurasi yang sangat baik, menunjukkan bahwa hampir semua prediksi yang dibuat oleh model benar.")
                    elif accuracy > 0.75:
                        st.write("- **Akurasi Model Baik**: Model ini memiliki akurasi yang cukup baik, namun masih ada beberapa prediksi yang salah.")
                    else:
                        st.write("- **Akurasi Model Rendah**: Akurasi model ini rendah, menunjukkan banyak prediksi yang salah. Model mungkin memerlukan penyempurnaan lebih lanjut.")

                    if f1 > 0.9:
                        st.write("- **F1 Score Sangat Tinggi**: Model memiliki keseimbangan yang baik antara precision dan recall, sangat cocok untuk data dengan distribusi kelas tidak seimbang.")
                    elif f1 > 0.75:
                        st.write("- **F1 Score Baik**: Model memiliki keseimbangan yang cukup baik antara precision dan recall, namun bisa ditingkatkan lebih lanjut.")
                    else:
                        st.write("- **F1 Score Rendah**: Model mungkin memiliki trade-off yang besar antara precision dan recall, sebaiknya evaluasi ulang kebutuhan bisnis.")

                    if recall > 0.9:
                        st.write("- **Recall Tinggi**: Model berhasil menangkap hampir semua kasus positif dengan baik.")
                    elif recall > 0.75:
                        st.write("- **Recall Cukup Baik**: Model mampu menangkap sebagian besar kasus positif, namun mungkin masih ada yang terlewat.")
                    else:
                        st.write("- **Recall Rendah**: Model mungkin melewatkan banyak kasus positif, sehingga perlu perbaikan untuk meningkatkan recall.")

                    if precision > 0.9:
                        st.write("- **Precision Tinggi**: Model menghasilkan prediksi positif yang sangat tepat.")
                    elif precision > 0.75:
                        st.write("- **Precision Cukup Baik**: Model cukup baik dalam menghasilkan prediksi positif yang akurat.")
                    else:
                        st.write("- **Precision Rendah**: Model sering membuat kesalahan pada prediksi positif. Mungkin perlu dilakukan penyempurnaan untuk mengurangi positif palsu.")

                    # Rekomendasi berdasarkan kombinasi metrik
                    if accuracy > 0.75 and f1 > 0.75:
                        st.write("Secara keseluruhan, model ini memiliki performa yang baik, cocok untuk diterapkan dalam kasus ini.")
                    elif recall < 0.5 or precision < 0.5:
                        st.write("Model ini mungkin kurang optimal dalam menangkap semua kasus atau memberikan prediksi yang akurat. Pertimbangkan untuk mengatur ulang model atau menambah data pelatihan.")
                    else:
                        st.write("Model ini dapat diterima namun sebaiknya terus dipantau dan disempurnakan agar performanya semakin baik.")


                    # Prediksi permohonan tahun depan
                    future_predictions = model.predict(X)
                    data['predicted_category'] = future_predictions

                    # Cari kategori applicant dan ticket yang paling sering muncul dari hasil prediksi
                    most_common_applicant_category = data.loc[data['predicted_category'] == data['predicted_category'].mode()[0], 'applicant_category'].mode()[0]
                    most_common_ticket_category = data.loc[data['predicted_category'] == data['predicted_category'].mode()[0], 'ticket_category'].mode()[0]

                    st.write(f"Prediksi untuk tahun depan menunjukkan bahwa kategori permohonan terbanyak diperkirakan akan datang dari `applicant_category` dengan ID: '{most_common_applicant_category}' dan `ticket_category` dengan ID: '{most_common_ticket_category}'.")

                elif model_choice == "K-Means":
                    model.fit(X)
                    labels = model.labels_
                    st.write("Cluster yang ditentukan:")
                    st.write(labels)

                # Visualisasi Rata-rata Waktu Penyelesaian per Tahun dengan Anotasi
                st.subheader("Rata-rata Waktu Penyelesaian per Tahun")
                data['year'] = data['created_at'].dt.year
                avg_completion_per_year = data[data['status'] == 'closed'].groupby('year')['completion_time'].mean().reset_index()

                if not avg_completion_per_year.empty:
                    fig, ax = plt.subplots()
                    bars = ax.bar(avg_completion_per_year['year'].astype(str), avg_completion_per_year['completion_time'], color='skyblue')
                    ax.set_xlabel("Tahun")
                    ax.set_ylabel("Rata-rata Waktu Penyelesaian (jam)")
                    
                    # Menambahkan label nilai di atas setiap batang
                    for bar in bars:
                        yval = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.1f} jam', ha='center', va='bottom')

                    st.pyplot(fig)


                # Visualisasi Jumlah Permohonan per Kategori per Tahun
                st.subheader("Jumlah Permohonan per Kategori per Tahun")
                ticket_analysis = data.groupby(['year', 'ticket_category']).size().reset_index(name='count')
                if not ticket_analysis.empty:
                    fig = px.bar(ticket_analysis, x='year', y='count', color='ticket_category', text='count')
                    fig.update_traces(texttemplate='%{text}', textposition='outside')
                    st.plotly_chart(fig)

                # Visualisasi Jumlah Permohonan per Kategori per Bulan
                st.subheader("Analisis Jumlah Permohonan Per Bulan")
                data['month'] = data['created_at'].dt.to_period('M').astype(str)
                monthly_analysis = data.groupby(['month', 'ticket_category']).size().reset_index(name='count')
                if not monthly_analysis.empty:
                    monthly_fig = px.bar(monthly_analysis, x='month', y='count', color='ticket_category', text='count')
                    monthly_fig.update_traces(texttemplate='%{text}', textposition='outside')
                    st.plotly_chart(monthly_fig)

                # Visualisasi Distribusi Churn
                st.subheader("Distribusi Churn")
                churn_counts = data['churn'].value_counts().reset_index()
                churn_counts.columns = ['churn', 'count']
                if not churn_counts.empty:
                    fig, ax = plt.subplots()
                    ax.bar(churn_counts['churn'].astype(str), churn_counts['count'], color='skyblue')
                    ax.set_xlabel("Churn (0 = No, 1 = Yes)")
                    ax.set_ylabel("Jumlah")
                    st.pyplot(fig)
