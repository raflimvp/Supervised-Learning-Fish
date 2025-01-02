import streamlit as st 
import pandas as pd 
import pickle
import numpy as np

# Memuat model
with open('perceptron_fish.pkl', 'rb') as f:
    model_perceptron = pickle.load(f)

with open('svm_fish.pkl', 'rb') as f:
   model_svm = pickle.load(f)

with open('random_forest_fish.pkl', 'rb') as f:
    model_random_forest = pickle.load(f)


st.title("Prediksi Spesies Ikan") 
st.markdown("Prediksi/Label adalah Output utama dari model adalah prediksi untuk variabel dependen berdasarkan variabel independen yang diberikan.") 
st.sidebar.title("Inputkan data Anda di sini")  

# Pilih model
model_choice = st.sidebar.selectbox('Pilih Model untuk Prediksi:', 
                                    ('Perceptron', 'SVM', 'random forest')) 

# Inisialisasi atau reset hasil jika model berubah
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = model_choice
    st.session_state['results'] = []
elif st.session_state['selected_model'] != model_choice:
    st.session_state['selected_model'] = model_choice
    st.session_state['results'] = []  # Hapus hasil jika model berubah
length = st.sidebar.slider("Panjang Ikan (length):", 0, 100, 0)
weight = st.sidebar.number_input('Berat Ikan (weight):', min_value=0.0)
w_l_ratio = st.sidebar.number_input('Rasio Berat ke Panjang (w_l_ratio):', min_value=0.0)

# Tombol untuk memprediksi spesies ikan
if st.sidebar.button('Prediksi Spesies'):
    features = np.array([[length, weight, w_l_ratio]])
    
    # Memilih model berdasarkan pilihan pengguna
    if model_choice == 'Perceptron':
        model = model_perceptron
        with open('label_encoder_fish_Perseptron.pkl', 'rb') as encoder_file:
            encoder = pickle.load(encoder_file)
    elif model_choice == 'SVM':
        model = model_svm
        with open('label_encoder_fish_SVM.pkl', 'rb') as encoder_file:
            encoder = pickle.load(encoder_file)
    else:
        model = model_random_forest
        with open('label_encoder_fish_forest.pkl', 'rb') as encoder_file:
            encoder = pickle.load(encoder_file)
 
    predicted_species_encoded = model.predict(features)[0]
    
    # Dekode hasil prediksi menggunakan encoder
    predicted_species = encoder.inverse_transform([predicted_species_encoded])[0]
    
    # Menyimpan hasil ke dalam session_state
    st.session_state['results'].append({
        'Length': length,
        'Weight': weight,
        'W_L_Ratio': w_l_ratio,
        'Model': model_choice,
        'Predicted Species': predicted_species
    })

# Menampilkan semua hasil prediksi dalam tabel
if st.session_state['results']:
    result_df = pd.DataFrame(st.session_state['results'])
    st.subheader('Tabel Hasil Prediksi')
    st.dataframe(result_df)
