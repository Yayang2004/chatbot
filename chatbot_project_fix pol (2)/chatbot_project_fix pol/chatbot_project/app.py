from flask import Flask, render_template, request, jsonify
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, silhouette_score
import plotext as plt  # Mengganti matplotlib dengan plotext

# Unduh resource NLTK jika belum ada
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Membaca file JSON dari path yang diberikan
dataset = pd.read_json(r"D:\SEMESTER 4\AI\chatbot_project_fix pol (2)\chatbot_project_fix pol\chatbot_project\data\intents.json")

# Menampilkan beberapa baris dataset untuk memastikan data terbaca dengan benar
print(dataset.head())

# Flatten the dataset to work with intents and responses
# Mengambil data intents dan text dari file JSON dan merubahnya menjadi DataFrame yang sesuai
intents = []
patterns = []
responses = []

for intent in dataset['intents']:
    for pattern in intent['text']:
        intents.append(intent['intent'])
        patterns.append(pattern)
        responses.append(intent['responses'][0])  # Ambil respon pertama (bisa diubah sesuai kebutuhan)

# Membuat DataFrame dengan intent, pattern, dan response
data = pd.DataFrame({
    'intent': intents,
    'pattern': patterns,
    'response': responses
})

# Preprocessing: Lemmatization function
def lemmatize_text(text):
    tokens = word_tokenize(text.lower())
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmas)  # Gabungkan kembali jadi string

# Terapkan preprocessing ke kolom pattern untuk data training
data['pattern_lemma'] = data['pattern'].apply(lemmatize_text)

# Siapkan fitur dan label
X_text = data['pattern_lemma'].tolist()
y = data['intent'].tolist()

# Vectorizer dan model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X, y)

# KMeans Clustering (Menggunakan vektorisasi teks)
X_text_vec = vectorizer.transform(X_text)  # Pastikan hanya sekali menggunakan fit_transform
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_text_vec.toarray())

# Implementasi KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_scaled)

# Fungsi prediksi intent dari input pengguna
def predict_intent(user_input):
    user_input_lemma = lemmatize_text(user_input)
    user_vec = vectorizer.transform([user_input_lemma])
    intent_pred = dt_model.predict(user_vec)[0]
    return intent_pred

# Fungsi ambil response berdasarkan intent
def get_response(user_input):
    intent = predict_intent(user_input)
    responses = data[data['intent'] == intent]['response'].tolist()
    if responses:
        response = responses[0]  # Bisa juga random.choice jika ingin variasi
    else:
        response = "Maaf, saya belum mengerti pertanyaan Anda. Silakan hubungi admin."
    
    # Evaluasi model klasifikasi dan clustering saat memberikan jawaban
    print(f"Evaluasi Model untuk Input: {user_input}")  # Menambahkan print untuk memeriksa input
    evaluate_classification()  # Evaluasi klasifikasi
    evaluate_clustering()  # Evaluasi clustering
    
    return response

# Evaluasi model klasifikasi (Decision Tree)
def evaluate_classification():
    # Evaluasi terhadap data yang sudah diubah ke vektor
    dt_pred = dt_model.predict(X)
    dt_accuracy = accuracy_score(y, dt_pred)
    dt_cm = confusion_matrix(y, dt_pred)
    dt_classification_report = classification_report(y, dt_pred)
    
    print("Decision Tree Accuracy:", dt_accuracy)
    print("Decision Tree Confusion Matrix:\n", dt_cm)
    print("Decision Tree Classification Report:\n", dt_classification_report)
    
    dt_precision = precision_score(y, dt_pred, average='macro')
    dt_recall = recall_score(y, dt_pred, average='macro')
    print("Decision Tree Precision (Macro Average):", dt_precision)
    print("Decision Tree Recall (Macro Average):", dt_recall)

# Evaluasi clustering (KMeans)
def evaluate_clustering():
    silhouette = silhouette_score(X_scaled, kmeans.labels_)
    inertia = kmeans.inertia_

    print("Silhouette Score:", silhouette)
    print("Inertia:", inertia)

    # Visualisasi Hasil Clustering (grafik berbasis teks)
    # Menampilkan hasil clustering dengan plotext (grafik berbasis teks di terminal)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], color=kmeans.labels_)
    plt.title('Hasil Clustering (KMeans)')
    plt.show()

# Route untuk homepage
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_input = data.get("user_input", "")
    response = get_response(user_input)  # Dapatkan respons sesuai intent
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
