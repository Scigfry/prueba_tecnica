import os
import joblib
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

# Conexión a la base de datos de Docker
conexion = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="root",
    database="atc"
)

cursor = conexion.cursor()

# Obtenemos todo menos los que tienen impagos
consulta = """
    SELECT e.email, GROUP_CONCAT(c.nombre) AS categorias
    FROM emails e
    LEFT JOIN emails_categorias ec ON e.id = ec.email_id
    LEFT JOIN categorias c ON ec.categoria_id = c.id
    WHERE e.client_id NOT IN (
        SELECT clientes.id 
        FROM clientes
        INNER JOIN impagos ON clientes.id = impagos.client_id
    )
    GROUP BY e.id
"""

cursor.execute(consulta)
resultados = cursor.fetchall()

cursor.close()
conexion.close()

texts = [item[0] for item in resultados]
categories = [item[1].split(",") if item[1] else [] for item in resultados]

X_train, X_test, y_train, y_test = train_test_split(texts, categories, test_size=0.22, random_state=42)

mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

classifier = OneVsRestClassifier(MultinomialNB())
classifier.fit(X_train_tfidf, y_train_bin)

# Directorio para almacenar los modelos
model_directory = 'app/model/'
os.makedirs(model_directory, exist_ok=True)

joblib.dump(classifier, os.path.join(model_directory, 'modelo_entrenado_multilabel.joblib'))
joblib.dump(vectorizer, os.path.join(model_directory, 'vectorizador_tfidf_multilabel.joblib'))
joblib.dump(mlb, os.path.join(model_directory, 'multi_label_binarizer.joblib'))
