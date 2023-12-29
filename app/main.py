import joblib
import mysql.connector
from flask import Flask, request, jsonify
from sklearn.preprocessing import MultiLabelBinarizer

app = Flask(__name__)

classifier = joblib.load('./model/modelo_entrenado_multilabel.joblib')
vectorizer = joblib.load('./model/vectorizador_tfidf_multilabel.joblib')
mlb = joblib.load('./model/multi_label_binarizer.joblib')

@app.route('/classify-email', methods=['POST'])
def classify_email():
    try:
        data = request.get_json()

        client_id = data.get('client_id')
        fecha_envio = data.get('fecha_envio')
        email_body = data.get('email_body')
        
        # Conexión a la base de datos de Docker
        conexion = mysql.connector.connect(
            host='db',
            user='root',
            password='root',
            database='atc'
        )

        cursor = conexion.cursor()

        consulta = 'SELECT COUNT(*) FROM impagos WHERE client_id = %s'  

        cursor.execute(consulta, (client_id,))
        resultado = cursor.fetchone()

        cursor.close()
        conexion.close()

        if resultado[0] > 0:
            response = {
                'exito': False,
                'razon': 'El cliente tiene impagos'
            }
        else:
            new_text_tfidf = vectorizer.transform([email_body])
            predicciones_bin = classifier.predict(new_text_tfidf)
            predicciones_multilabel = mlb.inverse_transform(predicciones_bin)
            
            response = {
                'exito': True,
                'prediccion': list(predicciones_multilabel)
            }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': 'Error en el servidor' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")