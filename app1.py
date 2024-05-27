from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from deeplearning import object_detection
import firebase_admin
from firebase_admin import credentials, storage
import werkzeug
import datetime
from whatsapp_api_client_python import API
import json


cred = credentials.Certificate('./ia-car-plates-firebase-adminsdk-61xhu-df58abe964.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'ia-car-plates.appspot.com'
})

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')
PREDICT_PATH = os.path.join(BASE_PATH, 'static/predict/')
texts_by_filename = {}
entradas = {}

def upload_file_to_firebase(filename):
    bucket = storage.bucket()
    blob = bucket.blob('upload/' + filename)
    blob.upload_from_filename(filename)

    blob.make_public()
    return blob.public_url

# def calcular_tarifa(minutos):
#     tarifa_por_hora = 5500
#     horas = minutos / 60
#     if horas < 1:
#         return tarifa_por_hora
#     else:
#         return tarifa_por_hora * int(horas + (1 if minutos % 60 > 0 else 0))  
MAX_REPEAT_COUNT = 2

def calcular_tarifa(minutos):
    tarifa_por_hora = 5500
    horas = minutos / 60
    return tarifa_por_hora * int(horas + (1 if minutos % 60 > 0 else 0))

@app.route('/api/calcular-tarifa', methods=['POST'])
def calcular_tarifa_endpoint():
    data = request.get_json()
    placa = data['placa']
    if placa in entradas:
        entrada = entradas[placa]
        if 'tarifa' in entrada and entrada['tarifa'] > 0:
            return jsonify({'error': 'Tarifa ya calculada'}), 400
        hora_entrada = entrada['last_entry_time']
        hora_actual = datetime.datetime.now()
        duracion = hora_actual - hora_entrada
        minutos = duracion.total_seconds() / 60
        tarifa = calcular_tarifa(minutos)
        entrada['tarifa'] = tarifa
        entrada['time_spent'] = minutos
        return jsonify({'tarifa': tarifa, 'time_spent': minutos, 'success': True}), 200
    return jsonify({'error': 'Placa no encontrada', 'success': False}), 404


@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image_name' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image_name']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        path_save = os.path.join(UPLOAD_PATH, filename)
        file.save(path_save)

        try:
            public_url = upload_file_to_firebase(path_save)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

        text_list = object_detection(path_save, filename)
        texts_by_filename[filename] = text_list  

        if text_list:
            response_text = []
            for plate_text in text_list:
                current_time = datetime.datetime.now()
                if plate_text not in entradas:
                    entradas[plate_text] = {
                        'count': 1,
                        'last_entry_time': current_time,
                        'time_spent': 0,
                        'tarifa': 0,
                        'firebase_url': public_url  
                    }
                else:
                    entrada_actual = entradas[plate_text]
                    if entrada_actual['count'] >= MAX_REPEAT_COUNT:
                        entradas[plate_text] = {
                            'count': 1,
                            'last_entry_time': current_time,
                            'time_spent': 0,
                            'tarifa': 0,
                            'firebase_url': public_url 
                        }
                    else:
                        entrada_actual['count'] += 1
                        time_difference = (current_time - entrada_actual['last_entry_time']).total_seconds() / 60.0
                        tarifa = calcular_tarifa(time_difference)
                        entrada_actual['time_spent'] = time_difference
                        entrada_actual['tarifa'] = tarifa
                        entrada_actual['last_entry_time'] = current_time
                        response_text.append({
                            'placa': plate_text,
                            'time_spent': time_difference,
                            'tarifa': tarifa,
                            'firebase_url': public_url 
                        })

            return jsonify({
                'upload_image': filename,
                'texts': text_list,
                'firebase_url': public_url,
                'detalles': response_text
            }), 200
        else:
            return jsonify({
                'upload_image': filename,
                'texts': text_list,
                'firebase_url': public_url,
                'error': 'No text detected'
            }), 400

@app.route('/api/actualizarplaca', methods=['POST'])
def actualizar_placa():
    data = request.get_json()
    placa_original = data['placa_original']
    nueva_placa = data['nueva_placa']

    if placa_original in entradas:
        entrada = entradas.pop(placa_original)
        entrada['editado'] = True
        entradas[nueva_placa] = entrada
        return jsonify({'status': 'success', 'message': 'Placa actualizada correctamente'}), 200
    else:
        return jsonify({'status': 'error', 'message': 'Placa original no encontrada'}), 404


@app.route('/api/obtener-entradas', methods=['GET'])
def obtener_entradas():
    lista_entradas = []
    for placa, entrada in entradas.items():
        entrada_info = {
            'placa': placa,
            'hora_entrada': entrada['last_entry_time'].isoformat(),
            'firebase_url': entrada['firebase_url'], 
            'editado': entrada.get('editado', False),
            'time_spent': entrada.get('time_spent', 0),
            'tarifa': entrada.get('tarifa', 0)
        }
        lista_entradas.append(entrada_info)
    return jsonify(lista_entradas)


@app.route('/api/ver-placa/<filename>', methods=['GET'])
def ver_placa(filename):
    text_list = texts_by_filename.get(filename)
    if text_list is not None:
        return jsonify({'filename': filename, 'texts': text_list})
    else:
        return jsonify({'error': 'File not found or no text detected'}), 404

@app.route('/api/images/<filename>', methods=['GET'])
def get_image(filename):
    return send_from_directory(PREDICT_PATH, filename)

entradas = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    if request.method == 'POST' and 'placa' in request.form:
        placa = request.form['placa']
        if placa not in entradas:
            entradas[placa] = datetime.datetime.now()
        else:
            error = "La placa ya fue registrada."
    return render_template('index.html', entradas=entradas, error=error)

@app.route('/finalizar/<placa>', methods=['POST'])
def finalizar(placa):
    hora_entrada = entradas.pop(placa, None)
    if hora_entrada:
        hora_salida = datetime.datetime.now()
        duracion = hora_salida - hora_entrada
        minutos = divmod(duracion.total_seconds(), 60)[0]
        return render_template('resultado.html', minutos=minutos, placa=placa, hora_entrada=hora_entrada, hora_salida=hora_salida)
    else:
        return redirect(url_for('index'))

@app.route('/whatsapp/webhook', methods=['POST'])
def whatsapp_webhook():
    if request.method == 'POST':
        data = request.get_json()
        print("entro con " + json.dumps(data))
        procesar_mensaje(data)
        return jsonify({"status": "success"}), 200
    else:
        return jsonify({"status": "error"}), 400
    
@app.route('/api/total-ingresos-dia', methods=['GET'])
def total_ingresos_dia():
    total_ingresos = sum(entrada['tarifa'] for entrada in entradas.values() if entrada['last_entry_time'].date() == datetime.datetime.now().date())
    return jsonify({'total_ingresos': total_ingresos})

@app.route('/api/ultima-placa', methods=['GET'])
def obtener_ultima_placa():
    if entradas:
        # Ordena las entradas por la fecha y hora de la última entrada
        ultima_placa = max(entradas.items(), key=lambda x: x[1]['last_entry_time'])
        entrada_info = {
            'placa': ultima_placa[0],
            'hora_entrada': ultima_placa[1]['last_entry_time'].isoformat(),
            'firebase_url': ultima_placa[1]['firebase_url'],
            'editado': ultima_placa[1].get('editado', False),
            'time_spent': ultima_placa[1].get('time_spent', 0),
            'tarifa': ultima_placa[1].get('tarifa', 0)
        }
        return jsonify(entrada_info)
    else:
        return jsonify({'error': 'No hay placas registradas'}), 404


def procesar_mensaje(data):
    try:
        message_data = data['messageData']
        if message_data['typeMessage'] == 'textMessage':
            message = message_data['textMessageData']['textMessage']
            sender = data['senderData']['chatId']
        elif message_data['typeMessage'] == 'extendedTextMessage':
            message = message_data['extendedTextMessageData']['text']
            sender = data['senderData']['chatId']
        else:
            raise KeyError("Tipo de mensaje no soportado")

        whatsapp = API.GreenAPI('7103931186', 'deae7727f47b4592aff2780288b5b5e9c948008aa0594bcd90')

        if message.lower() == 'entrando parqueadero' or message.lower() == 'iniciar':
            ultima_placa_info = obtener_ultima_placa().get_json()  
            if 'error' not in ultima_placa_info:
                mensaje_respuesta = f"*Muchas gracias por tu ingreso a nuestro parqueadero🚗*. Esta es tu placa: *{ultima_placa_info['placa']}*\nHora de Entrada: *{ultima_placa_info['hora_entrada']}*\nesta es la imgen de tu vehiculo 🚗: {ultima_placa_info['firebase_url']}\n"
                whatsapp.sending.sendMessage(sender, mensaje_respuesta)
            else:
                whatsapp.sending.sendMessage(sender, "Lo sentimos, no hay registros de placas recientes.")

    except KeyError as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
