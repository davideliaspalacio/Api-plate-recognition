import os
import datetime
import logging
import time
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import firebase_admin
from firebase_admin import credentials, storage, firestore
import uuid
# from deeplearning import object_detection
from deeplearling import object_detection
# from whatsapp_api_client_python import API
import json
import google.api_core.exceptions
import difflib

cred = credentials.Certificate('./ia-car-plates-firebase-adminsdk-61xhu-df58abe964.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'ia-car-plates.appspot.com'
})
db = firestore.client()

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)

texts_by_filename = {}
entradas = {}
last_request_time = None  
last_detections = {}

MAX_REPEAT_COUNT = 2
COOLDOWN_DURATION = 3 

def check_cooldown():
    global last_request_time
    current_time = time.time()

    if last_request_time and current_time - last_request_time < COOLDOWN_DURATION:
        return True, current_time
    else:
        last_request_time = current_time
        return False, current_time

def calcular_tarifa(minutos):
    tarifa_por_hora = 5500
    horas = minutos / 60
    return tarifa_por_hora * int(horas + (1 if minutos % 60 > 0 else 0))

@app.route('/api/calcular-tarifa', methods=['POST'])
def calcular_tarifa_endpoint():
    try:
        data = request.get_json()
        app.logger.debug(f"Datos recibidos: {data}")
        entry_id = data['id']
        doc_ref = db.collection('entries').document(entry_id)
        doc = doc_ref.get()
        if doc.exists:
            entrada = doc.to_dict()
            app.logger.debug(f"Datos de entrada: {entrada}")
            if 'tarifa' in entrada and entrada['tarifa'] > 0:
                return jsonify({'error': 'Tarifa ya calculada'}), 400
            hora_entrada = entrada['last_entry_time']
            hora_actual = datetime.datetime.now(datetime.timezone.utc)
            duracion = hora_actual - hora_entrada
            minutos = duracion.total_seconds() / 60
            tarifa = calcular_tarifa(minutos)
            entrada['tarifa'] = tarifa
            entrada['time_spent'] = minutos
            entrada['hora_salida'] = hora_actual.isoformat()
            doc_ref.update(entrada)
            return jsonify({'tarifa': tarifa, 'time_spent': minutos, 'success': True}), 200
        else:
            return jsonify({'error': 'Entrada no encontrada', 'success': False}), 404
    except Exception as e:
        app.logger.error(f"Error en calcular_tarifa_endpoint: {e}", exc_info=True)
        return jsonify({'error': 'Error interno del servidor', 'message': str(e)}), 500

last_detections = {}

SIMILARITY_THRESHOLD = 0.6  # Ajustar el umbral según sea necesario

def are_plates_similar(plate1, plate2, threshold=0.8):
    return plate1 == plate2  # Simplicidad: compara si son exactamente iguales. Mejora esto para similitud real.

@app.route('/api/upload', methods=['POST'])
def upload_image():
    global last_request_time, last_plate_entry_time

    # Control de tiempo para la solicitud de subida
    current_time = time.time()
    if last_request_time and current_time - last_request_time < 3:
        return jsonify({'error': 'Cooldown en efecto, intente nuevamente después de unos segundos'}), 429
    last_request_time = current_time

    if 'image_name' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image_name']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        try:
            text_list, result_url, plate_urls = object_detection(image, filename)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

        texts_by_filename[filename] = text_list  

        if text_list:
            response_text = []
            current_time_dt = datetime.datetime.now(datetime.timezone.utc)
            for plate_text, plate_url in zip(text_list, plate_urls):
                # Verificar el tiempo de la última entrada por placa y similitud
                ignore_plate = False
                for existing_plate, last_entry_time in last_plate_entry_time.items():
                    time_diff = current_time - last_entry_time
                    if time_diff < 120:  # 2 minutos en segundos
                        if plate_text == existing_plate or are_plates_similar(plate_text, existing_plate):
                            ignore_plate = True
                            break

                if ignore_plate:
                    continue  # Ignorar la placa si se encontró una similar en los últimos 2 minutos

                # Actualizar el tiempo de la última entrada por placa
                last_plate_entry_time[plate_text] = current_time

                try:
                    doc_ref = db.collection('entries').where('placa', '==', plate_text).order_by('last_entry_time', direction=firestore.Query.DESCENDING).limit(1)
                    docs = doc_ref.stream()
                except google.api_core.exceptions.FailedPrecondition as e:
                    return jsonify({'error': 'El índice necesario está en proceso de creación. Por favor, inténtelo más tarde.'}), 500

                entrada_actual = None
                for doc in docs:
                    entrada_actual = doc.to_dict()
                    entry_id = doc.id
                
                if entrada_actual:
                    if entrada_actual['count'] >= MAX_REPEAT_COUNT:
                        nueva_entrada_id = str(uuid.uuid4())
                        db.collection('entries').document(nueva_entrada_id).set({
                            'id': nueva_entrada_id,
                            'placa': plate_text,
                            'count': 1,
                            'last_entry_time': current_time_dt,
                            'time_spent': 0,
                            'tarifa': 0,
                            'hora_salida': None,
                            'firebase_url': result_url,
                            'editado': False,
                            'entrada_image_url': result_url,
                            'salida_image_url': None,
                            'plate_image_url': plate_url
                        })
                    else:
                        entrada_actual['count'] += 1
                        time_difference = (current_time_dt - entrada_actual['last_entry_time']).total_seconds() / 60.0
                        tarifa = calcular_tarifa(time_difference)
                        entrada_actual['time_spent'] = time_difference
                        entrada_actual['tarifa'] = tarifa
                        entrada_actual['hora_salida'] = current_time_dt.isoformat()
                        entrada_actual['salida_image_url'] = result_url
                        db.collection('entries').document(entry_id).update(entrada_actual)
                        response_text.append({
                            'id': entry_id,
                            'placa': plate_text,
                            'time_spent': time_difference,
                            'tarifa': tarifa,
                            'firebase_url': result_url,
                            'plate_image_url': plate_url
                        })
                else:
                    nueva_entrada_id = str(uuid.uuid4())
                    db.collection('entries').document(nueva_entrada_id).set({
                        'id': nueva_entrada_id,
                        'placa': plate_text,
                        'count': 1,
                        'last_entry_time': current_time_dt,
                        'time_spent': 0,
                        'tarifa': 0,
                        'hora_salida': None,
                        'firebase_url': result_url,
                        'editado': False,
                        'entrada_image_url': result_url,
                        'salida_image_url': None,
                        'plate_image_url': plate_url
                    })

            return jsonify({
                'upload_image': filename,
                'texts': text_list,
                'firebase_url': result_url,
                'detalles': response_text
            }), 200
        else:
            # Manejar caso de vehículos sin placa
            nueva_entrada_id = str(uuid.uuid4())
            current_time_dt = datetime.datetime.now(datetime.timezone.utc)
            no_plate_text = "NO_PLATE"
            plate_url = None  # No hay imagen de la placa porque no se detectó ninguna

            db.collection('entries').document(nueva_entrada_id).set({
                'id': nueva_entrada_id,
                'placa': no_plate_text,
                'count': 1,
                'last_entry_time': current_time_dt,
                'time_spent': 0,
                'tarifa': 0,
                'hora_salida': None,
                'firebase_url': result_url,
                'editado': False,
                'entrada_image_url': result_url,
                'salida_image_url': None,
                'plate_image_url': plate_url
            })

            return jsonify({
                'upload_image': filename,
                'texts': ["NO_PLATE"],
                'firebase_url': result_url,
                'detalles': [{
                    'id': nueva_entrada_id,
                    'placa': no_plate_text,
                    'time_spent': 0,
                    'tarifa': 0,
                    'firebase_url': result_url,
                    'plate_image_url': plate_url
                }]
            }), 200

# Inicializar el diccionario de tiempos de la última entrada por placa
last_plate_entry_time = {}

@app.route('/api/borrar-hora-salida', methods=['POST'])
def borrar_hora_salida():
    try:
        data = request.get_json()
        entry_id = data.get('id')
        if not entry_id:
            return jsonify({'status': 'error', 'message': 'ID no proporcionado'}), 400

        doc_ref = db.collection('entries').document(entry_id)
        doc = doc_ref.get()
        if doc.exists:
            entrada = doc.to_dict()
            entrada['hora_salida'] = None
            entrada['salida_image_url'] = None
            entrada['time_spent'] = 0
            entrada['tarifa'] = 0
            doc_ref.update(entrada)
            return jsonify({'status': 'success', 'message': 'Hora de salida borrada correctamente'}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Entrada no encontrada'}), 404
    except Exception as e:
        app.logger.error(f"Error en borrar_hora_salida: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/borrar-entrada', methods=['POST'])
def borrar_entrada():
    try:
        data = request.get_json()
        entry_id = data.get('id')
        if not entry_id:
            return jsonify({'status': 'error', 'message': 'ID no proporcionado'}), 400

        doc_ref = db.collection('entries').document(entry_id)
        doc = doc_ref.get()
        if doc.exists:
            doc_ref.delete()
            return jsonify({'status': 'success', 'message': 'Entrada borrada correctamente'}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Entrada no encontrada'}), 404
    except Exception as e:
        app.logger.error(f"Error en borrar_entrada: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
@app.route('/api/crear-instancia', methods=['POST'])
def crear_instancia():
    try:
        data = request.get_json()
        placa = data.get('placa')
        if not placa:
            return jsonify({'status': 'error', 'message': 'Placa no proporcionada'}), 400

        nueva_entrada_id = str(uuid.uuid4())
        current_time = datetime.datetime.now(datetime.timezone.utc)
        imagen_predeterminada = 'https://static.vecteezy.com/system/resources/thumbnails/008/585/294/small/3d-rendering-sport-blue-car-on-white-bakcground-jpg-free-photo.jpg'
        db.collection('entries').document(nueva_entrada_id).set({
            'id': nueva_entrada_id,
            'placa': placa,
            'count': 1,
            'last_entry_time': current_time,
            'time_spent': 0,
            'tarifa': 0,
            'hora_salida': None,
            'firebase_url': None,
            'editado': False,
            'entrada_image_url': imagen_predeterminada,
            'salida_image_url': None
        })
        return jsonify({'status': 'success', 'message': 'Instancia creada correctamente'}), 200
    except Exception as e:
        app.logger.error(f"Error en crear_instancia: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/actualizarplaca', methods=['POST'])
def actualizar_placa():
    try:
        data = request.get_json()
        app.logger.debug(f"Datos recibidos: {data}")
        entry_id = data.get('id') 
        nueva_placa = data.get('nueva_placa')

        if not entry_id:
            return jsonify({'status': 'error', 'message': 'ID no proporcionado'}), 400

        if not nueva_placa:
            return jsonify({'status': 'error', 'message': 'Nueva placa no proporcionada'}), 400

        doc_ref = db.collection('entries').document(entry_id)
        doc = doc_ref.get()
        if doc.exists:
            entrada = doc.to_dict()
            entrada['editado'] = True
            entrada['placa'] = nueva_placa

            existing_docs = db.collection('entries').where('placa', '==', nueva_placa).stream()
            for existing_doc in existing_docs:
                if existing_doc.id != entry_id:
                    existing_entry = existing_doc.to_dict()

                    current_time = datetime.datetime.now(datetime.timezone.utc)
                    time_difference = (current_time - existing_entry['last_entry_time']).total_seconds() / 60.0
                    tarifa = calcular_tarifa(time_difference)
                    
                    existing_entry['hora_salida'] = current_time.isoformat()
                    existing_entry['salida_image_url'] = entrada['entrada_image_url'] 
                    existing_entry['time_spent'] = time_difference
                    existing_entry['tarifa'] = tarifa

                    db.collection('entries').document(existing_doc.id).update(existing_entry)
                    db.collection('entries').document(entry_id).delete()
                    return jsonify({'status': 'success', 'message': 'Placa actualizada, entrada duplicada eliminada, y tarifa calculada', 'tarifa': tarifa, 'time_spent': time_difference}), 200

            doc_ref.update(entrada)
            return jsonify({'status': 'success', 'message': 'Placa actualizada correctamente'}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Entrada no encontrada'}), 404
    except Exception as e:
        app.logger.error(f"Error en actualizar_placa: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/obtener-entradas', methods=['GET'])
def obtener_entradas():
    try:
        filtro = request.args.get('filtro', 'dia')
        hoy = datetime.datetime.now(datetime.timezone.utc)
        
        if filtro == 'dia':
            fecha = request.args.get('fecha', None)
            if fecha:
                inicio = datetime.datetime.fromisoformat(fecha).replace(hour=0, minute=0, second=0, microsecond=0)
                fin = inicio + datetime.timedelta(days=1)
            else:
                inicio = hoy.replace(hour=0, minute=0, second=0, microsecond=0)
                fin = inicio + datetime.timedelta(days=1)
        elif filtro == 'semana':
            inicio = hoy - datetime.timedelta(days=hoy.weekday())
            inicio = inicio.replace(hour=0, minute=0, second=0, microsecond=0)
            fin = inicio + datetime.timedelta(days=7)
        elif filtro == 'mes':
            inicio = hoy.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            fin = (inicio + datetime.timedelta(days=32)).replace(day=1)
        else:
            try:
                filtro_date = datetime.datetime.fromisoformat(filtro)
                inicio = filtro_date.replace(hour=0, minute=0, second=0, microsecond=0)
                fin = inicio + datetime.timedelta(days=1)
            except ValueError:
                return jsonify({'error': 'Filtro no válido'}), 400

        docs = db.collection('entries').where(
            'last_entry_time', '>=', inicio
        ).where(
            'last_entry_time', '<', fin
        ).stream()
        
        lista_entradas = []

        for doc in docs:
            entrada = doc.to_dict()
            entrada_info = {
                'id': doc.id,
                'placa': entrada['placa'],
                'hora_entrada': entrada['last_entry_time'].isoformat(),
                'entrada_image_url': entrada.get('entrada_image_url'),
                'salida_image_url': entrada.get('salida_image_url'),
                'editado': entrada.get('editado', False),
                'time_spent': entrada.get('time_spent', 0),
                'tarifa': entrada.get('tarifa', 0),
                'hora_salida': entrada.get('hora_salida')
            }
            lista_entradas.append(entrada_info)
        
        return jsonify(lista_entradas), 200
    except Exception as e:
        app.logger.error(f"Error al obtener las entradas: {e}", exc_info=True)
        return jsonify({'error': 'Error interno del servidor', 'message': str(e)}), 500

@app.route('/api/total-ingresos-fecha', methods=['GET'])
def total_ingresos_fecha():
    try:
        fecha = request.args.get('fecha')
        if not fecha:
            return jsonify({'error': 'Fecha no proporcionada'}), 400
        
        fecha_inicio = datetime.datetime.fromisoformat(fecha).replace(hour=0, minute=0, second=0, microsecond=0)
        fecha_fin = fecha_inicio + datetime.timedelta(days=1)
        
        docs = db.collection('entries').where(
            'last_entry_time', '>=', fecha_inicio
        ).where(
            'last_entry_time', '<', fecha_fin
        ).stream()
        
        total_ingresos = 0.0

        for doc in docs:
            entrada = doc.to_dict()
            if 'tarifa' in entrada:
                total_ingresos += entrada['tarifa']

        return jsonify({'total_ingresos': total_ingresos}), 200
    except Exception as e:
        app.logger.error(f"Error al obtener el total de ingresos por fecha: {e}", exc_info=True)
        return jsonify({'error': 'Error interno del servidor', 'message': str(e)}), 500


@app.route('/api/ver-placa/<filename>', methods=['GET'])
def ver_placa(filename):
    text_list = texts_by_filename.get(filename)
    if text_list is not None:
        return jsonify({'filename': filename, 'texts': text_list})
    else:
        return jsonify({'error': 'File not found or no text detected'}), 404

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

@app.route('/api/total-ingresos-dia', methods=['GET'])
def total_ingresos_dia():
    try:
        docs = db.collection('entries').stream()
        total_ingresos = 0.0
        hoy = datetime.datetime.now(datetime.timezone.utc).date()

        for doc in docs:
            entrada = doc.to_dict()
            hora_entrada = entrada['last_entry_time']
            if isinstance(hora_entrada, str):
                hora_entrada = datetime.datetime.fromisoformat(hora_entrada)
            if hora_entrada.date() == hoy and 'tarifa' in entrada:
                total_ingresos += entrada['tarifa']

        return jsonify({'total_ingresos': total_ingresos}), 200
    except Exception as e:
        app.logger.error(f"Error al obtener el total de ingresos: {e}", exc_info=True)
        return jsonify({'error': 'Error interno del servidor', 'message': str(e)}), 500

@app.route('/api/ultima-placa', methods=['GET'])
def obtener_ultima_placa():
    docs = db.collection('entries').order_by('last_entry_time', direction=firestore.Query.DESCENDING).limit(1).stream()
    for doc in docs:
        entrada = doc.to_dict()
        entrada_info = {
            'id': doc.id,
            'placa': entrada['placa'],
            'hora_entrada': entrada['last_entry_time'].isoformat(),
            'firebase_url': entrada['firebase_url'],
            'editado': entrada.get('editado', False),
            'time_spent': entrada.get('time_spent', 0),
            'tarifa': entrada.get('tarifa', 0)
        }
        return jsonify(entrada_info)
    return jsonify({'error': 'No hay placas registradas'}), 404

@app.route('/api/buscar-placa', methods=['GET'])
def buscar_placa():
    try:
        placa = request.args.get('placa')
        if not placa:
            return jsonify({'error': 'Placa no proporcionada'}), 400

        docs = db.collection('entries').where(
            'placa', '==', placa
        ).stream()
        
        resultados = []
        for doc in docs:
            entrada = doc.to_dict()
            entrada_info = {
                'id': doc.id,
                'placa': entrada['placa'],
                'hora_entrada': entrada['last_entry_time'].isoformat(),
                'firebase_url': entrada['firebase_url'],
                'editado': entrada.get('editado', False),
                'time_spent': entrada.get('time_spent', 0),
                'tarifa': entrada.get('tarifa', 0)
            }
            resultados.append(entrada_info)
        
        if resultados:
            return jsonify(resultados), 200
        else:
            return jsonify({'error': 'Placa no encontrada'}), 404
    except Exception as e:
        app.logger.error(f"Error en buscar_placa: {e}", exc_info=True)
        return jsonify({'error': 'Error interno del servidor', 'message': str(e)}), 500



if __name__ == '__main__':
    app.run()
