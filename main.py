import whisper
import os
import time

# Asegúrate de instalar el paquete con:
# pip install -U openai-whisper


def transcribe_audio(file_path: str, model_name: str = "base") -> str:
    """
    Carga el modelo Whisper y transcribe el archivo de audio dado.

    :param file_path: Ruta al archivo de audio (por ejemplo, "audio.wav").
    :param model_name: Nombre del modelo Whisper a usar (tiny, base, small, medium, large).
    :return: Texto transcrito.
    """
    # Verificar que el archivo exista
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")


    t1 = time.time()
    # Cargar el modelo
    model = whisper.load_model(model_name)
    print(f'Tiempo de carga del modelo: {time.time() - t1}')

    t1 = time.time()
    # Realizar la transcripción
    result = model.transcribe(file_path)
    print(f'Tiempo de transcripcion: {time.time() - t1}')

    # Devolver el texto transcrito
    return result.get("text", "")


if __name__ == "__main__":
    # Archivo de audio en la raíz del proyecto
    audio_file = "audio.wav"

    try:
        print("Transcribiendo audio, por favor espera...")
        transcription = transcribe_audio(audio_file, model_name="tiny")
        print("\n--- Transcripción ---")
        print(transcription)
    except Exception as e:
        print(f"Error al transcribir: {e}")
