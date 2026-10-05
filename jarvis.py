import os
import re
import json
import time
import wave
import webbrowser
import subprocess
import threading
import difflib
import urllib.parse
import urllib.request
import ctypes
import unicodedata
from datetime import datetime
from html.parser import HTMLParser

import numpy as np
import sounddevice as sd
import webrtcvad
import psutil
import pynvml
from faster_whisper import WhisperModel
from openai import OpenAI
from piper import PiperVoice

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MEMORIA_FILE = os.path.join(
    BASE_DIR,
    "memoria.json"
)

APRENDIZAJE_FILE = os.path.join(
    BASE_DIR,
    "aprendizaje.json"
)

AUDIO_FILE = os.path.join(
    BASE_DIR,
    "voz.wav"
)

PARAR_FILE = os.path.join(
    BASE_DIR,
    "comando_parar.wav"
)

GEMMA_MODEL = "google/gemma-4-e4b"
LM_STUDIO_URL = "http://localhost:1234/v1"

GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
).strip()

PIPER_MODEL = os.path.join(
    BASE_DIR,
    "es_ES-davefx-medium.onnx"
)

CUDA_CUBLAS = os.path.join(
    BASE_DIR,
    ".venv",
    "Lib",
    "site-packages",
    "nvidia",
    "cublas",
    "bin"
)

if os.path.exists(CUDA_CUBLAS):
    os.environ["PATH"] += (
        os.pathsep
        + CUDA_CUBLAS
    )


# ============================================================
# PERSONALIDAD
# ============================================================

SYSTEM_PROMPT = """
Eres Jarvis, un asistente de IA para Windows.

Responde siempre en español.

Sé breve, directo y natural.

Normalmente responde en una o dos frases.

No uses Markdown, asteriscos, listas ni encabezados.

No inventes información.

No afirmes haber realizado una acción que Python no haya
realizado realmente.

No muestres razonamiento interno.

Cuando recibas información obtenida de Internet, úsala como
fuente de información y responde directamente a la pregunta
del usuario.
"""


# ============================================================
# AUDIO
# ============================================================

RATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(
    RATE * FRAME_MS / 1000
)

SILENCIO_SEGUNDOS = 1.0

vad = webrtcvad.Vad(2)

interrumpir_voz = threading.Event()
escuchando_para = threading.Event()

esperando_ciudad_clima = False

confirmacion_pendiente = None


# ============================================================
# WHISPER
# ============================================================

print("Cargando Whisper...")

whisper = WhisperModel(
    "small",
    device="cuda",
    compute_type="float16"
)

print("Whisper cargado.")


# ============================================================
# GEMMA
# ============================================================

print("Conectando con Gemma...")

client = OpenAI(
    base_url=LM_STUDIO_URL,
    api_key="lm-studio"
)

print("Gemma conectado.")


# ============================================================
# GEMINI + GOOGLE SEARCH
# ============================================================

gemini_client = None

if genai is None:
    print(
        "⚠️ google-genai no está instalado. "
        "Gemini con Google Search no estará disponible."
    )

elif not GEMINI_API_KEY:
    print(
        "⚠️ GEMINI_API_KEY no está configurada. "
        "Gemini con Google Search no estará disponible."
    )

else:
    try:
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print(
            "Gemini + Google Search conectado."
        )

    except Exception as e:
        print(
            f"⚠️ No se pudo conectar con Gemini: "
            f"{repr(e)}"
        )


# ============================================================
# PIPER
# ============================================================

print("Cargando voz...")

voice = PiperVoice.load(
    PIPER_MODEL
)

print("Voz cargada.")


# ============================================================
# MEMORIA
# ============================================================

def cargar_memoria():

    if not os.path.exists(
        MEMORIA_FILE
    ):
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

    try:

        with open(
            MEMORIA_FILE,
            "r",
            encoding="utf-8"
        ) as archivo:

            memoria = json.load(
                archivo
            )

        if not isinstance(
            memoria,
            list
        ):
            raise ValueError()

        if not memoria:

            memoria = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            ]

        else:

            memoria[0] = {
                "role": "system",
                "content": SYSTEM_PROMPT
            }

        return memoria

    except Exception:

        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]


memoria = cargar_memoria()


def cargar_aprendizaje():

    if not os.path.exists(
        APRENDIZAJE_FILE
    ):
        return {}

    try:

        with open(
            APRENDIZAJE_FILE,
            "r",
            encoding="utf-8"
        ) as archivo:

            datos = json.load(
                archivo
            )

        if isinstance(
            datos,
            dict
        ):
            return datos

    except Exception as e:

        print(
            f"⚠️ Error cargando aprendizaje: {e}"
        )

    return {}


aprendizaje = cargar_aprendizaje()


def normalizar_aprendizaje(texto):

    texto = (
        quitar_acentos(texto.lower())
        if "quitar_acentos" in globals()
        else texto.lower()
    )

    texto = re.sub(
        r"[^a-z0-9áéíóúüñ\s]",
        " ",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    return texto


def guardar_aprendizaje():

    try:

        with open(
            APRENDIZAJE_FILE,
            "w",
            encoding="utf-8"
        ) as archivo:

            json.dump(
                aprendizaje,
                archivo,
                ensure_ascii=False,
                indent=2
            )

    except Exception as e:

        print(
            f"⚠️ Error guardando aprendizaje: {e}"
        )


def aprender_comando(
    frase,
    comando
):

    frase = normalizar_aprendizaje(
        frase
    )

    comando = comando.strip()

    if (
        not frase
        or not comando
        or len(frase) < 2
    ):
        return False

    aprendizaje[frase] = comando

    guardar_aprendizaje()

    print(
        f"🧠 Aprendido: "
        f"{frase!r} → {comando!r}"
    )

    return True


def aplicar_aprendizaje(texto):

    clave = normalizar_aprendizaje(
        texto
    )

    if not clave:
        return texto

    comando = aprendizaje.get(
        clave
    )

    if comando:

        print(
            f"🧠 Aprendizaje aplicado: "
            f"{clave!r} → {comando!r}"
        )

        return comando

    texto_sin_cortesia = re.sub(
        r"\b(?:oye|jarvis|por favor|porfa|gracias|puedes|podrias|podrías|me puedes|me podrias|me podrías)\b",
        " ",
        clave,
        flags=re.IGNORECASE
    )

    texto_sin_cortesia = re.sub(
        r"\s+",
        " ",
        texto_sin_cortesia
    ).strip()

    comando = aprendizaje.get(
        texto_sin_cortesia
    )

    if comando:

        print(
            f"🧠 Aprendizaje aplicado: "
            f"{texto_sin_cortesia!r} → {comando!r}"
        )

        return comando

    return texto


def extraer_orden_aprendizaje(texto):

    limpio = texto.strip(
        " .?!"
    )

    patrones = [

        r"aprende que (.+?) significa que (.+)",

        r"aprende que (.+?) significa (.+)",

        r"aprende que (.+?) quiere decir que (.+)",

        r"aprende que (.+?) quiere decir (.+)",

        r"aprende que (.+?) es (.+)",

        r"cuando diga (.+?) significa que (.+)",

        r"cuando diga (.+?) significa (.+)",

        r"cuando te diga (.+?) significa que (.+)",

        r"cuando te diga (.+?) significa (.+)",

        r"cuando diga (.+?),? haz (.+)",

        r"cuando te diga (.+?),? haz (.+)",

        r"si digo (.+?),? haz (.+)",

        r"si te digo (.+?),? haz (.+)"
    ]

    for patron in patrones:

        coincidencia = re.fullmatch(
            patron,
            limpio,
            flags=re.IGNORECASE
        )

        if coincidencia:

            frase = coincidencia.group(
                1
            ).strip(
                " .?!\"'"
            )

            comando = coincidencia.group(
                2
            ).strip(
                " .?!\"'"
            )

            comando = re.sub(
                r"^(que\s+|quiero\s+que\s+|quiero\s+|debes\s+|deberás\s+|deberas\s+|tienes\s+que\s+|tendrás\s+que\s+)",
                "",
                comando,
                flags=re.IGNORECASE
            ).strip()

            comando = re.sub(
                r"^abras\s+",
                "abre ",
                comando,
                flags=re.IGNORECASE
            )

            comando = re.sub(
                r"^abres\s+",
                "abre ",
                comando,
                flags=re.IGNORECASE
            )

            comando = re.sub(
                r"^habras\s+",
                "abre ",
                comando,
                flags=re.IGNORECASE
            )

            comando = re.sub(
                r"^abrir\s+",
                "abre ",
                comando,
                flags=re.IGNORECASE
            )

            if frase and comando:
                return frase, comando

    return None


def guardar_memoria():

    try:

        with open(
            MEMORIA_FILE,
            "w",
            encoding="utf-8"
        ) as archivo:

            json.dump(
                memoria,
                archivo,
                ensure_ascii=False,
                indent=2
            )

    except Exception as e:

        print(
            f"⚠️ Error guardando memoria: {e}"
        )


# ============================================================
# LIMPIAR TEXTO PARA PIPER
# ============================================================

def limpiar_para_voz(texto):

    texto = re.sub(
        r"```.*?```",
        "",
        texto,
        flags=re.DOTALL
    )

    texto = texto.replace(
        "**",
        ""
    )

    texto = texto.replace(
        "__",
        ""
    )

    texto = texto.replace(
        "*",
        ""
    )

    texto = texto.replace(
        "_",
        ""
    )

    texto = re.sub(
        r"^\s*#+\s*",
        "",
        texto,
        flags=re.MULTILINE
    )

    texto = re.sub(
        r"^\s*>\s*",
        "",
        texto,
        flags=re.MULTILINE
    )

    texto = re.sub(
        r"^\s*[-•]\s+",
        "",
        texto,
        flags=re.MULTILINE
    )

    texto = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        texto
    )

    texto = re.sub(
        r"(\d+)\s*%",
        r"\1 por ciento",
        texto
    )

    texto = re.sub(
        r"[ \t]+",
        " ",
        texto
    )

    texto = re.sub(
        r"\n{3,}",
        "\n\n",
        texto
    )

    return texto.strip()


# ============================================================
# CORRECCIONES WHISPER
# ============================================================

def corregir_terminos(texto):

    resultado = texto

    correcciones = {

        "titan falo": "Titanfall",
        "titan fal": "Titanfall",
        "titan fallo": "Titanfall",
        "titan fall": "Titanfall",
        "titan file": "Titanfall",
        "titanfile": "Titanfall",
        "titanfil": "Titanfall",

        "discort": "Discord",
        "discor": "Discord",
        "disporos": "Discord",
        "dispor": "Discord",
        "discoros": "Discord",

        "stiem": "Steam",
        "stim": "Steam",

        "prime vídeo": "Prime Video",
        "prime video": "Prime Video",

        "disney plas": "Disney Plus",
        "disney plus": "Disney Plus",

        "packet tracer": "Cisco Packet Tracer",
        "paquet tracer": "Cisco Packet Tracer",
        "packet trace": "Cisco Packet Tracer",

        "libre office": "LibreOffice",
        "libreofice": "LibreOffice",

        "rtg5070": "RTX 5070",
        "rtx5070": "RTX 5070",

        "zodac": "Zotac",
        "zodack": "Zotac",
        "zotack": "Zotac",

        "play station 5": "PlayStation 5",
        "play station": "PlayStation"
    }

    for incorrecto, correcto in correcciones.items():

        resultado = re.sub(
            r"\b"
            + re.escape(incorrecto)
            + r"\b",
            correcto,
            resultado,
            flags=re.IGNORECASE
        )

    correcciones_volumen = {

        "bajá": "baja",
        "subí": "sube",
        "suba": "sube",

        "bolumen": "volumen",
        "volúmen": "volumen",
        "volumenes": "volumen",

        "abilita": "habilita",
        "abilitar": "habilitar",
        "abilítalo": "habilitalo",
        "abilitalo": "habilitalo",

        "desabilita": "deshabilita",
        "desabilitar": "deshabilitar",

        "silencia de volumen":
            "silencia el volumen",

        "silencia volumen":
            "silencia el volumen"
    }

    for incorrecto, correcto in correcciones_volumen.items():

        resultado = re.sub(
            r"\b"
            + re.escape(incorrecto)
            + r"\b",
            correcto,
            resultado,
            flags=re.IGNORECASE
        )

    palabras = resultado.split()

    if palabras:

        primera = palabras[0].lower()

        ratio = difflib.SequenceMatcher(
            None,
            primera,
            "abre"
        ).ratio()

        if (
            ratio >= 0.65
            and primera != "abre"
        ):

            palabras[0] = "abre"

            resultado = " ".join(
                palabras
            )

    return resultado


# ============================================================
# ESCUCHAR "PARA"
# ============================================================

def escuchar_para():

    escuchando_para.set()

    try:

        while not interrumpir_voz.is_set():

            audio = []

            def callback(
                indata,
                frames,
                tiempo,
                status
            ):

                if status:
                    return

                audio.append(
                    bytes(indata)
                )

            try:

                with sd.RawInputStream(
                    samplerate=RATE,
                    blocksize=FRAME_SIZE,
                    dtype="int16",
                    channels=1,
                    callback=callback
                ):

                    inicio = time.time()

                    while (
                        time.time() - inicio < 2
                        and not interrumpir_voz.is_set()
                    ):

                        time.sleep(
                            0.05
                        )

            except Exception:

                break

            if not audio:
                continue

            datos = b"".join(
                audio
            )

            try:

                with wave.open(
                    PARAR_FILE,
                    "wb"
                ) as wav:

                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(RATE)

                    wav.writeframes(
                        datos
                    )

                segmentos, _ = whisper.transcribe(
                    PARAR_FILE,
                    language="es",
                    beam_size=1
                )

                texto = " ".join(
                    segmento.text
                    for segmento in segmentos
                ).lower()

                palabras_parar = [
                    "para",
                    "parar",
                    "párate",
                    "parate",
                    "cancela",
                    "cancelar"
                ]

                if any(
                    palabra in texto
                    for palabra in palabras_parar
                ):

                    interrumpir_voz.set()

                    sd.stop()

                    break

            except Exception:

                pass

    finally:

        escuchando_para.clear()


# ============================================================
# HABLAR
# ============================================================

def hablar(texto):

    texto = limpiar_para_voz(
        texto
    )

    if not texto:
        return

    interrumpir_voz.clear()

    try:

        with wave.open(
            "voz_generada.wav",
            "wb"
        ) as wav_file:

            voice.synthesize_wav(
                texto,
                wav_file
            )

        with wave.open(
            "voz_generada.wav",
            "rb"
        ) as wav_file:

            audio = wav_file.readframes(
                wav_file.getnframes()
            )

            samplerate = wav_file.getframerate()
            channels = wav_file.getnchannels()

        audio = np.frombuffer(
            audio,
            dtype=np.int16
        )

        if channels > 1:

            audio = audio.reshape(
                -1,
                channels
            )

        hilo = threading.Thread(
            target=escuchar_para,
            daemon=True
        )

        hilo.start()

        sd.play(
            audio,
            samplerate
        )

        duracion = len(audio) / samplerate
        inicio = time.time()

        while (
            time.time() - inicio
            < duracion
        ):

            if interrumpir_voz.is_set():

                sd.stop()

                break

            time.sleep(
                0.05
            )

    except Exception as e:

        print(
            f"⚠️ Error de voz: {e}"
        )

    finally:

        sd.stop()

        interrumpir_voz.clear()


# ============================================================
# GRABAR
# ============================================================

def grabar_hasta_silencio():

    print(
        "🎤 Habla..."
    )

    frames = []
    hablando = False
    silencio_inicio = None

    def callback(
        indata,
        frames_count,
        tiempo,
        status
    ):

        if status:
            return

        frames.append(
            bytes(indata)
        )

    with sd.RawInputStream(
        samplerate=RATE,
        blocksize=FRAME_SIZE,
        dtype="int16",
        channels=1,
        callback=callback
    ):

        inicio = time.time()

        while True:

            time.sleep(
                FRAME_MS / 1000
            )

            if len(frames) < 3:
                continue

            frame = frames[-1]

            try:

                es_voz = vad.is_speech(
                    frame,
                    RATE
                )

            except Exception:

                es_voz = False

            if es_voz:

                hablando = True
                silencio_inicio = None

            elif hablando:

                if silencio_inicio is None:

                    silencio_inicio = time.time()

                elif (
                    time.time()
                    - silencio_inicio
                    >= SILENCIO_SEGUNDOS
                ):

                    break

            if (
                not hablando
                and time.time() - inicio > 8
            ):

                break

    datos = b"".join(
        frames
    )

    with wave.open(
        AUDIO_FILE,
        "wb"
    ) as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)

        wav.writeframes(
            datos
        )

    return AUDIO_FILE


# ============================================================
# TRANSCRIBIR
# ============================================================

def transcribir(archivo):

    try:

        segmentos, _ = whisper.transcribe(
            archivo,
            language="es",
            beam_size=1
        )

        texto = " ".join(
            segmento.text
            for segmento in segmentos
        ).strip()

        return corregir_terminos(
            texto
        )

    except Exception as e:

        print(
            f"⚠️ Error Whisper: {e}"
        )

        return ""


# ============================================================
# WAKE WORD
# ============================================================

def es_jarvis(texto):

    texto = texto.lower().strip()

    texto = re.sub(
        r"[^\wáéíóúüñ ]",
        "",
        texto
    )

    palabras = texto.split()

    if (
        not palabras
        or len(palabras) > 2
    ):
        return False

    variantes = [
        "jarvis",
        "jervis",
        "jarbis",
        "jerbis",
        "gervis",
        "gerbis",
        "djervis",
        "chervis",
        "charvis",
        "harvis",
        "harbis",
        "hervis",
        "yervis",
        "havis",
        "dervis"
    ]

    for palabra in palabras:

        if palabra in variantes:
            return True

        if len(palabra) >= 4:

            ratio = difflib.SequenceMatcher(
                None,
                palabra,
                "jarvis"
            ).ratio()

            if ratio >= 0.72:
                return True

    return False


def esperar_jarvis():

    print(
        "🟢 Esperando «Jarvis»..."
    )

    while True:

        archivo = grabar_hasta_silencio()

        texto = transcribir(
            archivo
        )

        if not texto:
            continue

        print(
            f"👂 Detectado: ¡{texto}!"
        )

        if es_jarvis(texto):

            print(
                "🟢 «Jarvis» detectado."
            )

            return


# ============================================================
# ESTADO DEL PC
# ============================================================

def estado_ram():

    memoria_ram = psutil.virtual_memory()

    usada = memoria_ram.used / (
        1024 ** 3
    )

    total = memoria_ram.total / (
        1024 ** 3
    )

    return (
        f"Estás usando {usada:.1f} GB "
        f"de {total:.1f} GB de RAM, "
        f"un {memoria_ram.percent:.0f} por ciento."
    )


def estado_cpu():

    uso = psutil.cpu_percent(
        interval=0.5
    )

    return (
        f"La CPU está al "
        f"{uso:.0f} por ciento."
    )


def estado_gpu():

    try:

        handle = pynvml.nvmlDeviceGetHandleByIndex(
            0
        )

        nombre = pynvml.nvmlDeviceGetName(
            handle
        )

        if isinstance(
            nombre,
            bytes
        ):

            nombre = nombre.decode()

        uso = pynvml.nvmlDeviceGetUtilizationRates(
            handle
        ).gpu

        temperatura = pynvml.nvmlDeviceGetTemperature(
            handle,
            pynvml.NVML_TEMPERATURE_GPU
        )

        memoria = pynvml.nvmlDeviceGetMemoryInfo(
            handle
        )

        usada = memoria.used / (
            1024 ** 3
        )

        total = memoria.total / (
            1024 ** 3
        )

        try:

            potencia = (
                pynvml.nvmlDeviceGetPowerUsage(
                    handle
                )
                / 1000
            )

            potencia_texto = (
                f" y consume "
                f"{potencia:.0f} W"
            )

        except Exception:

            potencia_texto = ""

        return (
            f"La GPU es una {nombre}. "
            f"Está al {uso} por ciento, "
            f"a {temperatura} grados, "
            f"usando {usada:.1f} de "
            f"{total:.1f} GB de VRAM"
            f"{potencia_texto}."
        )

    except Exception:

        return (
            "No puedo consultar "
            "la GPU ahora."
        )


def estado_pc():

    return (
        estado_cpu()
        + " "
        + estado_ram()
        + " "
        + estado_gpu()
    )


# ============================================================
# HORA
# ============================================================

def obtener_hora():

    return (
        "Son las "
        + datetime.now().strftime(
            "%H:%M"
        )
        + "."
    )


# ============================================================
# CLIMA
# ============================================================

def normalizar_ciudad(ciudad):

    ciudad = ciudad.lower().strip()

    equivalencias = {

        "san boi":
            "Sant Boi de Llobregat",

        "san boy":
            "Sant Boi de Llobregat",

        "sant boi":
            "Sant Boi de Llobregat",

        "san boi de llobregat":
            "Sant Boi de Llobregat",

        "san boi de llubregat":
            "Sant Boi de Llobregat",

        "sante voy de liubrigad":
            "Sant Boi de Llobregat",

        "barcelona":
            "Barcelona",

        "lleida":
            "Lleida",

        "madrid":
            "Madrid",

        "extremadura":
            "Extremadura"
    }

    if ciudad in equivalencias:

        return equivalencias[
            ciudad
        ]

    ratio = difflib.SequenceMatcher(
        None,
        ciudad,
        "sant boi de llobregat"
    ).ratio()

    if ratio >= 0.55:

        return (
            "Sant Boi de Llobregat"
        )

    return ciudad


def obtener_coordenadas(ciudad):

    ciudad = normalizar_ciudad(
        ciudad
    )

    try:

        parametros = urllib.parse.urlencode({

            "name": ciudad,
            "count": 1,
            "language": "es",
            "format": "json"

        })

        url = (
            "https://geocoding-api.open-meteo.com/"
            f"v1/search?{parametros}"
        )

        solicitud = urllib.request.Request(
            url,
            headers={
                "User-Agent":
                    "Jarvis/1.0"
            }
        )

        with urllib.request.urlopen(
            solicitud,
            timeout=5
        ) as respuesta:

            datos = json.loads(
                respuesta.read().decode()
            )

        resultados = datos.get(
            "results",
            []
        )

        if not resultados:
            return None

        resultado = resultados[0]

        return {
            "lat":
                resultado["latitude"],
            "lon":
                resultado["longitude"],
            "nombre":
                resultado["name"]
        }

    except Exception:

        return None


def descripcion_tiempo(codigo):

    codigos = {

        0: "despejado",
        1: "mayormente despejado",
        2: "parcialmente nublado",
        3: "nublado",
        45: "niebla",
        48: "niebla",
        51: "llovizna ligera",
        53: "llovizna",
        55: "llovizna intensa",
        61: "lluvia ligera",
        63: "lluvia",
        65: "lluvia intensa",
        71: "nieve ligera",
        73: "nieve",
        75: "nieve intensa",
        80: "chubascos ligeros",
        81: "chubascos",
        82: "chubascos intensos",
        95: "tormenta",
        96: "tormenta con granizo",
        99: "tormenta con granizo"
    }

    return codigos.get(
        codigo,
        "condiciones desconocidas"
    )


def obtener_tiempo(ciudad):

    coordenadas = obtener_coordenadas(
        ciudad
    )

    if not coordenadas:

        return (
            f"No encuentro la localidad "
            f"{ciudad}."
        )

    try:

        parametros = urllib.parse.urlencode({

            "latitude":
                coordenadas["lat"],

            "longitude":
                coordenadas["lon"],

            "current":
                "temperature_2m,"
                "apparent_temperature,"
                "weather_code,"
                "wind_speed_10m",

            "daily":
                "temperature_2m_max,"
                "temperature_2m_min,"
                "precipitation_probability_max",

            "timezone":
                "auto"
        })

        url = (
            "https://api.open-meteo.com/"
            f"v1/forecast?{parametros}"
        )

        solicitud = urllib.request.Request(
            url,
            headers={
                "User-Agent":
                    "Jarvis/1.0"
            }
        )

        with urllib.request.urlopen(
            solicitud,
            timeout=5
        ) as respuesta:

            datos = json.loads(
                respuesta.read().decode()
            )

        actual = datos["current"]

        temperatura = actual[
            "temperature_2m"
        ]

        sensacion = actual[
            "apparent_temperature"
        ]

        estado = descripcion_tiempo(
            actual["weather_code"]
        )

        viento = actual[
            "wind_speed_10m"
        ]

        maxima = datos[
            "daily"
        ][
            "temperature_2m_max"
        ][0]

        minima = datos[
            "daily"
        ][
            "temperature_2m_min"
        ][0]

        lluvia = datos[
            "daily"
        ][
            "precipitation_probability_max"
        ][0]

        return (
            f"En {coordenadas['nombre']} hace "
            f"{temperatura:.0f} grados, "
            f"con sensación de "
            f"{sensacion:.0f}. "
            f"Está {estado}, hay viento "
            f"de {viento:.0f} kilómetros "
            f"por hora y la probabilidad "
            f"máxima de lluvia es del "
            f"{lluvia} por ciento. "
            f"Hoy la máxima será de "
            f"{maxima:.0f} y la mínima "
            f"de {minima:.0f} grados."
        )

    except Exception:

        return (
            "No he podido consultar "
            "el tiempo en este momento."
        )


def detectar_ciudad_clima(texto):

    patrones = [

        r"(?:tiempo|clima).*?"
        r"(?:en|de)\s+(.+?)(?:\?|$)",

        r"(?:qué|que)\s+tiempo\s+"
        r"(?:hace|hará|hara)\s+"
        r"(?:en\s+)?(.+?)(?:\?|$)",

        r"(?:temperatura).*?"
        r"(?:en|de)\s+(.+?)(?:\?|$)"
    ]

    for patron in patrones:

        coincidencia = re.search(
            patron,
            texto.lower().strip()
        )

        if coincidencia:

            ciudad = (
                coincidencia.group(1)
                .strip(" .,?!")
            )

            if ciudad:
                return ciudad

    return None


# ============================================================
# SITIOS WEB
# ============================================================

SITIOS = {

    "google":
        "https://www.google.com",

    "youtube":
        "https://www.youtube.com",

    "netflix":
        "https://www.netflix.com",

    "prime video":
        "https://www.primevideo.com",

    "amazon prime":
        "https://www.primevideo.com",

    "disney plus":
        "https://www.disneyplus.com",

    "disney+":
        "https://www.disneyplus.com",

    "hbo":
        "https://www.max.com",

    "hbo max":
        "https://www.max.com",

    "max":
        "https://www.max.com",

    "spotify":
        "https://open.spotify.com",

    "twitch":
        "https://www.twitch.tv",

    "gmail":
        "https://mail.google.com",

    "whatsapp":
        "https://web.whatsapp.com",

    "instagram":
        "https://www.instagram.com",

    "facebook":
        "https://www.facebook.com",

    "twitter":
        "https://x.com",

    "x":
        "https://x.com",

    "reddit":
        "https://www.reddit.com",

    "github":
        "https://github.com",

    "chatgpt":
        "https://chatgpt.com",

    "steam":
        "https://store.steampowered.com"
}


def parece_url(objetivo):

    objetivo = objetivo.lower().strip()

    return (
        objetivo.startswith(
            (
                "http://",
                "https://",
                "www."
            )
        )
        or objetivo.endswith(
            (
                ".com",
                ".es",
                ".net",
                ".org",
                ".tv",
                ".io",
                ".gg",
                ".dev"
            )
        )
    )


# ============================================================
# APLICACIONES WINDOWS
# ============================================================

def normalizar_nombre_app(nombre):

    nombre = nombre.lower().strip()

    aliases = {

        "ajustes":
            "configuración",

        "ajuste":
            "configuración",

        "settings":
            "configuración",

        "disco":
            "__NO_APP__",

        "discos":
            "__NO_APP__",

        "steam":
            "steam",

        "stiem":
            "steam",

        "stim":
            "steam",

        "discord":
            "discord",

        "discort":
            "discord",

        "discor":
            "discord",

        "disporos":
            "discord",

        "dispor":
            "discord",

        "chrome":
            "chrome",

        "google chrome":
            "chrome",

        "edge":
            "microsoft edge",

        "calculadora":
            "calculadora",

        "calc":
            "calculadora",

        "bloc de notas":
            "bloc de notas",

        "notepad":
            "notepad",

        "paint":
            "paint",

        "explorador":
            "explorador de archivos",

        "explorador de archivos":
            "explorador de archivos",

        "los sims":
            "the sims",

        "sims":
            "the sims",

        "the sims":
            "the sims",

        "libreoffice":
            "libreoffice",

        "libre office":
            "libreoffice",

        "writer":
            "libreoffice",

        "libreoffice writer":
            "libreoffice",

        "calc libreoffice":
            "libreoffice",

        "cisco packet tracer":
            "cisco packet tracer",

        "packet tracer":
            "cisco packet tracer"
    }

    return aliases.get(
        nombre,
        nombre
    )


def obtener_apps_windows():

    try:

        resultado = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "Get-StartApps | "
                "Select-Object Name,AppID | "
                "ConvertTo-Json -Compress"
            ],
            capture_output=True,
            text=True,
            timeout=8
        )

        if resultado.returncode != 0:
            return []

        salida = (
            resultado.stdout.strip()
        )

        if not salida:
            return []

        datos = json.loads(
            salida
        )

        if isinstance(
            datos,
            dict
        ):

            datos = [datos]

        return datos

    except Exception as e:

        print(
            f"⚠️ Error obteniendo aplicaciones: {e}"
        )

        return []


def buscar_aplicacion_windows(nombre):

    nombre = normalizar_nombre_app(
        nombre
    )

    if nombre == "__NO_APP__":
        return None

    apps = obtener_apps_windows()

    if not apps:
        return None

    mejor = None
    mejor_puntuacion = 0

    for app in apps:

        nombre_app = str(
            app.get(
                "Name",
                ""
            )
        ).strip()

        app_id = str(
            app.get(
                "AppID",
                ""
            )
        ).strip()

        if (
            not nombre_app
            or not app_id
        ):
            continue

        a = nombre_app.lower()
        b = nombre.lower()

        if a == b:
            return app

        if b in a:
            puntuacion = 0.95

        elif a in b:
            puntuacion = 0.90

        else:
            puntuacion = difflib.SequenceMatcher(
                None,
                a,
                b
            ).ratio()

        if puntuacion > mejor_puntuacion:

            mejor_puntuacion = puntuacion
            mejor = app

    if mejor_puntuacion >= 0.65:
        return mejor

    return None


def buscar_aplicacion_en_registro(nombre):

    nombre = nombre.lower().strip()

    powershell = r"""
$paths = @(
'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*',
'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*',
'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*'
)

$resultados = @()

foreach ($path in $paths) {

    $items = Get-ItemProperty $path -ErrorAction SilentlyContinue

    foreach ($item in $items) {

        if ($item.DisplayName) {

            [PSCustomObject]@{
                DisplayName = $item.DisplayName
                InstallLocation = $item.InstallLocation
                DisplayIcon = $item.DisplayIcon
            }
        }
    }
}

$resultados | ConvertTo-Json -Compress
"""

    try:

        resultado = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                powershell
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        if resultado.returncode != 0:
            return None

        salida = resultado.stdout.strip()

        if not salida:
            return None

        datos = json.loads(
            salida
        )

        if isinstance(
            datos,
            dict
        ):

            datos = [datos]

        mejor = None
        mejor_puntuacion = 0

        for programa in datos:

            display_name = str(
                programa.get(
                    "DisplayName",
                    ""
                )
            ).strip()

            if not display_name:
                continue

            a = display_name.lower()

            if a == nombre:
                puntuacion = 1.0

            elif nombre in a:
                puntuacion = 0.95

            elif a in nombre:
                puntuacion = 0.90

            else:
                puntuacion = difflib.SequenceMatcher(
                    None,
                    a,
                    nombre
                ).ratio()

            if puntuacion > mejor_puntuacion:

                mejor_puntuacion = puntuacion
                mejor = programa

        if mejor_puntuacion >= 0.65:
            return mejor

    except Exception:
        pass

    return None


def abrir_aplicacion_registro(nombre):

    app = buscar_aplicacion_en_registro(
        nombre
    )

    if not app:
        return False

    display_icon = str(
        app.get(
            "DisplayIcon",
            ""
        )
    ).strip()

    install_location = str(
        app.get(
            "InstallLocation",
            ""
        )
    ).strip()

    if display_icon:

        display_icon = (
            display_icon
            .split(",")[0]
            .strip('"')
        )

        if os.path.isfile(
            display_icon
        ):

            try:

                os.startfile(
                    display_icon
                )

                print(
                    "🚀 Aplicación encontrada "
                    f"en el registro: "
                    f"{app.get('DisplayName')}"
                )

                return True

            except Exception:
                pass

    if install_location:

        if os.path.isdir(
            install_location
        ):

            try:

                os.startfile(
                    install_location
                )

                print(
                    "🚀 Carpeta de aplicación "
                    f"abierta: "
                    f"{app.get('DisplayName')}"
                )

                return True

            except Exception:
                pass

    return False


def abrir_aplicacion_windows(nombre):

    especiales = {

        "configuración":
            "ms-settings:",

        "calculadora":
            "calc.exe",

        "notepad":
            "notepad.exe",

        "bloc de notas":
            "notepad.exe",

        "paint":
            "mspaint.exe",

        "explorador de archivos":
            "explorer.exe",

        "administrador de tareas":
            "taskmgr.exe",

        "panel de control":
            "control.exe",

        "cmd":
            "cmd.exe",

        "powershell":
            "powershell.exe",

        "terminal":
            "wt.exe",

        "administrador de discos":
            "diskmgmt.msc",

        "administrador de dispositivos":
            "devmgmt.msc"
    }

    nombre_normalizado = normalizar_nombre_app(
        nombre
    )

    if nombre_normalizado in especiales:

        try:

            os.startfile(
                especiales[
                    nombre_normalizado
                ]
            )

            print(
                f"🚀 Aplicación especial: "
                f"{nombre_normalizado}"
            )

            return True

        except Exception:
            pass

    app = buscar_aplicacion_windows(
        nombre
    )

    if app:

        nombre_app = app.get(
            "Name",
            nombre
        )

        app_id = app.get(
            "AppID",
            ""
        )

        if app_id:

            print(
                f"🚀 Aplicación encontrada: "
                f"{nombre_app}"
            )

            try:

                subprocess.Popen(
                    [
                        "explorer.exe",
                        f"shell:AppsFolder\\{app_id}"
                    ]
                )

                return True

            except Exception as e:

                print(
                    f"❌ Error abriendo aplicación: {e}"
                )

    if abrir_aplicacion_registro(
        nombre_normalizado
    ):

        return True

    return False


def abrir_objetivo(objetivo):

    objetivo = objetivo.strip(
        " .,?!"
    )

    objetivo_normalizado = (
        objetivo.lower()
    )

    if (
        objetivo_normalizado
        not in SITIOS
    ):

        if abrir_aplicacion_windows(
            objetivo
        ):

            return True

    if (
        objetivo_normalizado
        in SITIOS
    ):

        if objetivo_normalizado in {
            "steam",
            "discord",
            "spotify"
        }:

            if abrir_aplicacion_windows(
                objetivo
            ):

                return True

        try:

            webbrowser.open(
                SITIOS[
                    objetivo_normalizado
                ]
            )

            return True

        except Exception:

            return False

    if parece_url(
        objetivo
    ):

        if not objetivo.startswith(
            (
                "http://",
                "https://"
            )
        ):

            objetivo = (
                "https://"
                + objetivo
            )

        try:

            webbrowser.open(
                objetivo
            )

            return True

        except Exception:

            return False

    if os.path.exists(
        objetivo
    ):

        try:

            os.startfile(
                objetivo
            )

            return True

        except Exception:
            pass

    return False


# ============================================================
# DETECTAR APERTURA
# ============================================================

def extraer_objetivo_apertura(texto):

    limpio = texto.lower().strip()

    palabras = limpio.split()

    if palabras:

        primera = palabras[0]

        ratio = difflib.SequenceMatcher(
            None,
            primera,
            "abre"
        ).ratio()

        if (
            ratio >= 0.65
            and primera != "abre"
        ):

            limpio = (
                "abre "
                + " ".join(
                    palabras[1:]
                )
            )

    patrones = [

        r"abre\s+(.+)",
        r"abrir\s+(.+)",
        r"ejecuta\s+(.+)",
        r"ejecutar\s+(.+)",
        r"inicia\s+(.+)",
        r"iniciar\s+(.+)",
        r"lanza\s+(.+)",
        r"lanzar\s+(.+)"
    ]

    for patron in patrones:

        coincidencia = re.match(
            patron,
            limpio
        )

        if coincidencia:

            objetivo = (
                coincidencia.group(1)
                .strip(" .,?!")
            )

            objetivo = re.sub(
                r"\s+y\s+dime.*$",
                "",
                objetivo
            )

            objetivo = re.sub(
                r"^(la|el|la app|la aplicación|"
                r"el programa|programa|app)\s+",
                "",
                objetivo
            )

            if objetivo:
                return objetivo

    return None


# ============================================================
# BUSCADOR WEB
# ============================================================

class BuscadorHTML(
    HTMLParser
):

    def __init__(self):

        super().__init__()

        self.resultados = []
        self.enlace_actual = None
        self.texto_actual = ""
        self.capturando_titulo = False
        self.capturando_snippet = False

    def handle_starttag(
        self,
        tag,
        attrs
    ):

        atributos = dict(
            attrs
        )

        clase = atributos.get(
            "class",
            ""
        )

        if (
            tag == "a"
            and (
                "result__a"
                in clase
                or
                "result-link"
                in clase
            )
        ):

            self.enlace_actual = (
                atributos.get(
                    "href",
                    ""
                )
            )

            self.texto_actual = ""
            self.capturando_titulo = True

        elif (
            "result__snippet"
            in clase
            or
            "result-snippet"
            in clase
        ):

            self.texto_actual = ""
            self.capturando_snippet = True

    def handle_data(
        self,
        data
    ):

        if (
            self.capturando_titulo
            or self.capturando_snippet
        ):

            self.texto_actual += data

    def handle_endtag(
        self,
        tag
    ):

        if (
            tag == "a"
            and self.capturando_titulo
        ):

            titulo = (
                self.texto_actual.strip()
            )

            if (
                titulo
                and self.enlace_actual
            ):

                self.resultados.append({

                    "titulo":
                        titulo,

                    "url":
                        self.enlace_actual,

                    "snippet":
                        ""
                })

            self.enlace_actual = None
            self.texto_actual = ""
            self.capturando_titulo = False

        elif self.capturando_snippet:

            snippet = (
                self.texto_actual.strip()
            )

            if self.resultados:

                self.resultados[
                    -1
                ][
                    "snippet"
                ] = snippet

            self.texto_actual = ""
            self.capturando_snippet = False


class BuscadorGoogleHTML(
    HTMLParser
):

    def __init__(self):

        super().__init__()

        self.resultados = []

        self.enlace_actual = None

        self.capturando_titulo = False
        self.capturando_snippet = False

        self.texto_actual = ""

    def handle_starttag(
        self,
        tag,
        attrs
    ):

        atributos = dict(
            attrs
        )

        if tag == "a":

            href = atributos.get(
                "href",
                ""
            )

            if href:

                self.enlace_actual = href

        if tag == "h3":

            self.capturando_titulo = True
            self.texto_actual = ""

        if (
            tag == "div"
            and (
                "VwiC3b"
                in atributos.get(
                    "class",
                    ""
                )
                or
                "aCOpRe"
                in atributos.get(
                    "class",
                    ""
                )
            )
        ):

            self.capturando_snippet = True
            self.texto_actual = ""

    def handle_data(
        self,
        data
    ):

        if (
            self.capturando_titulo
            or
            self.capturando_snippet
        ):

            self.texto_actual += data

    def handle_endtag(
        self,
        tag
    ):

        if (
            tag == "h3"
            and self.capturando_titulo
        ):

            titulo = (
                self.texto_actual.strip()
            )

            if (
                titulo
                and self.enlace_actual
            ):

                url = self.enlace_actual

                if url.startswith(
                    "/url?"
                ):

                    parametros = urllib.parse.parse_qs(
                        urllib.parse.urlparse(
                            url
                        ).query
                    )

                    url = parametros.get(
                        "q",
                        [url]
                    )[0]

                self.resultados.append({

                    "titulo":
                        titulo,

                    "url":
                        url,

                    "snippet":
                        ""
                })

            self.capturando_titulo = False
            self.texto_actual = ""

        elif (
            tag == "div"
            and self.capturando_snippet
        ):

            snippet = (
                self.texto_actual.strip()
            )

            if self.resultados:

                self.resultados[
                    -1
                ][
                    "snippet"
                ] = snippet

            self.capturando_snippet = False
            self.texto_actual = ""


def descargar_buscador_post(
    url,
    consulta
):

    datos = urllib.parse.urlencode({

        "q":
            consulta,

        "kl":
            "es-es",

        "kp":
            "-2"
    }).encode(
        "utf-8"
    )

    solicitud = urllib.request.Request(
        url,
        data=datos,
        headers={

            "User-Agent":
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/154.0 Safari/537.36",

            "Accept":
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,*/*;q=0.8",

            "Accept-Language":
                "es-ES,es;q=0.9",

            "Referer":
                "https://duckduckgo.com/",

            "Origin":
                "https://duckduckgo.com",

            "Sec-Fetch-Site":
                "same-site",

            "Sec-Fetch-Mode":
                "navigate",

            "Sec-Fetch-Dest":
                "document",

            "Sec-Fetch-User":
                "?1"
        },
        method="POST"
    )

    with urllib.request.urlopen(
        solicitud,
        timeout=10
    ) as respuesta:

        return respuesta.read().decode(
            "utf-8",
            errors="ignore"
        )


def descargar_buscador_get(
    url
):

    solicitud = urllib.request.Request(
        url,
        headers={

            "User-Agent":
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/154.0 Safari/537.36",

            "Accept-Language":
                "es-ES,es;q=0.9"
        }
    )

    with urllib.request.urlopen(
        solicitud,
        timeout=10
    ) as respuesta:

        return respuesta.read().decode(
            "utf-8",
            errors="ignore"
        )


def extraer_texto_gemini(
    respuesta
):
    try:
        texto = getattr(
            respuesta,
            "text",
            None
        )

        if isinstance(
            texto,
            str
        ):
            texto = texto.strip()

            if texto:
                return texto

    except Exception:
        pass

    try:
        candidatos = getattr(
            respuesta,
            "candidates",
            []
        )

        partes = []

        for candidato in candidatos:
            contenido = getattr(
                candidato,
                "content",
                None
            )

            if not contenido:
                continue

            partes_contenido = getattr(
                contenido,
                "parts",
                []
            )

            for parte in partes_contenido:
                texto = getattr(
                    parte,
                    "text",
                    None
                )

                if texto:
                    partes.append(
                        str(texto)
                    )

        return " ".join(
            partes
        ).strip()

    except Exception:
        return ""


def mostrar_citas_gemini(
    respuesta
):
    try:
        citas = []
        vistos = set()

        candidatos = getattr(
            respuesta,
            "candidates",
            []
        )

        for candidato in candidatos:
            grounding = getattr(
                candidato,
                "grounding_metadata",
                None
            )

            if not grounding:
                continue

            chunks = getattr(
                grounding,
                "grounding_chunks",
                []
            )

            for chunk in chunks:
                fuente = getattr(
                    chunk,
                    "web",
                    None
                )

                if not fuente:
                    continue

                titulo = str(
                    getattr(
                        fuente,
                        "title",
                        ""
                    )
                    or ""
                ).strip()

                url = str(
                    getattr(
                        fuente,
                        "uri",
                        ""
                    )
                    or ""
                ).strip()

                clave = (
                    titulo,
                    url
                )

                if (
                    url
                    and clave not in vistos
                ):
                    vistos.add(
                        clave
                    )

                    citas.append({
                        "titulo":
                            titulo,
                        "url":
                            url
                    })

        if not citas:
            return

        print(
            "🔗 Fuentes de Google:"
        )

        for cita in citas[:8]:
            if cita["titulo"]:
                print(
                    f"   - {cita['titulo']}: "
                    f"{cita['url']}"
                )
            else:
                print(
                    f"   - {cita['url']}"
                )

    except Exception as e:
        print(
            f"⚠️ No se pudieron mostrar "
            f"las citas de Google: {e}"
        )


def preguntar_gemini_google(
    pregunta
):
    if gemini_client is None:
        return None

    pregunta = normalizar_consulta_busqueda(
        pregunta
    )

    if not pregunta:
        return None

    print(
        "🔎 Preguntando a Gemini con "
        "Google Search..."
    )

    prompt = (
        "Responde en español y de forma breve, "
        "directa y natural a la pregunta del usuario.\n"
        "Usa Google Search para comprobar la información "
        "actual antes de responder.\n"
        "Da prioridad a fuentes oficiales, tiendas "
        "oficiales y fuentes fiables.\n"
        "Si preguntas por precios, busca precios actuales "
        "en España y especifica claramente si es un precio "
        "desde, aproximado o de una tienda concreta.\n"
        "No inventes datos.\n"
        "No muestres URLs.\n"
        "No muestres razonamiento interno.\n"
        "Responde solamente con la respuesta final.\n\n"
        "Pregunta del usuario:\n"
        + pregunta
    )

    try:
        respuesta = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                tools=[
                    types.Tool(
                        google_search=types.GoogleSearch()
                    )
                ]
            )
        )

        texto = extraer_texto_gemini(
            respuesta
        )

        if not texto:
            print(
                "⚠️ Gemini no devolvió "
                "contenido final."
            )
            return None

        print(
            "✅ Respuesta obtenida de "
            "Gemini + Google Search."
        )

        mostrar_citas_gemini(
            respuesta
        )

        return texto.strip()

    except Exception as e:
        print(
            f"❌ ERROR GEMINI + GOOGLE SEARCH: "
            f"{repr(e)}"
        )

        return None


def buscar_internet(
    consulta
):

    consulta = normalizar_consulta_busqueda(
        consulta
    )

    print(
        f"🌐 Buscando en Internet: "
        f"{consulta}"
    )

    # --------------------------------------------------------
    # 1. GEMINI + GOOGLE SEARCH
    # --------------------------------------------------------

    respuesta_gemini = preguntar_gemini_google(
        consulta
    )

    if respuesta_gemini:
        return respuesta_gemini

    print(
        "🔁 Gemini no disponible o falló. "
        "Usando el buscador web de respaldo..."
    )

    # --------------------------------------------------------
    # 2. DUCKDUCKGO HTML POR POST
    # --------------------------------------------------------

    try:

        html = descargar_buscador_post(
            "https://html.duckduckgo.com/html/",
            consulta
        )

        parser = BuscadorHTML()

        parser.feed(
            html
        )

        resultados = parser.resultados[:5]

        if resultados:

            print(
                f"📄 {len(resultados)} "
                f"resultados obtenidos "
                f"(DuckDuckGo HTML)."
            )

            return construir_informacion_resultados(
                resultados
            )

    except Exception as e:

        print(
            f"⚠️ DuckDuckGo HTML falló: "
            f"{e}"
        )

    # --------------------------------------------------------
    # 2. DUCKDUCKGO LITE POR POST
    # --------------------------------------------------------

    try:

        print(
            "🔁 Probando DuckDuckGo Lite..."
        )

        html = descargar_buscador_post(
            "https://lite.duckduckgo.com/lite/",
            consulta
        )

        parser = BuscadorHTML()

        parser.feed(
            html
        )

        resultados = parser.resultados[:5]

        if resultados:

            print(
                f"📄 {len(resultados)} "
                f"resultados obtenidos "
                f"(DuckDuckGo Lite)."
            )

            return construir_informacion_resultados(
                resultados
            )

    except Exception as e:

        print(
            f"⚠️ DuckDuckGo Lite falló: "
            f"{e}"
        )

    # --------------------------------------------------------
    # 3. GOOGLE HTML
    # --------------------------------------------------------

    try:

        print(
            "🔁 Probando Google..."
        )

        parametros = urllib.parse.urlencode({

            "q":
                consulta,

            "hl":
                "es",

            "gl":
                "es",

            "gbv":
                "1",

            "num":
                "10"
        })

        url = (
            "https://www.google.com/search?"
            + parametros
        )

        html = descargar_buscador_get(
            url
        )

        parser = BuscadorGoogleHTML()

        parser.feed(
            html
        )

        resultados = parser.resultados[:5]

        if resultados:

            print(
                f"📄 {len(resultados)} "
                f"resultados obtenidos "
                f"(Google)."
            )

            return construir_informacion_resultados(
                resultados
            )

    except Exception as e:

        print(
            f"⚠️ Google falló: "
            f"{e}"
        )

    print(
        "❌ Ningún buscador devolvió resultados."
    )

    return None


def construir_informacion_resultados(
    resultados
):

    texto = ""

    for i, resultado in enumerate(
        resultados,
        start=1
    ):

        titulo = str(
            resultado.get(
                "titulo",
                ""
            )
        ).strip()

        url = str(
            resultado.get(
                "url",
                ""
            )
        ).strip()

        snippet = str(
            resultado.get(
                "snippet",
                ""
            )
        ).strip()

        texto += (
            f"Resultado {i}\n"
            f"Título: "
            f"{titulo}\n"
            f"URL: "
            f"{url}\n"
            f"Resumen: "
            f"{snippet}\n\n"
        )

    return texto.strip()


def normalizar_consulta_busqueda(
    consulta
):

    consulta = str(
        consulta
    ).strip()

    consulta = re.sub(
        r"\bplay\s+station\s+5\b",
        "PlayStation 5",
        consulta,
        flags=re.IGNORECASE
    )

    consulta = re.sub(
        r"\bplay\s+station\b",
        "PlayStation",
        consulta,
        flags=re.IGNORECASE
    )

    consulta = re.sub(
        r"\bRTX\s*(\d{4})\s*GB\b",
        r"RTX \1",
        consulta,
        flags=re.IGNORECASE
    )

    consulta = re.sub(
        r"\bRTX\s*(\d{4})\b",
        r"RTX \1",
        consulta,
        flags=re.IGNORECASE
    )

    consulta = re.sub(
        r"\bZodac\b",
        "Zotac",
        consulta,
        flags=re.IGNORECASE
    )

    consulta = re.sub(
        r"\s+",
        " ",
        consulta
    ).strip()

    return consulta


def extraer_respuesta_gemma(
    respuesta
):

    try:

        if not respuesta.choices:
            return ""

        mensaje = (
            respuesta.choices[0]
            .message
        )

        contenido = getattr(
            mensaje,
            "content",
            None
        )

        if isinstance(
            contenido,
            str
        ):

            contenido = contenido.strip()

            if contenido:
                return contenido

        elif isinstance(
            contenido,
            list
        ):

            partes = []

            for parte in contenido:

                if isinstance(
                    parte,
                    dict
                ):

                    texto = parte.get(
                        "text",
                        ""
                    )

                    if texto:
                        partes.append(
                            str(texto)
                        )

                else:

                    texto = getattr(
                        parte,
                        "text",
                        None
                    )

                    if texto:
                        partes.append(
                            str(texto)
                        )

            contenido = " ".join(
                partes
            ).strip()

            if contenido:
                return contenido

        razonamiento = getattr(
            mensaje,
            "reasoning_content",
            None
        )

        if razonamiento:

            print(
                "⚠️ Gemma devolvió razonamiento "
                "pero no contenido final."
            )

        return ""

    except Exception as e:

        print(
            f"⚠️ Error extrayendo respuesta: "
            f"{e}"
        )

        return ""


def extraer_datos_relevantes_web(
    informacion
):

    if not informacion:
        return []

    bloques = re.split(
        r"\n\s*\n",
        informacion.strip()
    )

    resultados = []

    for bloque in bloques:

        titulo = re.search(
            r"Título:\s*(.+)",
            bloque
        )

        resumen = re.search(
            r"Resumen:\s*(.*)",
            bloque
        )

        if not titulo:
            continue

        titulo_texto = (
            titulo.group(1)
            .strip()
        )

        resumen_texto = (
            resumen.group(1)
            .strip()
            if resumen
            else ""
        )

        texto = (
            titulo_texto
            + " "
            + resumen_texto
        )

        texto = re.sub(
            r"https?://\S+",
            "",
            texto
        )

        texto = re.sub(
            r"\s+",
            " ",
            texto
        ).strip()

        resultados.append(
            texto
        )

    return resultados


def normalizar_importe(
    precio_texto
):

    s = str(
        precio_texto
    ).strip()

    s = re.sub(
        r"(?i)euros?|EUR|€",
        "",
        s
    )

    s = re.sub(
        r"\s+",
        "",
        s
    )

    if not s:
        raise ValueError()

    if (
        ","
        in s
        and "."
        in s
    ):

        if (
            s.rfind(",")
            >
            s.rfind(".")
        ):

            s = (
                s.replace(
                    ".",
                    ""
                )
                .replace(
                    ",",
                    "."
                )
            )

        else:

            s = s.replace(
                ",",
                ""
            )

    elif "," in s:

        partes = s.rsplit(
            ",",
            1
        )

        if (
            len(partes[1]) <= 2
        ):

            s = (
                partes[0]
                + "."
                + partes[1]
            )

        else:

            s = s.replace(
                ",",
                ""
            )

    elif "." in s:

        partes = s.split(
            "."
        )

        if (
            len(partes) > 1
            and
            len(partes[-1]) == 3
            and
            all(
                parte.isdigit()
                for parte in partes
            )
        ):

            s = "".join(
                partes
            )

    return float(
        s
    )


def extraer_precios_web(
    informacion,
    consulta=""
):

    textos = extraer_datos_relevantes_web(
        informacion
    )

    if not textos:
        return []

    patron_numero_euro = re.compile(
        r"(?<![\d.,])"
        r"(?:"
        r"\d{1,3}(?:[.\s]\d{3})+"
        r"(?:,\d{1,2})?"
        r"|"
        r"\d+(?:[.,]\d{1,2})?"
        r")"
        r"\s*"
        r"(?:€|EUR|euros?)"
        r"(?!\w)",
        flags=re.IGNORECASE
    )

    patron_euro_numero = re.compile(
        r"(?:€|EUR|euros?)"
        r"\s*"
        r"(?:"
        r"\d{1,3}(?:[.\s]\d{3})+"
        r"(?:,\d{1,2})?"
        r"|"
        r"\d+(?:[.,]\d{1,2})?"
        r")"
        r"(?!\w)",
        flags=re.IGNORECASE
    )

    palabras_consulta = [

        palabra

        for palabra in re.findall(
            r"[a-z0-9áéíóúüñ]+",
            quitar_acentos(
                consulta.lower()
            )
        )

        if len(palabra) >= 3
    ]

    encontrados = []

    for indice, texto in enumerate(
        textos
    ):

        coincidencias = []

        coincidencias.extend(
            patron_numero_euro.finditer(
                texto
            )
        )

        coincidencias.extend(
            patron_euro_numero.finditer(
                texto
            )
        )

        texto_normalizado = quitar_acentos(
            texto.lower()
        )

        coincidencias_consulta = sum(
            1
            for palabra in palabras_consulta
            if palabra in texto_normalizado
        )

        for coincidencia in coincidencias:

            bruto = (
                coincidencia.group(0)
                .strip()
            )

            numero_original = re.sub(
                r"(?i)euros?|EUR|€",
                "",
                bruto
            ).strip()

            try:

                numero = normalizar_importe(
                    numero_original
                )

            except Exception:

                continue

            if not (
                1
                <= numero
                <= 10000
            ):

                continue

            relevancia = (
                coincidencias_consulta
                * 10
            )

            if re.search(
                r"\b(?:precio|precios|pvp|coste|costo|desde|plan|planes|mensual|mes)\b",
                texto_normalizado
            ):

                relevancia += 5

            encontrados.append({

                "precio":
                    numero,

                "precio_original":
                    numero_original,

                "texto":
                    texto,

                "indice":
                    indice,

                "relevancia":
                    relevancia
            })

    unicos = []

    vistos = set()

    for item in encontrados:

        clave = (
            round(
                item["precio"],
                2
            ),
            item["texto"],
            item["precio_original"]
        )

        if clave in vistos:
            continue

        vistos.add(
            clave
        )

        unicos.append(
            item
        )

    unicos.sort(
        key=lambda x: (
            -x["relevancia"],
            x["indice"]
        )
    )

    return unicos


def es_consulta_precio(
    consulta
):

    normalizado = quitar_acentos(
        consulta.lower()
    )

    patrones = [

        r"\bprecio\b",

        r"\bprecios\b",

        r"\bcuanto cuesta\b",

        r"\bcuanto vale\b",

        r"\bcoste\b",

        r"\bcosto\b"
    ]

    return any(
        re.search(
            patron,
            normalizado
        )
        for patron in patrones
    )


def generar_consulta_web_precio(
    consulta
):

    normalizada = quitar_acentos(
        consulta.lower()
    )

    if (
        "netflix"
        in normalizada
    ):

        return (
            "Netflix España planes precios mensual euros"
        )

    if (
        re.search(
            r"play\s*station\s*5",
            normalizada
        )
        or
        "ps5"
        in normalizada
    ):

        return (
            "PlayStation 5 España precio euros"
        )

    if (
        "rtx"
        in normalizada
        or
        "geforce"
        in normalizada
    ):

        return (
            consulta.strip()
            + " España euros"
        )

    return (
        consulta.strip()
        + " España euros"
    )


def combinar_informacion_web(
    original,
    adicional
):

    if not original:
        return adicional

    if not adicional:
        return original

    bloques = []

    for fuente in (
        original,
        adicional
    ):

        for bloque in re.split(
            r"\n\s*\n",
            fuente.strip()
        ):

            bloque = bloque.strip()

            if (
                bloque
                and bloque not in bloques
            ):

                bloques.append(
                    bloque
                )

    return "\n\n".join(
        bloques[:10]
    )


def limpiar_texto_web(
    texto
):

    texto = re.sub(
        r"https?://\S+",
        "",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    return texto


def crear_resumen_web_fallback(
    informacion,
    consulta=""
):

    if not informacion:

        return (
            "No he podido obtener "
            "información útil de Internet."
        )

    precios = extraer_precios_web(
        informacion,
        consulta
    )

    if precios:

        mejor = precios[0]

        precio_texto = (
            mejor["precio_original"]
            .strip()
        )

        texto_normalizado = quitar_acentos(
            mejor["texto"].lower()
        )

        if (
            "desde"
            in texto_normalizado
        ):

            return (
                f"He encontrado un precio "
                f"desde aproximadamente "
                f"{precio_texto} euros."
            )

        return (
            f"He encontrado un precio de "
            f"{precio_texto} euros."
        )

    textos = extraer_datos_relevantes_web(
        informacion
    )

    if not textos:

        return (
            "He encontrado información en Internet, "
            "pero no he podido interpretarla."
        )

    if es_consulta_precio(
        consulta
    ):

        return (
            "No he encontrado un precio "
            "concreto en los resultados."
        )

    primero = limpiar_texto_web(
        textos[0]
    )

    if len(primero) > 350:

        primero = (
            primero[:350]
            .rsplit(
                " ",
                1
            )[0]
            + "."
        )

    return (
        "He encontrado esta información: "
        + primero
    )


def respuesta_web_es_generica(
    texto
):

    if not texto:
        return True

    normalizado = quitar_acentos(
        texto.lower()
    )

    frases_genericas = [

        "varia segun la tienda",
        "varia segun las ofertas",
        "depende de la tienda",
        "depende de las ofertas",
        "puede variar segun",
        "los precios pueden variar",
        "el precio puede variar",
        "depende del modelo"
    ]

    for frase in frases_genericas:

        if frase in normalizado:
            return True

    return False


def preguntar_gemma_con_web(
    pregunta,
    consulta,
    informacion
):

    consulta = normalizar_consulta_busqueda(
        consulta
    )

    es_precio = es_consulta_precio(
        consulta
    )

    if es_precio:

        print(
            "💰 Consulta de precio detectada."
        )

        precios = extraer_precios_web(
            informacion,
            consulta
        )

        if not precios:

            print(
                "🔎 No se encontró un precio "
                "claro. Haciendo una búsqueda "
                "de refuerzo..."
            )

            consulta_refuerzo = (
                generar_consulta_web_precio(
                    consulta
                )
            )

            if (
                consulta_refuerzo.lower()
                != consulta.lower()
            ):

                informacion_adicional = (
                    buscar_internet(
                        consulta_refuerzo
                    )
                )

                if informacion_adicional:

                    informacion = (
                        combinar_informacion_web(
                            informacion,
                            informacion_adicional
                        )
                    )

                    precios = extraer_precios_web(
                        informacion,
                        consulta
                    )

                    if precios:

                        print(
                            "💰 Precio encontrado "
                            "tras la búsqueda de refuerzo."
                        )

        if precios:

            print(
                "🧾 Precio extraído directamente "
                "de los resultados."
            )

            return crear_resumen_web_fallback(
                informacion,
                consulta
            )

    prompt = (
        "Responde a la pregunta del usuario usando "
        "los resultados de Internet.\n\n"
        "IMPORTANTE:\n"
        "Debes dar una respuesta concreta siempre "
        "que los resultados permitan hacerlo.\n"
        "Si preguntan por un precio y aparece un precio "
        "en los resultados, dilo.\n"
        "Si aparecen varios precios, puedes indicar "
        "el precio más bajo encontrado y aclarar "
        "que depende del modelo o tienda.\n"
        "No respondas únicamente que depende de la tienda "
        "si los resultados contienen precios concretos.\n"
        "No inventes precios.\n"
        "No muestres URLs.\n"
        "No muestres razonamiento interno.\n"
        "No expliques el proceso de búsqueda.\n"
        "Responde únicamente con la respuesta final.\n"
        "Sé breve y natural.\n\n"
        "Pregunta:\n"
        + pregunta
        + "\n\n"
        "Resultados:\n"
        + informacion
    )

    print(
        "🧠 Pasando resultados a Gemma..."
    )

    try:

        respuesta = client.chat.completions.create(

            model=GEMMA_MODEL,

            messages=[

                {
                    "role":
                        "system",

                    "content":
                        "Eres Jarvis. "
                        "Debes responder en español "
                        "con datos concretos de los "
                        "resultados proporcionados. "
                        "No muestres razonamiento."
                },

                {
                    "role":
                        "user",

                    "content":
                        prompt
                }
            ],

            temperature=0.1,
            max_tokens=250
        )

        texto = extraer_respuesta_gemma(
            respuesta
        )

        print(
            "🔍 Respuesta Gemma web: "
            + repr(texto[:300])
        )

        if (
            texto
            and not respuesta_web_es_generica(
                texto
            )
        ):

            return texto

        if texto:

            print(
                "⚠️ Gemma respondió de forma "
                "demasiado genérica. "
                "Intentando obtener una respuesta concreta..."
            )

        else:

            print(
                "⚠️ Gemma web devolvió vacío. "
                "Haciendo segundo intento..."
            )

        prompt_directo = (
            "Contesta esta pregunta usando "
            "ÚNICAMENTE los datos de los resultados.\n\n"
            "Pregunta: "
            + pregunta
            + "\n\n"
            "Resultados:\n"
            + informacion
            + "\n\n"
            "Si hay un precio concreto en los resultados, "
            "dime ese precio.\n"
            "Si hay varios precios, dime el más bajo "
            "que aparezca y aclara que es el precio "
            "más bajo encontrado.\n"
            "Si no hay ningún precio, dilo claramente.\n"
            "No inventes.\n"
            "No muestres URLs.\n"
            "No muestres razonamiento.\n"
            "Responde en una sola frase."
        )

        respuesta = client.chat.completions.create(

            model=GEMMA_MODEL,

            messages=[

                {
                    "role":
                        "system",

                    "content":
                        "Extrae el dato concreto "
                        "que pide el usuario. "
                        "No hagas razonamiento visible."
                },

                {
                    "role":
                        "user",

                    "content":
                        prompt_directo
                }
            ],

            temperature=0,
            max_tokens=150
        )

        texto = extraer_respuesta_gemma(
            respuesta
        )

        print(
            "🔍 Segundo intento Gemma: "
            + repr(texto[:300])
        )

        if (
            texto
            and not respuesta_web_es_generica(
                texto
            )
        ):

            return texto

        print(
            "⚠️ Gemma no ha generado una "
            "respuesta suficientemente concreta."
        )

        print(
            "🧾 Extrayendo datos directamente "
            "de los resultados..."
        )

        return crear_resumen_web_fallback(
            informacion,
            consulta
        )

    except Exception as e:

        print(
            f"❌ ERROR REAL DE GEMMA WEB: "
            f"{repr(e)}"
        )

        return crear_resumen_web_fallback(
            informacion,
            consulta
        )


def extraer_consulta_busqueda(
    texto
):

    limpio = texto.lower().strip()

    patrones = [

        r"busca(?:me)?\s+en\s+google\s+(.+)",
        r"buscar\s+en\s+google\s+(.+)",
        r"búscame\s+en\s+google\s+(.+)",
        r"busta\s+en\s+google\s+(.+)",
        r"busca\s+en\s+internet\s+(.+)",
        r"buscar\s+en\s+internet\s+(.+)",
        r"búscame\s+en\s+internet\s+(.+)",
        r"busca\s+(.+)",
        r"buscar\s+(.+)",
        r"googlea\s+(.+)"
    ]

    for patron in patrones:

        coincidencia = re.match(
            patron,
            limpio
        )

        if not coincidencia:
            continue

        consulta = (
            coincidencia.group(1)
            .strip()
        )

        consulta = re.sub(
            r"^\s*y\s+dime\s+(?:qué|que)\s+es\s+",
            "",
            consulta
        )

        consulta = re.sub(
            r"^\s*y\s+dime\s+",
            "",
            consulta
        )

        consulta = re.sub(
            r"^\s*y\s+(?:qué|que)\s+es\s+",
            "",
            consulta
        )

        consulta = consulta.strip(
            " .,?!"
        )

        if consulta:
            return consulta

    return None


def ejecutar_busqueda_web(
    texto
):

    consulta = (
        extraer_consulta_busqueda(
            texto
        )
    )

    if not consulta:
        return None

    print(
        f"🔎 Consulta: {consulta}"
    )

    if (
        "google"
        in texto.lower()
    ):

        try:

            url = (
                "https://www.google.com/search?q="
                + urllib.parse.quote_plus(
                    consulta
                )
            )

            webbrowser.open(
                url
            )

        except Exception:
            pass

    informacion = buscar_internet(
        consulta
    )

    if not informacion:

        return (
            "He intentado buscarlo, "
            "pero no he podido obtener "
            "resultados de Internet."
        )

    return informacion


# ============================================================
# CONTROL DE VOLUMEN
# ============================================================

def obtener_control_volumen():

    try:

        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import (
            AudioUtilities,
            IAudioEndpointVolume
        )

        dispositivos = AudioUtilities.GetSpeakers()

        interfaz_directa = getattr(
            dispositivos,
            "EndpointVolume",
            None
        )

        if interfaz_directa is not None:
            return interfaz_directa

        activar = getattr(
            dispositivos,
            "Activate",
            None
        )

        if activar is not None:

            interfaz = activar(
                IAudioEndpointVolume._iid_,
                CLSCTX_ALL,
                None
            )

            return cast(
                interfaz,
                POINTER(
                    IAudioEndpointVolume
                )
            )

        dispositivo_real = getattr(
            dispositivos,
            "_dev",
            None
        )

        activar = getattr(
            dispositivo_real,
            "Activate",
            None
        )

        if activar is not None:

            interfaz = activar(
                IAudioEndpointVolume._iid_,
                CLSCTX_ALL,
                None
            )

            return cast(
                interfaz,
                POINTER(
                    IAudioEndpointVolume
                )
            )

        raise AttributeError(
            "No se encontró una interfaz de volumen compatible."
        )

    except Exception as e:

        print(
            f"⚠️ No se pudo acceder al "
            f"control exacto de volumen: {e}"
        )

        return None


def establecer_volumen_porcentaje(
    porcentaje
):

    try:

        porcentaje = float(
            porcentaje
        )

        porcentaje = max(
            0,
            min(
                100,
                porcentaje
            )
        )

        volumen = obtener_control_volumen()

        if volumen is None:
            return False

        volumen.SetMasterVolumeLevelScalar(
            porcentaje / 100.0,
            None
        )

        if porcentaje > 0:

            try:

                volumen.SetMute(
                    0,
                    None
                )

            except Exception:
                pass

        return True

    except Exception as e:

        print(
            f"⚠️ Error estableciendo volumen: {e}"
        )

        return False


def obtener_volumen_porcentaje():

    try:

        volumen = obtener_control_volumen()

        if volumen is None:
            return None

        actual = (
            volumen.GetMasterVolumeLevelScalar()
            * 100
        )

        return round(
            actual
        )

    except Exception as e:

        print(
            f"⚠️ Error obteniendo volumen: {e}"
        )

        return None


def desmutear_volumen():

    try:

        volumen = obtener_control_volumen()

        if volumen is None:
            return False

        volumen.SetMute(
            0,
            None
        )

        return True

    except Exception as e:

        print(
            f"⚠️ Error desmuteando volumen: {e}"
        )

        return False


def subir_volumen():

    try:

        volumen = obtener_control_volumen()

        if volumen is not None:

            actual = (
                volumen.GetMasterVolumeLevelScalar()
                * 100
            )

            nuevo = min(
                100,
                actual + 10
            )

            volumen.SetMasterVolumeLevelScalar(
                nuevo / 100,
                None
            )

            volumen.SetMute(
                0,
                None
            )

            return True

    except Exception as e:

        print(
            f"⚠️ Error con pycaw: {e}"
        )

    try:

        for _ in range(5):

            ctypes.windll.user32.keybd_event(
                0xAF,
                0,
                0,
                0
            )

            ctypes.windll.user32.keybd_event(
                0xAF,
                0,
                2,
                0
            )

        return True

    except Exception as e:

        print(
            f"⚠️ Error subiendo volumen: {e}"
        )

        return False


def bajar_volumen():

    try:

        volumen = obtener_control_volumen()

        if volumen is not None:

            actual = (
                volumen.GetMasterVolumeLevelScalar()
                * 100
            )

            nuevo = max(
                0,
                actual - 10
            )

            volumen.SetMasterVolumeLevelScalar(
                nuevo / 100,
                None
            )

            return True

    except Exception as e:

        print(
            f"⚠️ Error con pycaw: {e}"
        )

    try:

        for _ in range(5):

            ctypes.windll.user32.keybd_event(
                0xAE,
                0,
                0,
                0
            )

            ctypes.windll.user32.keybd_event(
                0xAE,
                0,
                2,
                0
            )

        return True

    except Exception as e:

        print(
            f"⚠️ Error bajando volumen: {e}"
        )

        return False


def silenciar_volumen():

    try:

        volumen = obtener_control_volumen()

        if volumen is not None:

            volumen.SetMute(
                1,
                None
            )

            return True

    except Exception as e:

        print(
            f"⚠️ Error con pycaw: {e}"
        )

    try:

        ctypes.windll.user32.keybd_event(
            0xAD,
            0,
            0,
            0
        )

        ctypes.windll.user32.keybd_event(
            0xAD,
            0,
            2,
            0
        )

        return True

    except Exception as e:

        print(
            f"⚠️ Error silenciando volumen: {e}"
        )

        return False


# ============================================================
# NÚMEROS
# ============================================================

UNIDADES_NUMERO = {

    "cero": 0,
    "uno": 1,
    "un": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9
}

DECENAS_NUMERO = {

    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "dieciséis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90
}

NUMEROS_DIRECTOS = {

    **UNIDADES_NUMERO,
    **DECENAS_NUMERO,

    "cien": 100,
    "ciento": 100,

    "veintiuno": 21,
    "veintidós": 22,
    "veintidos": 22,
    "veintitrés": 23,
    "veintitres": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiséis": 26,
    "veintiseis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29
}


def quitar_acentos(texto):

    return "".join(
        caracter
        for caracter in unicodedata.normalize(
            "NFD",
            texto
        )
        if unicodedata.category(
            caracter
        ) != "Mn"
    )


def numero_escrito_a_entero(texto):

    texto = quitar_acentos(
        texto.lower().strip()
    )

    if texto in NUMEROS_DIRECTOS:

        return NUMEROS_DIRECTOS[
            texto
        ]

    partes = re.split(
        r"\s+y\s+",
        texto
    )

    if len(partes) == 2:

        izquierda = partes[0].strip()
        derecha = partes[1].strip()

        if (
            izquierda in DECENAS_NUMERO
            and derecha in UNIDADES_NUMERO
        ):

            return (
                DECENAS_NUMERO[izquierda]
                + UNIDADES_NUMERO[derecha]
            )

    return None


def extraer_porcentaje_volumen(texto):

    texto_normalizado = quitar_acentos(
        texto.lower()
    )

    nombres = sorted(
        NUMEROS_DIRECTOS.keys(),
        key=len,
        reverse=True
    )

    patron_escrito = (
        r"\b(?:al|a|en)\s+"
        r"("
        + "|".join(
            re.escape(
                quitar_acentos(nombre)
            )
            for nombre in nombres
        )
        + r")"
        r"(?:\s+por\s+ciento|\s*%)?"
        r"\b"
    )

    coincidencia = re.search(
        patron_escrito,
        texto_normalizado
    )

    if coincidencia:

        valor = numero_escrito_a_entero(
            coincidencia.group(1)
        )

        if valor is not None:

            return max(
                0,
                min(
                    100,
                    valor
                )
            )

    patron_numero = (
        r"\b(?:al|a|en)\s*"
        r"(\d{1,3})"
        r"(?:\s*%|\s+por\s+ciento)?\b"
    )

    coincidencia = re.search(
        patron_numero,
        texto_normalizado
    )

    if coincidencia:

        valor = int(
            coincidencia.group(1)
        )

        return max(
            0,
            min(
                100,
                valor
            )
        )

    patron_volumen_directo = (
        r"\bvolumen\s+"
        r"(\d{1,3})"
        r"(?:\s*%|\s+por\s+ciento)?\b"
    )

    coincidencia = re.search(
        patron_volumen_directo,
        texto_normalizado
    )

    if coincidencia:

        valor = int(
            coincidencia.group(1)
        )

        return max(
            0,
            min(
                100,
                valor
            )
        )

    return None


def parece_comando_volumen(texto):

    texto_normalizado = quitar_acentos(
        texto.lower()
    )

    palabras_volumen = [
        "volumen",
        "bolumen"
    ]

    if any(
        palabra in texto_normalizado
        for palabra in palabras_volumen
    ):

        return True

    variantes = [

        "sube",
        "subir",
        "baja",
        "bajar",
        "silencia",
        "silenciar",
        "habilita",
        "habilitar",
        "desmutea",
        "desmutear",
        "mute",
        "mutear"
    ]

    for palabra in variantes:

        if palabra in texto_normalizado:
            return True

    return False


# ============================================================
# WINDOWS
# ============================================================

def bloquear_pc():

    try:

        resultado = ctypes.windll.user32.LockWorkStation()

        return resultado != 0

    except Exception as e:

        print(
            f"⚠️ Error bloqueando PC: {e}"
        )

        return False


def cerrar_aplicacion(nombre):

    nombre = normalizar_nombre_app(
        nombre
    )

    procesos = {

        "discord":
            ["Discord.exe"],

        "steam":
            ["steam.exe"],

        "chrome":
            ["chrome.exe"],

        "microsoft edge":
            ["msedge.exe"],

        "spotify":
            ["Spotify.exe"],

        "notepad":
            ["notepad.exe"],

        "bloc de notas":
            ["notepad.exe"],

        "paint":
            ["mspaint.exe"],

        "explorador de archivos":
            ["explorer.exe"]
    }

    ejecutables = procesos.get(
        nombre
    )

    if not ejecutables:

        nombre_proceso = re.sub(
            r"[^a-zA-Z0-9_-]",
            "",
            nombre
        )

        if not nombre_proceso:
            return False

        ejecutables = [
            nombre_proceso + ".exe"
        ]

    cerrado = False

    for ejecutable in ejecutables:

        try:

            resultado = subprocess.run(
                [
                    "taskkill",
                    "/IM",
                    ejecutable,
                    "/F"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if resultado.returncode == 0:
                cerrado = True

        except Exception as e:

            print(
                f"⚠️ Error cerrando {ejecutable}: {e}"
            )

    return cerrado


def solicitar_confirmacion(
    accion
):

    global confirmacion_pendiente

    confirmacion_pendiente = accion

    if accion == "apagar":

        return (
            "¿Confirmas que quieres apagar el PC?"
        )

    if accion == "reiniciar":

        return (
            "¿Confirmas que quieres reiniciar el PC?"
        )

    if accion.startswith(
        "cerrar:"
    ):

        nombre = accion.split(
            ":",
            1
        )[1]

        return (
            f"¿Confirmas que quieres cerrar {nombre}?"
        )

    confirmacion_pendiente = None

    return None


def procesar_confirmacion(
    texto
):

    global confirmacion_pendiente

    if not confirmacion_pendiente:
        return None

    texto = texto.lower().strip()

    texto = quitar_acentos(
        texto
    )

    texto = re.sub(
        r"[^\w\s]",
        " ",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    afirmativas = {

        "si",
        "confirmo",
        "confirmado",
        "hazlo",
        "adelante",
        "vale",
        "ok",
        "okay",
        "de acuerdo",
        "si hazlo",
        "si adelante",
        "si confirmo",
        "hazlo ya",
        "puedes hacerlo"
    }

    negativas = {

        "no",
        "cancela",
        "cancelar",
        "cancelalo",
        "dejalo",
        "no lo hagas",
        "no quiero",
        "anula",
        "anular"
    }

    if texto in afirmativas:

        accion = confirmacion_pendiente
        confirmacion_pendiente = None

        if accion == "apagar":

            hablar(
                "De acuerdo. Apagando el PC."
            )

            time.sleep(
                0.5
            )

            subprocess.Popen(
                [
                    "shutdown",
                    "/s",
                    "/t",
                    "0"
                ]
            )

            return "cerrar"

        if accion == "reiniciar":

            hablar(
                "De acuerdo. Reiniciando el PC."
            )

            time.sleep(
                0.5
            )

            subprocess.Popen(
                [
                    "shutdown",
                    "/r",
                    "/t",
                    "0"
                ]
            )

            return "cerrar"

        if accion.startswith(
            "cerrar:"
        ):

            nombre = accion.split(
                ":",
                1
            )[1]

            if cerrar_aplicacion(
                nombre
            ):

                return (
                    f"He cerrado {nombre}."
                )

            return (
                f"No he podido cerrar {nombre}."
            )

    if texto in negativas:

        confirmacion_pendiente = None

        return (
            "Acción cancelada."
        )

    return (
        "Necesito que me digas sí o no."
    )


def extraer_objetivo_cierre(
    texto
):

    texto = texto.lower().strip()

    patrones = [

        r"cierra\s+(?:la\s+)?(?:app|aplicación|aplicacion|programa)\s+(.+)",

        r"cerrar\s+(?:la\s+)?(?:app|aplicación|aplicacion|programa)\s+(.+)",

        r"cierra\s+(.+)",

        r"cerrar\s+(.+)"
    ]

    for patron in patrones:

        coincidencia = re.match(
            patron,
            texto
        )

        if coincidencia:

            objetivo = (
                coincidencia.group(1)
                .strip(" .,?!")
            )

            if objetivo:
                return objetivo

    return None


# ============================================================
# HERRAMIENTAS
# ============================================================

HERRAMIENTAS_JARVIS = {

    "estado_ram": {
        "descripcion":
            "Consulta el uso actual de la memoria RAM del PC.",
        "argumentos": {}
    },

    "estado_cpu": {
        "descripcion":
            "Consulta el porcentaje de uso actual de la CPU.",
        "argumentos": {}
    },

    "estado_gpu": {
        "descripcion":
            "Consulta GPU, porcentaje de uso, temperatura, VRAM y potencia si está disponible.",
        "argumentos": {}
    },

    "estado_pc": {
        "descripcion":
            "Consulta el estado general del PC incluyendo CPU, RAM y GPU.",
        "argumentos": {}
    },

    "obtener_hora": {
        "descripcion":
            "Obtiene la hora actual del ordenador.",
        "argumentos": {}
    },

    "obtener_tiempo": {
        "descripcion":
            "Consulta el tiempo actual y la previsión del día para una ciudad.",
        "argumentos": {
            "ciudad":
                "nombre de la ciudad"
        }
    },

    "obtener_volumen": {
        "descripcion":
            "Consulta el volumen actual del sistema en porcentaje.",
        "argumentos": {}
    },

    "subir_volumen": {
        "descripcion":
            "Sube el volumen del sistema.",
        "argumentos": {}
    },

    "bajar_volumen": {
        "descripcion":
            "Baja el volumen del sistema.",
        "argumentos": {}
    },

    "silenciar_volumen": {
        "descripcion":
            "Silencia el volumen del sistema.",
        "argumentos": {}
    },

    "desmutear_volumen": {
        "descripcion":
            "Quita el silencio del volumen del sistema.",
        "argumentos": {}
    },

    "establecer_volumen": {
        "descripcion":
            "Establece el volumen del sistema a un porcentaje.",
        "argumentos": {
            "porcentaje":
                "número entre 0 y 100"
        }
    },

    "abrir": {
        "descripcion":
            "Abre una aplicación, sitio web, URL o archivo.",
        "argumentos": {
            "objetivo":
                "nombre de aplicación, sitio web, URL o ruta"
        }
    },

    "buscar_internet": {
        "descripcion":
            "Busca información actual en Internet usando el buscador web de Jarvis.",
        "argumentos": {
            "consulta":
                "consulta que se debe buscar"
        }
    }
}


def obtener_descripcion_herramientas():

    texto = ""

    for nombre, datos in HERRAMIENTAS_JARVIS.items():

        texto += (
            f"- {nombre}: "
            f"{datos['descripcion']}\n"
        )

        if datos["argumentos"]:

            texto += (
                "  Argumentos: "
                + json.dumps(
                    datos["argumentos"],
                    ensure_ascii=False
                )
                + "\n"
            )

    return texto.strip()


def extraer_json_gemma(
    texto
):

    if not texto:
        return None

    texto = texto.strip()

    try:

        datos = json.loads(
            texto
        )

        if isinstance(
            datos,
            dict
        ):
            return datos

    except Exception:
        pass

    coincidencia = re.search(
        r"\{.*\}",
        texto,
        flags=re.DOTALL
    )

    if not coincidencia:
        return None

    try:

        datos = json.loads(
            coincidencia.group(0)
        )

        if isinstance(
            datos,
            dict
        ):
            return datos

    except Exception:
        pass

    return None


def decidir_herramienta(
    texto
):

    prompt = (
        "Debes decidir si para responder a la petición "
        "del usuario necesitas ejecutar una herramienta de Jarvis.\n\n"
        "Solo puedes elegir una herramienta de la lista proporcionada.\n"
        "No inventes herramientas.\n"
        "Si ninguna herramienta es necesaria, devuelve exactamente:\n"
        '{"tool":null,"args":{}}\n\n'
        "Si una herramienta es necesaria, devuelve únicamente "
        "JSON válido con este formato:\n"
        '{"tool":"nombre_herramienta","args":{}}\n\n'
        "No escribas explicaciones.\n"
        "No uses Markdown.\n\n"
        "Herramientas disponibles:\n"
        + obtener_descripcion_herramientas()
        + "\n\nPetición del usuario:\n"
        + texto
    )

    try:

        respuesta = client.chat.completions.create(

            model=GEMMA_MODEL,

            messages=[

                {
                    "role":
                        "system",

                    "content":
                        "Eres el router de herramientas "
                        "de Jarvis. Devuelve únicamente "
                        "JSON válido."
                },

                {
                    "role":
                        "user",

                    "content":
                        prompt
                }
            ],

            temperature=0,
            max_tokens=150
        )

        contenido = extraer_respuesta_gemma(
            respuesta
        )

        print(
            "🧰 Decisión de Gemma: "
            + repr(contenido)
        )

        datos = extraer_json_gemma(
            contenido
        )

        if not datos:
            return None

        herramienta = datos.get(
            "tool"
        )

        argumentos = datos.get(
            "args",
            {}
        )

        if herramienta is None:
            return None

        if herramienta not in HERRAMIENTAS_JARVIS:
            return None

        if not isinstance(
            argumentos,
            dict
        ):

            argumentos = {}

        return {
            "tool":
                herramienta,
            "args":
                argumentos
        }

    except Exception as e:

        print(
            f"⚠️ Error en router de herramientas: "
            f"{repr(e)}"
        )

        return None


def ejecutar_herramienta(
    nombre,
    argumentos
):

    if nombre not in HERRAMIENTAS_JARVIS:
        return None

    argumentos = (
        argumentos
        if isinstance(
            argumentos,
            dict
        )
        else {}
    )

    print(
        f"🛠️ Ejecutando herramienta: "
        f"{nombre} | {argumentos}"
    )

    if nombre == "estado_ram":
        return estado_ram()

    if nombre == "estado_cpu":
        return estado_cpu()

    if nombre == "estado_gpu":
        return estado_gpu()

    if nombre == "estado_pc":
        return estado_pc()

    if nombre == "obtener_hora":
        return obtener_hora()

    if nombre == "obtener_tiempo":

        ciudad = argumentos.get(
            "ciudad"
        )

        if not ciudad:

            return (
                "No se ha especificado "
                "ninguna ciudad."
            )

        return obtener_tiempo(
            str(ciudad)
        )

    if nombre == "obtener_volumen":

        volumen = obtener_volumen_porcentaje()

        if volumen is None:

            return (
                "No he podido consultar "
                "el volumen actual."
            )

        return (
            f"El volumen está al "
            f"{volumen} por ciento."
        )

    if nombre == "subir_volumen":

        if subir_volumen():

            return (
                "He subido el volumen."
            )

        return (
            "No he podido subir el volumen."
        )

    if nombre == "bajar_volumen":

        if bajar_volumen():

            return (
                "He bajado el volumen."
            )

        return (
            "No he podido bajar el volumen."
        )

    if nombre == "silenciar_volumen":

        if silenciar_volumen():

            return (
                "Volumen silenciado."
            )

        return (
            "No he podido silenciar el volumen."
        )

    if nombre == "desmutear_volumen":

        if desmutear_volumen():

            return (
                "Volumen activado."
            )

        return (
            "No he podido activar el volumen."
        )

    if nombre == "establecer_volumen":

        porcentaje = argumentos.get(
            "porcentaje"
        )

        try:

            porcentaje = float(
                porcentaje
            )

        except Exception:

            return (
                "El porcentaje de volumen "
                "no es válido."
            )

        if establecer_volumen_porcentaje(
            porcentaje
        ):

            return (
                f"Volumen establecido al "
                f"{int(porcentaje)} por ciento."
            )

        return (
            "No he podido establecer "
            "el volumen."
        )

    if nombre == "abrir":

        objetivo = argumentos.get(
            "objetivo"
        )

        if not objetivo:

            return (
                "No se ha especificado "
                "qué se debe abrir."
            )

        objetivo = str(
            objetivo
        ).strip()

        if abrir_objetivo(
            objetivo
        ):

            return (
                f"He abierto {objetivo}."
            )

        return (
            f"No he encontrado una aplicación "
            f"o sitio llamado {objetivo}."
        )

    if nombre == "buscar_internet":

        consulta = argumentos.get(
            "consulta"
        )

        if not consulta:

            return (
                "No se ha especificado "
                "qué buscar."
            )

        consulta = normalizar_consulta_busqueda(
            str(
                consulta
            ).strip()
        )

        informacion = buscar_internet(
            consulta
        )

        if not informacion:

            return (
                "No he podido obtener "
                "resultados de Internet."
            )

        return informacion

    return None


def responder_con_herramienta(
    pregunta,
    herramienta,
    resultado
):

    if herramienta == "buscar_internet":

        return str(
            resultado
        )

    prompt = (
        "Responde en español a la petición del usuario.\n"
        "La herramienta de Jarvis ya se ha ejecutado.\n"
        "Usa exclusivamente el resultado proporcionado "
        "para los datos que dependan de la herramienta.\n"
        "No inventes información.\n"
        "No hables de herramientas internas.\n"
        "Sé breve, directo y natural.\n"
        "Devuelve solamente la respuesta final.\n\n"
        "Petición:\n"
        + pregunta
        + "\n\n"
        "Resultado real:\n"
        + str(resultado)
    )

    try:

        respuesta = client.chat.completions.create(

            model=GEMMA_MODEL,

            messages=[

                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT
                },

                {
                    "role":
                        "user",

                    "content":
                        prompt
                }
            ],

            temperature=0.4,
            max_tokens=200
        )

        texto = extraer_respuesta_gemma(
            respuesta
        )

        if texto:
            return texto

    except Exception as e:

        print(
            f"⚠️ Error generando respuesta: "
            f"{repr(e)}"
        )

    return str(
        resultado
    )


def ejecutar_router_herramientas(
    texto
):

    decision = decidir_herramienta(
        texto
    )

    if not decision:
        return None

    herramienta = decision.get(
        "tool"
    )

    argumentos = decision.get(
        "args",
        {}
    )

    resultado = ejecutar_herramienta(
        herramienta,
        argumentos
    )

    if resultado is None:
        return None

    return responder_con_herramienta(
        texto,
        herramienta,
        resultado
    )


# ============================================================
# COMANDOS DIRECTOS
# ============================================================

def ejecutar_comando_directo(
    texto
):

    global esperando_ciudad_clima
    global confirmacion_pendiente

    texto_limpio = (
        texto.lower().strip()
    )

    cancelaciones = {
        "para",
        "parar",
        "párate",
        "parate",
        "cancela",
        "cancelar",
        "cancelalo",
        "cancelarlo",
        "déjalo",
        "dejalo"
    }

    texto_sin_acentos = quitar_acentos(
        texto_limpio
    )

    if texto_sin_acentos in {
        quitar_acentos(
            palabra
        )
        for palabra in cancelaciones
    }:

        if esperando_ciudad_clima:

            esperando_ciudad_clima = False

            return (
                "__CANCELAR_SILENCIOSO__"
            )

        if confirmacion_pendiente:

            confirmacion_pendiente = None

            return (
                "Acción cancelada."
            )

    if confirmacion_pendiente:

        return procesar_confirmacion(
            texto
        )

    if esperando_ciudad_clima:

        esperando_ciudad_clima = False

        ciudad = texto.strip(
            " .,?!"
        )

        if not ciudad:

            esperando_ciudad_clima = True

            return (
                "Dime la ciudad de la que "
                "quieres saber el tiempo."
            )

        return obtener_tiempo(
            ciudad
        )

    if (
        "bloquea el pc"
        in texto_limpio
        or
        "bloquea mi pc"
        in texto_limpio
        or
        "bloquea el ordenador"
        in texto_limpio
        or
        "bloquear pc"
        in texto_limpio
        or
        "bloquear el pc"
        in texto_limpio
    ):

        if bloquear_pc():

            return (
                "PC bloqueado."
            )

        return (
            "No he podido bloquear el PC."
        )

    texto_volumen = quitar_acentos(
        texto_limpio
    )

    porcentaje = extraer_porcentaje_volumen(
        texto_limpio
    )

    if (
        porcentaje is not None
        and parece_comando_volumen(
            texto_limpio
        )
    ):

        if establecer_volumen_porcentaje(
            porcentaje
        ):

            return (
                f"Volumen establecido al "
                f"{porcentaje} por ciento."
            )

        return (
            "No he podido establecer el volumen."
        )

    if (
        "sube el volumen"
        in texto_volumen
        or
        "subir el volumen"
        in texto_volumen
        or
        "sube volumen"
        in texto_volumen
        or
        "subir volumen"
        in texto_volumen
        or
        "mas volumen"
        in texto_volumen
    ):

        if subir_volumen():

            return (
                "He subido el volumen."
            )

        return (
            "No he podido cambiar el volumen."
        )

    if (
        "baja el volumen"
        in texto_volumen
        or
        "bajar el volumen"
        in texto_volumen
        or
        "baja volumen"
        in texto_volumen
        or
        "bajar volumen"
        in texto_volumen
        or
        "menos volumen"
        in texto_volumen
    ):

        if bajar_volumen():

            return (
                "He bajado el volumen."
            )

        return (
            "No he podido cambiar el volumen."
        )

    if (
        "habilita el volumen"
        in texto_volumen
        or
        "habilitar el volumen"
        in texto_volumen
        or
        "desmutea el volumen"
        in texto_volumen
        or
        "desmutear el volumen"
        in texto_volumen
        or
        "quita el mute"
        in texto_volumen
        or
        "quitar el mute"
        in texto_volumen
        or
        "activa el volumen"
        in texto_volumen
        or
        "activar el volumen"
        in texto_volumen
    ):

        if desmutear_volumen():

            return (
                "Volumen activado."
            )

        return (
            "No he podido activar el volumen."
        )

    if (
        "silencia el volumen"
        in texto_volumen
        or
        "silencia volumen"
        in texto_volumen
        or
        "silenciar volumen"
        in texto_volumen
        or
        "silencia"
        == texto_volumen.strip()
        or
        "mute"
        == texto_volumen.strip()
        or
        "mutear"
        == texto_volumen.strip()
    ):

        if silenciar_volumen():

            return (
                "Volumen silenciado."
            )

        return (
            "No he podido silenciar el volumen."
        )

    if (
        "apaga el pc"
        in texto_limpio
        or
        "apagar el pc"
        in texto_limpio
        or
        "apaga mi pc"
        in texto_limpio
        or
        "apagar mi pc"
        in texto_limpio
        or
        "apaga el ordenador"
        in texto_limpio
        or
        "apagar el ordenador"
        in texto_limpio
    ):

        return solicitar_confirmacion(
            "apagar"
        )

    if (
        "reinicia el pc"
        in texto_limpio
        or
        "reiniciar el pc"
        in texto_limpio
        or
        "reinicia mi pc"
        in texto_limpio
        or
        "reiniciar mi pc"
        in texto_limpio
        or
        "reinicia el ordenador"
        in texto_limpio
        or
        "reiniciar el ordenador"
        in texto_limpio
    ):

        return solicitar_confirmacion(
            "reiniciar"
        )

    objetivo_cierre = extraer_objetivo_cierre(
        texto
    )

    if objetivo_cierre:

        return solicitar_confirmacion(
            "cerrar:" + objetivo_cierre
        )

    objetivo = extraer_objetivo_apertura(
        texto
    )

    if objetivo:

        print(
            f"🔓 Intentando abrir: "
            f"{objetivo}"
        )

        if abrir_objetivo(
            objetivo
        ):

            return (
                f"He abierto {objetivo}."
            )

        return (
            f"No he encontrado una aplicación "
            f"o sitio llamado {objetivo}."
        )

    if extraer_consulta_busqueda(
        texto
    ):

        return ejecutar_busqueda_web(
            texto
        )

    if (
        "qué hora es"
        in texto_limpio
        or
        "que hora es"
        in texto_limpio
        or
        "dime la hora"
        in texto_limpio
    ):

        return obtener_hora()

    if (
        "estado del pc"
        in texto_limpio
        or
        "estado del ordenador"
        in texto_limpio
        or
        "cómo está el pc"
        in texto_limpio
        or
        "como está el pc"
        in texto_limpio
        or
        "cómo está mi pc"
        in texto_limpio
        or
        "como está mi pc"
        in texto_limpio
    ):

        return estado_pc()

    if (
        "ram"
        in texto_limpio
        or
        "memoria ram"
        in texto_limpio
    ):

        return estado_ram()

    if (
        "cpu"
        in texto_limpio
        or
        "procesador"
        in texto_limpio
    ):

        return estado_cpu()

    if (
        "gpu"
        in texto_limpio
        or
        "gráfica"
        in texto_limpio
        or
        "grafica"
        in texto_limpio
        or
        "tarjeta gráfica"
        in texto_limpio
        or
        "tarjeta grafica"
        in texto_limpio
        or
        "temperatura de la gráfica"
        in texto_limpio
        or
        "temperatura de la grafica"
        in texto_limpio
        or
        "temperatura de gpu"
        in texto_limpio
        or
        "temperatura gpu"
        in texto_limpio
    ):

        return estado_gpu()

    if (
        "tiempo"
        in texto_limpio
        or
        "clima"
        in texto_limpio
        or
        "temperatura"
        in texto_limpio
    ):

        ciudad = detectar_ciudad_clima(
            texto
        )

        if ciudad:

            return obtener_tiempo(
                ciudad
            )

        esperando_ciudad_clima = True

        return (
            "Dime la ciudad de la que "
            "quieres saber el tiempo."
        )

    return None


# ============================================================
# GEMMA NORMAL
# ============================================================

def preguntar_gemma(
    texto
):

    memoria.append({

        "role":
            "user",

        "content":
            texto
    })

    try:

        respuesta = client.chat.completions.create(

            model=GEMMA_MODEL,

            messages=memoria,

            temperature=0.5,

            max_tokens=200
        )

        respuesta_texto = extraer_respuesta_gemma(
            respuesta
        )

        if not respuesta_texto:

            print(
                "⚠️ Gemma devolvió una respuesta "
                "vacía. Reintentando..."
            )

            respuesta = client.chat.completions.create(

                model=GEMMA_MODEL,

                messages=[

                    {
                        "role":
                            "system",

                        "content":
                            SYSTEM_PROMPT
                    },

                    {
                        "role":
                            "user",

                        "content":
                            texto
                    }
                ],

                temperature=0.5,

                max_tokens=200
            )

            respuesta_texto = extraer_respuesta_gemma(
                respuesta
            )

        if not respuesta_texto:

            respuesta_texto = (
                "No he recibido una "
                "respuesta de Gemma."
            )

    except Exception as e:

        print(
            f"❌ ERROR GEMMA: "
            f"{repr(e)}"
        )

        respuesta_texto = (
            "Ha ocurrido un error "
            "al comunicarme con Gemma."
        )

    memoria.append({

        "role":
            "assistant",

        "content":
            respuesta_texto
    })

    guardar_memoria()

    return respuesta_texto


# ============================================================
# PROCESAR ORDEN
# ============================================================

def procesar_orden():

    archivo = grabar_hasta_silencio()

    texto = transcribir(
        archivo
    )

    if not texto:

        print(
            "No he entendido nada."
        )

        return "continuar"

    print(
        f"🗣️ Tú: {texto}"
    )

    texto_limpio = (
        texto.lower().strip()
    )

    if confirmacion_pendiente:

        resultado = ejecutar_comando_directo(
            texto
        )

        if resultado == "cerrar":
            return "cerrar"

        if resultado:

            print(
                f"🤖 Jarvis: {resultado}"
            )

            if resultado != "__CANCELAR_SILENCIOSO__":

                hablar(
                    resultado
                )

        return "continuar"

    orden_aprendizaje = (
        extraer_orden_aprendizaje(
            texto
        )
    )

    if orden_aprendizaje:

        frase, comando = orden_aprendizaje

        if aprender_comando(
            frase,
            comando
        ):

            respuesta = (
                "De acuerdo. Lo recordaré."
            )

            print(
                f"🤖 Jarvis: {respuesta}"
            )

            hablar(
                respuesta
            )

            return "continuar"

        respuesta = (
            "No he podido guardar "
            "ese aprendizaje."
        )

        print(
            f"🤖 Jarvis: {respuesta}"
        )

        hablar(
            respuesta
        )

        return "continuar"

    texto = aplicar_aprendizaje(
        texto
    )

    texto_limpio = (
        texto.lower().strip()
    )

    if (
        texto_limpio == "salir"
        or
        "apágate"
        in texto_limpio
        or
        "apagate"
        in texto_limpio
        or
        "cerrar jarvis"
        in texto_limpio
    ):

        hablar(
            "De acuerdo. Cerrando Jarvis."
        )

        return "cerrar"

    despedidas = [

        "adiós",
        "adios",
        "hasta luego",
        "hasta la próxima",
        "hasta la proxima",
        "nos vemos",
        "me voy",
        "y adiós",
        "y adios"
    ]

    if any(
        frase in texto_limpio
        for frase in despedidas
    ):

        hablar(
            "Hasta luego."
        )

        return "esperar"

    resultado = ejecutar_comando_directo(
        texto
    )

    if resultado:

        if resultado == "cerrar":
            return "cerrar"

        if resultado == "__CANCELAR_SILENCIOSO__":
            return "continuar"

        print(
            f"🤖 Jarvis: {resultado}"
        )

        hablar(
            resultado
        )

        return "continuar"

    respuesta_herramienta = (
        ejecutar_router_herramientas(
            texto
        )
    )

    if respuesta_herramienta:

        memoria.append({

            "role":
                "user",

            "content":
                texto
        })

        memoria.append({

            "role":
                "assistant",

            "content":
                respuesta_herramienta
        })

        guardar_memoria()

        print(
            f"🤖 Jarvis: "
            f"{respuesta_herramienta}"
        )

        hablar(
            respuesta_herramienta
        )

        return "continuar"

    respuesta = preguntar_gemma(
        texto
    )

    print(
        f"🤖 Jarvis: {respuesta}"
    )

    hablar(
        respuesta
    )

    return "continuar"


# ============================================================
# INICIO
# ============================================================

print()

print(
    "=" * 50
)

print(
    "                  JARVIS"
)

print(
    "=" * 50
)

print()

print(
    "Di «Jarvis» para despertarme."
)


try:

    pynvml.nvmlInit()

    try:

        handle = pynvml.nvmlDeviceGetHandleByIndex(
            0
        )

        nombre = pynvml.nvmlDeviceGetName(
            handle
        )

        if isinstance(
            nombre,
            bytes
        ):

            nombre = nombre.decode()

        print(
            f"🎮 GPU detectada: {nombre}"
        )

    except Exception:
        pass

except Exception as e:

    print(
        f"⚠️ NVIDIA Management "
        f"no disponible: {e}"
    )


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

try:

    while True:

        esperar_jarvis()

        hablar(
            "Dime."
        )

        while True:

            resultado = procesar_orden()

            if resultado == "esperar":
                break

            if resultado == "cerrar":
                raise SystemExit

except KeyboardInterrupt:

    print(
        "\nCerrando Jarvis..."
    )

except SystemExit:

    pass

finally:

    interrumpir_voz.set()

    escuchando_para.clear()

    try:

        sd.stop()

    except Exception:
        pass

    try:

        pynvml.nvmlShutdown()

    except Exception:
        pass

    guardar_memoria()

    print(
        "Jarvis cerrado."
    )