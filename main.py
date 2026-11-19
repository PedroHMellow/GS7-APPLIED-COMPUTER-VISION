import os
import sys
import json
import base64
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

from utility.box_format import box_resize, draw_box, box_time_stamp
from utility.stream_format import format_stream
from utility.classes import class_classify, build_message

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


VIDEO_URL = os.getenv("VIDEO_URL", "https://www.youtube.com/watch?v=zIIIr0ZFQY4")

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "chave")
ROBOFLOW_URL = "https://serverless.roboflow.com"
MODEL_ID = "pedestrian-light-crossing-uoz6o/1"

INTERVALO_ANALISE = 0.5     # segundos entre análises
CONFIANCA_MIN = 0.50        # ignora detecções abaixo disso (0 a 1)
MAX_RESOLUTION = 720            # altura máxima do stream do YouTube
LARGURA_JANELA = 960        # largura máxima das janelas exibidas
REPETIR_VIDEO = False       # True = reinicia o vídeo quando acabar
SALVAR_DETECCOES = False    # True = também salva cada frame detectado em disco
PASTA_DETECCOES = "deteccoes"

# Evidências: com SALVAR_ETAPAS=1 no .env, salva a imagem de cada etapa do
# processamento (frame original, JPEG, entrada 640x640, saída bruta, filtrada)
# dos primeiros MAX_ETAPAS frames que tiverem alguma detecção.
SALVAR_ETAPAS = os.getenv("SALVAR_ETAPAS", "0") == "1"
PASTA_ETAPAS = os.path.join("evidencia", "etapas")
MAX_ETAPAS = int(os.getenv("MAX_ETAPAS", "8"))
TAMANHO_MODELO = 640        # o modelo foi treinado com "Stretch to 640x640"

JANELA_VIDEO = "Video"
JANELA_DETECCAO = "Deteccoes"

## CLIENTE ROBOFLOW
CLIENT = InferenceHTTPClient(
    api_url=ROBOFLOW_URL,
    api_key=ROBOFLOW_API_KEY,
).configure(InferenceConfiguration(
    api_key_transport="header"  # autenticação por header (inference v1.5.0+)
))

## Estado compartilhado entre a thread principal e a thread de análise
_lock = threading.Lock()
_ultimo_anotado = {"img": None, "versao": 0}
_classes_vistas = set()
_etapas_salvas = 0
_ultimas_classes = frozenset()


## EVIDÊNCIAS (salva as transformações intermediárias de um frame)
def salvar_etapas(frame, numero_frame: int, tempo_video: float, resultado, deteccoes):
    global _etapas_salvas
    pasta = os.path.join(PASTA_ETAPAS, f"frame_{numero_frame:05d}")
    os.makedirs(pasta, exist_ok=True)

    # Etapa 1 - frame decodificado (matriz BGR uint8)
    cv2.imwrite(os.path.join(pasta, "1_frame_original.png"), frame)

    # Etapa 3 - mesma codificação do inference-sdk (cv2.imencode .jpg + base64)
    ok, jpeg = cv2.imencode(".jpg", frame)
    jpeg_bytes = jpeg.tobytes()
    with open(os.path.join(pasta, "3_frame_enviado.jpg"), "wb") as f:
        f.write(jpeg_bytes)
    tamanho_b64 = len(base64.b64encode(jpeg_bytes))
    decodificado = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    diferenca = cv2.absdiff(frame, decodificado)
    mapa = cv2.applyColorMap(
        cv2.convertScaleAbs(cv2.cvtColor(diferenca, cv2.COLOR_BGR2GRAY), alpha=30),
        cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(pasta, "3_diferenca_jpeg.png"), mapa)

    # Etapa 4 - reprodução do pré-processamento do servidor (Stretch to 640x640)
    entrada = cv2.resize(frame, (TAMANHO_MODELO, TAMANHO_MODELO),
                         interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(pasta, "4_entrada_modelo_640x640.png"), entrada)

    # Etapa 5 - resposta bruta do modelo (todas as detecções, sem filtro)
    brutas = [{
        "x": p["x"], "y": p["y"], "width": p["width"], "height": p["height"],
        "classe": str(p.get("class", "")), "confianca": float(p.get("confidence", 0)),
        "cor": class_classify(str(p.get("class", ""))),
    } for p in resultado.get("predictions", [])]
    cv2.imwrite(os.path.join(pasta, "5_saida_bruta.png"), draw_box(frame, brutas))
    with open(os.path.join(pasta, "5_resposta_modelo.json"), "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    # Etapa 6 - só o que passou no filtro de confiança + mensagem final
    filtrado = draw_box(frame, deteccoes)
    cv2.imwrite(os.path.join(pasta, "6_apos_filtro.png"), filtrado)

    resumo = [
        f"frame: {numero_frame}  tempo: {box_time_stamp(tempo_video)} ({tempo_video:.2f} s)",
        f"etapa 1: shape={frame.shape} dtype={frame.dtype} bytes={frame.nbytes}",
        f"etapa 3: jpeg={len(jpeg_bytes)} bytes "
        f"({frame.nbytes / len(jpeg_bytes):.1f}x menor) base64={tamanho_b64} caracteres "
        f"diferenca_media={diferenca.mean():.2f} diferenca_max={int(diferenca.max())}",
        f"etapa 4: {frame.shape[1]}x{frame.shape[0]} -> {TAMANHO_MODELO}x{TAMANHO_MODELO} "
        f"(escala x={TAMANHO_MODELO / frame.shape[1]:.2f}, y={TAMANHO_MODELO / frame.shape[0]:.2f})",
        f"etapa 5: {len(brutas)} deteccoes brutas, tempo de inferencia={resultado.get('time')}",
    ] + [f"    {d['classe']} {d['confianca']:.2f} x={d['x']} y={d['y']} "
         f"w={d['width']} h={d['height']}" for d in brutas] + [
        f"etapa 6: {len(deteccoes)} apos filtro >= {CONFIANCA_MIN} -> \"{build_message(deteccoes)}\"",
    ]
    with open(os.path.join(pasta, "resumo.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(resumo) + "\n")

    _etapas_salvas += 1
    print(f"    (evidências do frame {numero_frame} salvas em {pasta})")


## ANÁLISE (roda em thread separada para não travar o vídeo)
def analisar(frame, tempo_video: float, numero_frame: int = 0):
    try:
        resultado = CLIENT.infer(frame, model_id=MODEL_ID)
    except Exception as e:
        print(f"[{box_time_stamp(tempo_video)}] Erro ao consultar o Roboflow: {e}")
        return

    deteccoes = []
    for p in resultado.get("predictions", []):
        classe = str(p.get("class", ""))
        conf = float(p.get("confidence", 0))

        if classe not in _classes_vistas:
            _classes_vistas.add(classe)
            print(f"(classe do modelo encontrada: '{classe}')")

        if conf < CONFIANCA_MIN:
            continue

        deteccoes.append({
            "x": p["x"], "y": p["y"], "width": p["width"], "height": p["height"],
            "classe": classe, "confianca": conf, "cor": class_classify(classe),
        })

    carimbo = box_time_stamp(tempo_video)
    mensagem = build_message(deteccoes)

    # salva quando o conjunto de classes detectadas muda (evita frames repetidos)
    global _ultimas_classes
    classes_brutas = frozenset(str(p.get("class")) for p in resultado.get("predictions", []))
    if SALVAR_ETAPAS and _etapas_salvas < MAX_ETAPAS and classes_brutas \
            and classes_brutas != _ultimas_classes:
        _ultimas_classes = classes_brutas
        try:
            salvar_etapas(frame, numero_frame, tempo_video, resultado, deteccoes)
        except Exception as e:
            print(f"    (falha ao salvar evidências: {e})")

    if not deteccoes:
        print(f"[{carimbo}] {mensagem}")
        return

    anotado = box_resize(draw_box(frame, deteccoes), LARGURA_JANELA)
    cv2.putText(anotado, f"Video {carimbo}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Publica o frame e imprime a mensagem juntos, para saírem ao mesmo tempo
    with _lock:
        _ultimo_anotado["img"] = anotado
        _ultimo_anotado["versao"] += 1
        print(f"[{carimbo}] {mensagem}")

    if SALVAR_DETECCOES:
        os.makedirs(PASTA_DETECCOES, exist_ok=True)
        nome = f"deteccao_{int(tempo_video * 1000):08d}.jpg"
        cv2.imwrite(os.path.join(PASTA_DETECCOES, nome), anotado)


def main():
    if ROBOFLOW_API_KEY in ("", "chave"):
        print("\t>Defina a variável ROBOFLOW_API_KEY (ou um .env) com a sua chave do Roboflow.")
        sys.exit(1)

    if VIDEO_URL == "":
        print("\t>Variável VIDEO_URL não foi definida corretamente. Insira seu vídeo de preferência do youtube ou elimine a variável para utilizar o vídeo padrão")
        sys.exit(1)

    source = format_stream(VIDEO_URL, MAX_RESOLUTION)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Não foi possível abrir o vídeo:", VIDEO_URL)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 30.0
    intervalo_frame = 1.0 / fps

    cv2.namedWindow(JANELA_VIDEO, cv2.WINDOW_AUTOSIZE)
    janela_deteccao_aberta = False

    executor = ThreadPoolExecutor(max_workers=1)
    futuro = None
    proxima_analise = time.time()
    proximo_frame = time.time()
    frames_lidos = 0
    versao_exibida = 0

    print("Rodando... (Q ou ESC para sair)\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if REPETIR_VIDEO:
                    cap.release()
                    cap = cv2.VideoCapture(source)
                    frames_lidos = 0
                    continue
                print("\nVídeo terminou.")
                break

            frames_lidos += 1
            tempo_video = frames_lidos / fps

            # --- análise a cada INTERVALO_ANALISE s (se a anterior já terminou) ---
            agora = time.time()
            if agora >= proxima_analise and (futuro is None or futuro.done()):
                futuro = executor.submit(analisar, frame.copy(), tempo_video, frames_lidos)
                proxima_analise = agora + INTERVALO_ANALISE

            # --- janela do vídeo ---
            cv2.imshow(JANELA_VIDEO, box_resize(frame, LARGURA_JANELA))

            # --- janela única de detecções (só atualiza quando há frame novo) ---
            with _lock:
                versao = _ultimo_anotado["versao"]
                img = _ultimo_anotado["img"]
            if img is not None and versao != versao_exibida:
                cv2.imshow(JANELA_DETECCAO, img)
                janela_deteccao_aberta = True
                versao_exibida = versao

            # --- ritmo do vídeo (reprodução em tempo real) ---
            proximo_frame += intervalo_frame
            espera = proximo_frame - time.time()
            if espera < -1.0:            # ficou muito atrasado: ressincroniza
                proximo_frame = time.time()
                espera = 0
            tecla = cv2.waitKey(max(1, int(espera * 1000))) & 0xFF
            if tecla in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(JANELA_VIDEO, cv2.WND_PROP_VISIBLE) < 1:
                break
            if janela_deteccao_aberta and \
                    cv2.getWindowProperty(JANELA_DETECCAO, cv2.WND_PROP_VISIBLE) < 1:
                # usuário fechou a janela de detecções: ela reabre na próxima detecção
                janela_deteccao_aberta = False
                versao_exibida = -1
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        cap.release()
        cv2.destroyAllWindows()
        print("Encerrado.")


if __name__ == "__main__":
    main()