############################################################################
#  COLINHA ODS — I4 / S6 / barramento
#  Convenção: [WIN] = PowerShell no Windows · [WSL] = terminal do WSL/Linux
#  Cada bloco "T#" é um terminal separado. Ctrl+C encerra todos
#  (não deixe nada rodando: gasta crédito do GCP).
############################################################################


############################################################################
# 0. ANTES DE TUDO (uma vez por terminal [WSL])
############################################################################
# Endereço do broker NÃO é fixo no código. Padrão = 34.66.235.91:9093 (nuvem, teste).
# Quando o Kafka for local, é só trocar aqui:
export ODS_KAFKA_BOOTSTRAP=34.66.235.91:9093

# Conferir que o broker responde e quais tópicos existem:
python3 -c "from confluent_kafka.admin import AdminClient; from ods_bus.client import bootstrap_servers; print(sorted(AdminClient({'bootstrap.servers': bootstrap_servers()}).list_topics(timeout=10).topics))"

# Só se o léxico ainda não existe (ou as features mudaram):
python3 -m s6_synthesis.lexicon.build_lexicon --features-dir dataset/features_v2 --output-dir lexicon


############################################################################
# 1. EXECUÇÃO REAL  (webcam -> server -> shm -> I4 -> Kafka -> S6)
#    Subir NESTA ordem. O server só aceita UMA conexão de vídeo por vez.
############################################################################

# --- T1 [WSL] server C++ (detecta pessoa+rosto, escreve /dev/shm/frames_cam)
# Precisa rodar DENTRO de src/: os modelos MobileNetSSD e a pasta "capturas" são relativos.
cd src
./server                      # porta 18080, rota /ws/video
./sh.sh                       # (o que o sh.sh faz? deixei como estava)
cd ..

# --- T2 [WIN] webcam -> JPEG -> WebSocket (na raiz do repo)
py .\webcam.py --url ws://localhost:18080/ws/video --preview
#   opções: --camera 0 --width 640 --height 480 --fps 25 --quality 85
#   (q na janela do preview fecha; feche antes de abrir outro cliente)

# --- T3 [WSL] I4: lê a shm, segmenta por pausa, reconhece o sinal e publica
#   publica: ods.i4.keypoints (1 por quadro), ods.i4.sign, ods.i4.facial_marker
# a) só sinais
python3 -m pipeline.live.ao_vivo --model model_v3_e100s2.pt
# b) sinais + expressões por modelo treinado
python3 -m pipeline.live.ao_vivo --model model_v3_e100s2.pt --expr-model expression_model.pt
# c) sinais + expressões por regras (sem treino)
python3 -m pipeline.live.ao_vivo --model model_v3_e100s2.pt --expr-rules padrao
#    ou com limiares calibrados: --expr-rules limiares.json
#   ajustes: --gap 1.0 (s sem frame = fim do sinal) --min-ms 500 --max-s 6.0 --device cpu

# --- T4 [WSL] S6: atende ods.s6.synthesis.command e responde em .reply
python3 -m s6_synthesis.bus_service --lexicon-dir lexicon --trajectory-dir trajectories --lexicon-version lexicon-a6-1.3
#   opções: --group-id S6-synthesis --offset-reset latest --transition-frames 8 --fps 25
#   o A6 precisa enxergar a pasta trajectories/ (trajectory_ref = s6://trajectories/xxx.json)

# --- T5 [WSL] (opcional) monitorar o que o I4 está publicando
python3 -m ods_bus.debug tail ods.i4.sign ods.i4.facial_marker --max 30 --timeout 120


############################################################################
# 2. TESTES DO BARRAMENTO  (não precisam de câmera nem de server)
############################################################################

# ---------- 2.1 LER / VALIDAR ----------
# O tail valida envelope, schema x tópico, nº de pontos dos keypoints
# (33 corpo / 21 mãos / 92 rosto) e sequence repetida/faltando.
# Cada execução usa group.id novo: não mexe no offset de ninguém.

# só o que chegar de agora em diante
python3 -m ods_bus.debug tail ods.i4.sign --max 5 --timeout 120
python3 -m ods_bus.debug tail ods.i4.keypoints ods.i4.sign ods.i4.facial_marker --max 300 --timeout 120

# histórico do tópico (desde o início)
python3 -m ods_bus.debug tail ods.i4.sign --from-start --max 20 --timeout 30
python3 -m ods_bus.debug tail ods.s6.synthesis.reply --from-start --max 20 --timeout 60
python3 -m ods_bus.debug tail ods.s6.synthesis.command --from-start --max 20 --timeout 60

# ---------- 2.2 PUBLICAR MOCK DO I4 (sinal + keypoints, sem câmera) ----------
# Deixe o tail (2.1) rodando em outro terminal antes.
# Usa um .parquet de features_v2 e o mesmo I4BusPublisher do ao_vivo.
python3 -m pipeline.publica_amostra --features-dir dataset/features_v2
python3 -m pipeline.publica_amostra --parquet dataset/features_v2/XXXX.parquet --confidence 0.9 --source-id cam0 --size 1280x720

# ---------- 2.3 PUBLICAR MOCK DE facial_marker ----------
python3 - <<'EOF'
import time
from ods_bus.client import BusProducer, ensure_topics, topic_for
from pipeline.bus_publisher import I4BusPublisher
from pipeline.schema.facial_marker import FacialMarker

ensure_topics([topic_for("ods.i4.facial_marker")])
bus = I4BusPublisher(BusProducer(), source_id="cam0")
now = int(time.time() * 1000)
# marcador, confiança, início e fim em epoch ms; NEUTRO não é publicado
bus.publish_facial_marker(FacialMarker("INTERROGATIVA_SN", 0.80, now, now + 1500))
EOF

# ---------- 2.4 SIMULAR O A6 PEDINDO SÍNTESE AO S6 ----------
# Precisa do S6 rodando (T4). Envia o comando, espera a reply por correlation_id,
# valida o envelope (perfil RPL) e imprime o payload.
python3 -m ods_bus.debug synth ALUNO BANCO --lexicon-version lexicon-a6-1.3
python3 -m ods_bus.debug synth ALUNO BANCO CINCO --speed 1.5 --marker INTERROGATIVA_SN
python3 -m ods_bus.debug synth ALUNO GLOSA_INEXISTENTE      # testa unknown_signs
python3 -m ods_bus.debug synth ALUNO --lexicon-version lexicon-x  # deve dar UNKNOWN_LEXICON_VERSION
#   opções: --speed 1.0  --marker <MARCADOR> (cobre todas as glosas)  --timeout 20

# ---------- 2.5 TESTE PONTA A PONTA SEM CÂMERA ----------
# T-a: S6 rodando        -> python3 -m s6_synthesis.bus_service --lexicon-dir lexicon --trajectory-dir trajectories --lexicon-version lexicon-a6-1.3
# T-b: tail              -> python3 -m ods_bus.debug tail ods.i4.sign ods.i4.keypoints ods.s6.synthesis.reply --max 100 --timeout 90
# T-c: publicar mock I4  -> python3 -m pipeline.publica_amostra --features-dir dataset/features_v2
# T-d: pedir síntese     -> python3 -m ods_bus.debug synth ALUNO BANCO


############################################################################
# 3. TESTES OFFLINE (sem barramento)
############################################################################

# checkpoint carrega e classifica? (acertos aqui NÃO medem generalização:
# o modelo final viu esses vídeos no treino)
python3 -m pipeline.model.verificar_modelo --model model_v3_e100s2.pt --features-dir dataset/features_v2

# pipeline inteiro sobre um vídeo (sem shm/server), imprime o I4Output completo
python3 -m pipeline.pipeline --model model_v3_e100s2.pt --video caminho.mp4 --full-output

# expressões por regras sobre um vídeo seu (começar com ~2 s de rosto neutro e parado)
python3 -m pipeline.model.expression_rules video meu_video.mp4 --rules limiares.json

# calibrar limiares das regras (em clipes SEM expressão)
python3 -m pipeline.model.expression_rules calibrate --features-dir dataset/features_v2 --write limiares.json

# testar a câmera crua (WSL/Linux, V4L2)
python3 z_testeCamera.py