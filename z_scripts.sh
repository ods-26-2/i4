# rodar server.exe na pasta src
./src/server
./sh.sh


# execução dos abaixo rodar na pasta raiz

# ativar webcam
py .\websocket.py --url ws://localhost:18080/ws/video --preview

# escutar o barraemnto
python3 -m ods_bus.debug tail ods.i4.sign --max 5 --timeout 120
python3 -m ods_bus.debug tail ods.i4.keypoints ods.i4.sign ods.i4.facial_marker --max 300 --timeout 120

# rodar "servidor de leitura dos frames"
python3 -m pipeline.live.ao_vivo --model model_v3_e100s2.pt 
python3 -m pipeline.live.ao_vivo --model model_v3_e100s2.pt --expr-model expression_model.pt

python3 -m s6_synthesis.bus_service --lexicon-dir lexicon --trajectory-dir trajectories --lexicon-version lexicon-a6-1.3

python3 -m ods_bus.debug tail ods.s6.synthesis.reply --from-start --max 20 --timeout 60