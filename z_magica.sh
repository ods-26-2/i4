# rodar server.exe na pasta src
./src/server

# escutar o barraemnto
python3 -m ods_bus.debug tail ods.i4.sign --max 5 --timeout 120

# rodar "servidor de leitura dos frames"
python3 -m pipeline.live.ao_vivo --model model_v3_e100s2.pt 

# mandar websocket
python3 websocket.py 