# 0. conferir broker
docker compose exec s6 python3 -c "from confluent_kafka.admin import AdminClient; from ods_bus.client import bootstrap_servers; print(sorted(AdminClient({'bootstrap.servers': bootstrap_servers()}).list_topics(timeout=10).topics))"

# 0. gerar léxico
docker compose run --rm s6 python3 -m s6_synthesis.lexicon.build_lexicon --features-dir dataset/features_v2 --output-dir lexicon

# T5 / 2.1 monitorar
docker compose exec s6 python3 -m ods_bus.debug tail ods.i4.sign ods.i4.facial_marker --max 30 --timeout 120

# 2.2 mock do I4 sem câmera
docker compose run --rm i4 python3 -m pipeline.publica_amostra --parquet lexicon/lexicon-Aluno.parquet --confidence 0.90

# 2.4 simular o A6
docker compose exec s6 python3 -m ods_bus.debug synth ALUNO BANCO

# 3. testes offline
docker compose run --rm i4 python3 -m pipeline.model.verificar_modelo --model model_v3_e100s2.pt --features-dir dataset/features_v2


# execução
docker compose up -d server s6 i4    # sobe
py .\webcam.py --url ws://localhost:18080/ws/video --preview   # Windows
docker compose logs -f i4            # acompanha
docker compose stop                  # ao terminar (crédito do GCP)





# comando docker
docker compose up -d --build 
docker compose logs -f i4.

docker system prune -a --volumes -f 
docker image prune -f

df -h 
docker system dfdocker builder prune -f



docker compose ps              # containers do projeto e status
docker compose logs -f         # logs de todos, ao vivo (Ctrl+C sai sem derrubar)
docker compose logs -f i4      # logs só do serviço i4
docker compose logs --tail 50 s6   # últimas 50 linhas do s6


docker compose stop            # para os containers, mantém eles e os volumes
docker compose down            # para e remove containers e a rede (volumes ficam)
docker compose down -v         # também apaga os volumes (cuidado: apaga `trajectories`)

docker compose up -d           # sobe em segundo plano (sem rebuild)
docker compose up -d --build   # rebuilda antes de subir (use depois de mudar código)


docker compose restart i4              # reinicia sem rebuildar
docker compose up -d --build i4        # rebuilda e recria só o i4
docker compose stop s6                 # para só o s6
docker compose start s6                # sobe de novo o que foi parado

docker compose exec i4 bash                        # shell no container em execução
docker compose exec s6 python -m ods_bus.debug tail ods.i4.sign --max 20
docker compose run --rm s6 bash                    # container novo e descartável