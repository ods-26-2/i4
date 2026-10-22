# Contrato de saída de I4 — pra S6 e A6

Este documento descreve o payload que I4 publica no barramento (evento
`ods.i4.keypoints · sign · facial_marker`, INV-2 na arquitetura),
consumido por **S6** (trajetória/síntese) e **A6** (Libras). A
integração de verdade com o broker (B3) ainda não existe neste
repositório — isso é responsabilidade de quem opera o barramento. O que
tem aqui é o **payload pronto**, gerado pelo pipeline de I4, no formato
em que vai ser publicado.

Três exemplos reais (gerados pelo modelo treinado, `model_v4.pt`) estão
em `examples/*.json`.

## Quando é publicado

Uma mensagem por **janela de sinal reconhecida** (não por frame) — ou
seja, I4 junta os frames de um sinal (ao vivo: da memória
compartilhada; aqui nos exemplos: de um vídeo inteiro), classifica, e
publica um payload só com a classificação + todos os keypoints da
janela.

## Forma do payload

```jsonc
{
  "sign": {
    "sinal": "Aluno",            // rótulo do sinal reconhecido
    "confianca": 0.90,           // confiança do classificador, 0-1
    "inicio_ms": 0,              // sempre 0 (início da janela)
    "fim_ms": 1280,              // duração da janela em ms
    "sinalizador_id": "Sinalizador02"  // opcional, se souber quem é
  },
  "keypoints": [                 // um item por frame da janela
    {
      "pose":  [{"x":.., "y":.., "z":.., "visibility":..}, ...],  // 33 pontos
      "left_hand_global":  [{"x":.., "y":.., "z":..}, ...],       // 21 pontos
      "right_hand_global": [...],                                 // 21 pontos
      "face": [...],                                               // 92 pontos (reduzido)
      "left_hand_local":  [...],                                   // 21 pontos
      "right_hand_local": [...],                                   // 21 pontos
      "left_hand_velocity":  [...],                                // 21 pontos
      "right_hand_velocity": [...]                                 // 21 pontos
    },
    ...
  ],
  "marker_layout": [              // definição estável dos marcadores acima
    {"name": "pose", "offset": 0, "num_points": 33, "dims_per_point": 4, "point_dims": ["x","y","z","visibility"]},
    {"name": "left_hand_global", "offset": 132, "num_points": 21, "dims_per_point": 3, "point_dims": ["x","y","z"]},
    ...
  ],
  "frame_timestamps_ms": [0, 40, 80, ...]   // 1 timestamp por item de `keypoints`, mesmo índice
}
```

## O que é cada marcador

| marcador | pontos | o que é |
|---|---|---|
| `pose` | 33 | corpo inteiro (MediaPipe Pose), com `visibility` |
| `left_hand_global` / `right_hand_global` | 21 cada | mãos, no referencial do corpo (centralizado/escalado pelos ombros) — é a posição da mão em relação ao corpo |
| `face` | 92 | sobrancelhas/olhos/boca (subconjunto semântico, não a malha densa) — são os marcadores faciais **brutos**; não tem classificação de expressão em cima disso ainda (ver "Fora de escopo" abaixo) |
| `left_hand_local` / `right_hand_local` | 21 cada | a mesma mão, mas normalizada pelo próprio pulso — isola o **formato da mão** (handshape) da posição |
| `left_hand_velocity` / `right_hand_velocity` | 21 cada | delta frame-a-frame da mão (referencial global) — a **trajetória/movimento** da mão |

Todas as coordenadas são normalizadas (adimensionais, escala relativa
ao corpo/mão, não pixel). `marker_layout` é a fonte da verdade dos
offsets — não assumam a ordem/tamanho fixos, leiam dali.

## Fora de escopo nesta entrega

- **Classificação de expressão facial**: `face` traz os pontos brutos
  (sobrancelha, olho, boca), mas não tem nenhum modelo classificando
  "cara de pergunta", "cara de negação" etc. em cima disso ainda — falta
  dataset próprio pra isso.
- **Canal de comando `ods.i4.lexicon_capture.command/.reply`** (A6 → I4,
  pra autoria de novos sinais no léxico): não implementado neste
  pipeline ainda.
- **Trajetória/transição** (a saída de S6 pra A6): não é
  responsabilidade de I4 — S6 que deriva isso a partir de `keypoints` +
  `frame_timestamps_ms`.

## Como gerar mais exemplos

```bash
python -m pipeline.pipeline --video caminho\do\video.mp4 --model model_v4.pt --full-output
```

Sem `--full-output`, imprime só `sign` (uso normal/produção, quando
quem chama já vai ler os keypoints separadamente do barramento). Em
produção (câmera ao vivo via shm), é o mesmo contrato, só que sem
`--video` — `pipeline.LibrasPipeline.run()`.
