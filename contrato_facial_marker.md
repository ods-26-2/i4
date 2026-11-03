## Evento `ods.i4.facial_marker` (EV-SINAL)

Publicado por I4 para A6 quando um marcador facial gramatical é reconhecido.

| campo | tipo | descrição |
|---|---|---|
| `marker` | string | um de `INTERROGATIVA_QU`, `INTERROGATIVA_SN`, `NEGACAO`, `AFIRMACAO` (conjunto fechado do LEXICON `lexicon-a6-1.3`) |
| `confidence` | float 0..1 | confiança do detector (0.5 = no limiar) |
| `start_ms` / `end_ms` | int | epoch ms absoluto, mesma base dos keypoints; `captured_at` = `start_ms` |

- Ausência de marcador **não gera evento** (NEUTRO não é publicado; confirmar com A6 se querem receber).
- Detector atual: regras sobre cabeça (yaw/pitch) e sobrancelhas. Validado só em NEGACAO; AFIRMACAO, SN e QU são de baixa confiança até haver dados rotulados.
- Exemplo: `examples/exemplo_facial_marker.json`.