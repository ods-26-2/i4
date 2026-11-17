# pipeline/live/segmenter.py
import numpy as np

class SignSegmenter:
    """Recebe (vetor de features, ts_ms) por frame; devolve (rows, ts) quando um sinal fecha."""

    def __init__(self, start_ms=150, end_ms=500, pre_ms=200,
                 min_ms=300, max_ms=6000, margin=0.15, fallback_y=1.2):
        self.start_ms, self.end_ms, self.pre_ms = start_ms, end_ms, pre_ms
        self.min_ms, self.max_ms = min_ms, max_ms
        self.margin, self.fallback_y = margin, fallback_y
        self.reset()

    def reset(self):
        self.active = False
        self.rows, self.ts = [], []          # trecho em andamento
        self.pre = []                        # pré-roll: [(row, ts)]
        self.up_since = None                 # início da sequência "mão levantada"
        self.down_since = None               # início da sequência "mão abaixada"

    def _hand_up(self, row) -> bool:
        pose = row[:132].reshape(33, 4)
        if not pose.any():
            return False                     # pose não detectada = não ativo
        wrists = pose[[15, 16], 1]
        hips = pose[[23, 24], 1]
        hip_y = hips.mean() if hips.any() else self.fallback_y
        return bool((wrists < hip_y - self.margin).any())

    def feed(self, row, ts):
        up = self._hand_up(row)

        if not self.active:
            self.pre.append((row, ts))
            self.pre = [(r, t) for r, t in self.pre if ts - t <= self.pre_ms]
            if up:
                self.up_since = self.up_since or ts
                if ts - self.up_since >= self.start_ms:
                    self.active = True
                    self.rows = [r for r, _ in self.pre]
                    self.ts = [t for _, t in self.pre]
                    self.pre, self.down_since = [], None
            else:
                self.up_since = None
            return None

        self.rows.append(row); self.ts.append(ts)
        if up:
            self.down_since = None
        else:
            self.down_since = self.down_since or ts
        fim = self.down_since is not None and ts - self.down_since >= self.end_ms
        estouro = ts - self.ts[0] >= self.max_ms
        if fim or estouro:
            rows, tss = self.rows, self.ts
            if fim:                          # tira a cauda parada
                keep = [i for i, t in enumerate(tss) if t <= self.down_since]
                rows, tss = [rows[i] for i in keep], [tss[i] for i in keep]
            self.reset()
            return (rows, tss) if tss and tss[-1] - tss[0] >= self.min_ms else None
        return None