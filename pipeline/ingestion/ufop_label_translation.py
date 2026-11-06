"""
ufop_label_translation.py
===========================
Tradução inglês -> português dos rótulos dos datasets da UFOP
(LIBRAS-UFOP de pares mínimos e LIBRAS-FACIAL-EXPRESIONS-BSL), pra manter
o mesmo idioma do resto do léxico (`Aluno`, `Banco`, ...) e do contrato
de I4 (CONTRATO_I4.md, campo `sinal`).

Confirmado com o time: traduções ambíguas resolvidas como
Battle->Batalha, Record->Recorde, Stamp->Selo, Screw_up->Errar,
Takes_a_little_care/Takes_great_care->Pouco_Cuidado/Muito_Cuidado.
"""

from __future__ import annotations

UFOP_SIGN_LABELS: dict[str, str] = {
    "Always": "Sempre",
    "Analyze": "Analisar",
    "Arrive": "Chegar",
    "Battle": "Batalha",
    "Bicycle": "Bicicleta",
    "Bodybuilding": "Musculacao",
    "Build": "Construir",
    "Calumny": "Calunia",
    "Close": "Fechar",
    "Cock": "Galo",
    "Day_1": "Dia_1",
    "Day_2": "Dia_2",
    "Day_3": "Dia_3",
    "Day_before_yesterday": "Anteontem",
    "Defend": "Defender",
    "Effort": "Esforco",
    "Exchange": "Trocar",
    "Fat": "Gordo",
    "Hen": "Galinha",
    "Idea": "Ideia",
    "Interact": "Interagir",
    "Learn": "Aprender",
    "Loss": "Perda",
    "Love": "Amor",
    "Nobody": "Ninguem",
    "Not": "Nao",
    "Nothing": "Nada",
    "Open": "Abrir",
    "Physical_Education": "Educacao_Fisica",
    "Physiotherapy": "Fisioterapia",
    "Record": "Recorde",
    "Run_fast": "Correr_Rapido",
    "Run_slow": "Correr_Devagar",
    "Safe": "Seguro",
    "Screw_up": "Errar",
    "Slip": "Escorregar",
    "Stamp": "Selo",
    "Strong": "Forte",
    "Strong_rain": "Chuva_Forte",
    "Strong_wind": "Vento_Forte",
    "Takes_a_little_care": "Pouco_Cuidado",
    "Takes_great_care": "Muito_Cuidado",
    "Talk": "Conversar",
    "Television": "Televisao",
    "Thin": "Magro",
    "Weak": "Fraco",
    "Weak_rain": "Chuva_Fraca",
    "Weak_wind": "Vento_Fraco",
    "Week": "Semana",
    "Week_1": "Semana_1",
    "Win": "Ganhar",
    "Work": "Trabalhar",
    "Year_1": "Ano_1",
    "Year_2": "Ano_2",
    "Year_3": "Ano_3",
    "Yesterday": "Ontem",
}

# Expressões faciais (não-manuais) — dataset separado dos sinais acima.
UFOP_FACIAL_EXPRESSION_LABELS: dict[str, str] = {
    "0_Angry": "Bravo",
    "1_Laugh": "Rir",
    "2_Surprise": "Surpresa",
    "3_Yawn": "Bocejar",
    "4_Cry": "Chorar",
    "5_Full": "Cheio",
    "6_Agree": "Concordar",
    "7_Empty": "Vazio",
}


def translate_sign(english_label: str) -> str:
    try:
        return UFOP_SIGN_LABELS[english_label]
    except KeyError:
        raise KeyError(
            f"Sinal '{english_label}' não tem tradução cadastrada em UFOP_SIGN_LABELS"
        ) from None


def translate_facial_expression(english_label: str) -> str:
    try:
        return UFOP_FACIAL_EXPRESSION_LABELS[english_label]
    except KeyError:
        raise KeyError(
            f"Expressão '{english_label}' não tem tradução cadastrada em "
            "UFOP_FACIAL_EXPRESSION_LABELS"
        ) from None
