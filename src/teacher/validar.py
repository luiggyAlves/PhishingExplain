# Validadores da resposta do professor: só entra no treino o que passa em todos eles

import re
import unicodedata

from pydantic import ValidationError

from src.schema import Resposta

MARCADOR = re.compile(r"\[(EMAIL|LINK)\]")
# "classificação informada", "rótulo fornecido", "classificação que foi dada", "conforme indicado"...
MENCIONA_ROTULO = re.compile(
    r"\b(classifica[çc][ãa]o|r[óo]tulo)\s+(\w+\s+){0,2}(informad|fornecid|recebid|indicad|passad|dad)[ao]\b"
    r"|\b(conforme|como)\s+(foi\s+)?(informad|indicad)o\b",
    re.IGNORECASE,
)


def normal(texto: str) -> str:
    """Texto comparável: NFC, espaços colapsados, sem diferenciar maiúsculas."""
    return " ".join(unicodedata.normalize("NFC", texto).split()).casefold()


def validar(texto_json: str, email: str, rotulo: str) -> list[str]:
    """Lista de erros da resposta (vazia = aprovada). `email` é o texto de formatar_email."""
    try:
        r = Resposta.model_validate_json(texto_json)
    except ValidationError as e:
        return [f"contrato: {'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()]
    texto = normal(email)
    return (
        [f"classificação {r.classificacao!r} difere do rótulo {rotulo!r}"] * (r.classificacao != rotulo)
        + [f"trecho fora do e-mail: {i.trecho!r}" for i in r.indicadores if normal(i.trecho) not in texto]
        + [f"trecho só com marcador: {i.trecho!r}" for i in r.indicadores if not re.search(r"\w", MARCADOR.sub("", i.trecho))]
        + ["explicação menciona a classificação dada"] * bool(MENCIONA_ROTULO.search(r.explicacao))
    )
