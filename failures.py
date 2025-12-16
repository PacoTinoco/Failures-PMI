import streamlit as st
import pandas as pd
import re
import unicodedata
from collections import Counter
from io import BytesIO

# =========================
# Configuración general
# =========================
st.set_page_config(page_title="Clasificador de pérdidas", layout="wide")
st.title("Clasificación de Principales Pérdidas y Sub‑palabras (CSV/XLSX)")

# =========================
# Utilidades
# =========================
def strip_accents(s: str) -> str:
    """Elimina acentos/diacríticos."""
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

def normalize_text(text: str) -> str:
    """
    Normaliza: minúsculas, sin acentos, solo [a-z0-9 espacio], colapsa espacios.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", strip_accents(text.lower()))).strip()

def discover_subprefixes(pairs, seed, top_m=10, min_prefix_len=3, stopwords=None):
    """
    (Opcional) Sugerir prefijos frecuentes cercanos a una 'seed' observados en comentarios.
    """
    seed_norm = normalize_text(seed)
    bag = []
    for _, comment in pairs:
        tokens = normalize_text(comment).split()
        if seed_norm in tokens:
            for t in tokens:
                if t != seed_norm and len(t) >= min_prefix_len and (stopwords is None or t not in stopwords):
                    bag.append(t[:min_prefix_len])
    return [p for p, _ in Counter(bag).most_common(top_m)]

def count_matches(tokens, seeds_to_prefixes):
    """
    Devuelve conteos por seed. Suma por token si seed in tok o algún prefijo in tok.
    (Si quieres evitar sobre-conteo por token, cambia 'any(p in tok' a 'tok.startswith(p)' o regex de palabra completa).
    """
    counts = {seed: 0 for seed in seeds_to_prefixes}
    for seed, prefixes in seeds_to_prefixes.items():
        for tok in tokens:
            if seed in tok or any(p in tok for p in prefixes):
                counts[seed] += 1
    return counts

def enforce_unique_subs(seeds_to_prefixes, pairs):
    """
    Asigna cada prefijo al seed con el que más co-ocurre en comentarios.
    seeds_to_prefixes: dict { seed: [prefijos...] } -> retorna dict con propiedad única.
    """
    pref_counts = {}
    for seed, prefs in seeds_to_prefixes.items():
        for p in prefs:
            pref_counts.setdefault(p, {})
            pref_counts[p].setdefault(seed, 0)

    for _, comment in pairs:
        tokens = normalize_text(comment).split()
        for p in pref_counts:
            if any(p in tok for tok in tokens):
                for seed in seeds_to_prefixes:
                    if p in seeds_to_prefixes[seed]:
                        pref_counts[p][seed] += 1

    owner = {p: max(pref_counts[p], key=pref_counts[p].get) for p in pref_counts}
    unique = {s: [] for s in seeds_to_prefixes}
    for p, s in owner.items():
        unique[s].append(p)
    return unique

def style_ties(df, tie_mask):
    """Estilo rojo para filas con 'Empate'."""
    def _row_style(row):
        return ['background-color: #ffb3b3'] * len(row) if tie_mask.loc[row.name] else [''] * len(row)
    return df.style.apply(_row_style, axis=1)

# =========================
# Estado inicial
# =========================
if "pairs" not in st.session_state:
    st.session_state.pairs = []           # list[(Duration, Comment)]
if "seeds" not in st.session_state:
    st.session_state.seeds = []           # list[str] (normalizados)
if "sub_prefixes" not in st.session_state:
    st.session_state.sub_prefixes = {}    # dict {seed: [prefijos...]}

# --- SANEAMIENTO DE ESTADO (clave para tu error) ---
def _sanitize_state():
    # Si por corridas anteriores quedó como 'set', conviértelo a dict por seed
    if isinstance(st.session_state.get("sub_prefixes"), set):
        set_vals = list(st.session_state["sub_prefixes"])
        seeds = st.session_state.get("seeds", [])
        # Si hay seeds, reparte el set a todas; si no, deja dict vacío
        st.session_state["sub_prefixes"] = {s: set_vals[:] for s in seeds} if seeds else {}
    # Si no es dict, fuerza dict vacío
    if not isinstance(st.session_state.get("sub_prefixes"), dict):
        st.session_state["sub_prefixes"] = {}
    # Normaliza seeds por si cambiaron títulos
    if "seeds" in st.session_state and st.session_state["seeds"]:
        st.session_state["seeds"] = [normalize_text(s) for s in st.session_state["seeds"]]

_sanitize_state()

# =========================
# Sidebar: Reset de estado
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Utilidades")
    if st.button("🔁 Resetear estado"):
        st.session_state.clear()
        st.experimental_rerun()

# =========================
# Configuración por máquina
# =========================
# Recomendado: definir 'subpalabras' por seed (dict). Si no tienes aún, deja {} (vacío).
machine_defaults = {
    "KDF-7": {
        "principales": ['Empalme', 'Mov', 'buffer', 'Canal bloq', 'cuchillas' ],  
        "subpalabras": {'empal', 'movim', 'lleno', 'caps', 'cuch', 'ator', 'bobin', 'canal', 'papel', 'hm'}   
    },

    "KDF-8": {
        "principales": ['Suciedad pva', 'Empalme', 'Batea Atorada', 'Polvo ODM', 'Filtro atorado en tambor', 'HCI'],  
        "subpalabras": {'pva', 'ato', 'tamb', 'filt', 'rue', 'gom', 'caj', 'temp'}},

    "KDF-9": {"principales": ['Suciedad', 'Batea', 'uña', 'banda', 'Canal bloq', 'buffer' ], 
              "subpalabras": {'lanz', 'charo', 'inser', 'lleno', 'bloq', 'canal', 'hm', 'cap'}},

    "KDF-10": {"principales": ['plug', 'gomero', 'limpieza', 'cola', 'enlace', 'Xepics', 'Papel'], 
               "subpalabras": {'plu', 'sucie', 'gome', 'tap', 'debaj', 'cola', 'hcf', 'fr', 'atasc', 'ator'}},

    "KDF-11": {"principales": ['varilla', 'filtro', 'empalme', 'hci', 'hcf', 'midas', 'cap'], 
               "subpalabras": {'limp', 'var', 'pap', 'rued', 'hcf', 'elev', 'resp', 'empal', 'tamb', 'bob'}},

    "KDF-17": {"principales": ['Filtro atorado', 'Varilla ab', 'Capsula ator', 'Empalme', 'cap cae', 'mov', 'batea'], 
               "subpalabras": {'filt', 'tamb', 'caps', 'bob', 'mov', 'batea'}}
}
STOPWORDS = {"de","la","el","y","en","por","se","al","a","no","lo","otro","otros"}

# =========================
# Paso 1: Subir CSV/XLSX (encabezados en Fila 9)
# =========================
st.markdown("### 1) Sube el archivo CSV/XLSX con columnas `Duration` y `Operator Comment` (encabezados en fila 9)")
uploaded_file = st.file_uploader("Selecciona el archivo", type=["csv", "xlsx"])
df = None

# Fila del encabezado (1-based = 9) -> 0-based = 8
HEADER_ROW_0_BASED = 8

if uploaded_file:
    fname = uploaded_file.name.lower()
    try:
        if fname.endswith(".csv"):
            df = pd.read_csv(uploaded_file, header=HEADER_ROW_0_BASED)
        elif fname.endswith(".xlsx"):
            xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
            sheet = st.selectbox("Hoja de Excel:", xls.sheet_names, index=0)
            df = pd.read_excel(uploaded_file, sheet_name=sheet, engine="openpyxl", header=HEADER_ROW_0_BASED)
        # Limpia posibles espacios en nombres de columna
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")

    if df is not None:
        st.write("Vista previa:")
        st.dataframe(df.head(10))

        if "Duration" not in df.columns or "Operator Comment" not in df.columns:
            st.error("El archivo debe tener columnas 'Duration' y 'Operator Comment' en la fila 9.")
        else:
            # Limpieza y preparación
            df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
            df = df.dropna(subset=["Duration", "Operator Comment"])
            df = df[df["Duration"] > 0]
            df["Operator Comment"] = df["Operator Comment"].astype(str).str.strip()
            df = df[df["Operator Comment"] != ""]
            st.session_state.pairs = list(zip(df["Duration"], df["Operator Comment"]))
            st.success(f"Filas válidas: {len(st.session_state.pairs)}")

# =========================
# Paso 2: Selección de máquina y principales pérdidas
# =========================
if st.session_state.pairs:
    st.markdown("### 2) Selecciona la máquina y principales pérdidas")
    machine = st.selectbox("Máquina:", list(machine_defaults.keys()))
    default_seeds_raw = machine_defaults[machine]["principales"]
    # Normaliza seeds
    default_seeds = [normalize_text(s) for s in default_seeds_raw]
    seeds_text = st.text_input("Principales pérdidas (puedes agregar más):", value=", ".join(default_seeds))

    if st.button("Cargar principales pérdidas"):
        seeds = [normalize_text(s) for s in seeds_text.split(",") if s.strip()]
        st.session_state.seeds = seeds

        # Inicializa SIEMPRE dict vacío por seed
        st.session_state.sub_prefixes = {s: [] for s in seeds}

        # Precarga subpalabras desde machine_defaults (solo si están mapeadas por seed)
        pre = machine_defaults[machine].get("subpalabras", {})
        if isinstance(pre, dict):
            # Normaliza claves/valores del dict de precarga
            pre_norm = {normalize_text(k): [normalize_text(p) for p in v] for k, v in pre.items()}
            for s in seeds:
                if s in pre_norm:
                    st.session_state.sub_prefixes[s] = pre_norm[s]

# --- Re-sanitiza por si cambió algo en los pasos anteriores ---
_sanitize_state()

# =========================
# Paso 3: Editar/agregar sub‑palabras
# =========================
if st.session_state.seeds:
    st.markdown("### 3) Edita/agrega sub‑palabras manualmente")
    with st.form("form_subs"):
        edited = {}
        for s in st.session_state.seeds:
            # Aquí ya garantizamos que sub_prefixes es dict
            default_val = ", ".join(st.session_state.sub_prefixes.get(s, []))
            edited[s] = st.text_input(f"Sub‑palabras para **{s}**:", value=default_val)
        saved = st.form_submit_button("Guardar cambios")
        if saved:
            final_subs = {s: [normalize_text(p) for p in edited[s].split(",") if p.strip()] for s in st.session_state.seeds}
            final_subs = enforce_unique_subs(final_subs, st.session_state.pairs)
            st.session_state.sub_prefixes = final_subs
            st.success("Sub‑palabras guardadas.")

# =========================
# Paso 4: Generar análisis
# =========================
if st.button("Generar análisis"):
    if not st.session_state.pairs:
        st.warning("Primero sube un archivo válido.")
    elif not st.session_state.seeds:
        st.warning("Primero carga las principales pérdidas.")
    elif not isinstance(st.session_state.sub_prefixes, dict) or not st.session_state.sub_prefixes:
        st.warning("Define al menos una sub‑palabra para alguna pérdida.")
    else:
        seeds_to_prefixes = st.session_state.sub_prefixes
        rows = []
        for dur, comment in st.session_state.pairs:
            tokens = normalize_text(comment).split()
            counts = count_matches(tokens, seeds_to_prefixes)
            rows.append({"Duración": dur, "Comentario": comment, **counts})

        df_counts = pd.DataFrame(rows)
        seed_cols = list(st.session_state.seeds)

        # Asegura columnas de seed aunque no haya matches
        for s in seed_cols:
            if s not in df_counts.columns:
                df_counts[s] = 0

        # Ganador por fila
        df_counts["MaxScore"] = df_counts[seed_cols].max(axis=1)

        def pick_winner(row):
            scores = [row[s] for s in seed_cols]
            maxv = max(scores)
            if maxv == 0:
                return "Sin coincidencias"
            winners = [s for s in seed_cols if row[s] == maxv]
            return winners[0] if len(winners) == 1 else "Empate"

        df_counts["Ganador"] = df_counts.apply(pick_winner, axis=1)
        tie_mask = (df_counts["Ganador"] == "Empate")

        # Matriz binaria
        df_binary = df_counts[["Duración", "Comentario"]].copy()
        for s in seed_cols:
            df_binary[s] = "-"

        for i, r in df_counts.iterrows():
            if r["Ganador"] in seed_cols:
                df_binary.at[i, r["Ganador"]] = 1

        totals_binary = {"Duración": "", "Comentario": "TOTAL"}
        for s in seed_cols:
            totals_binary[s] = (df_binary[s] == 1).sum()

        df_binary_total = pd.concat([df_binary, pd.DataFrame([totals_binary])], ignore_index=True)
        tie_mask = pd.concat([tie_mask, pd.Series([False], index=[len(tie_mask)])])

        # Resumen por pérdida
        resumen = []
        for s in seed_cols:
            stops = (df_binary[s] == 1).sum()
            dur_total = df_counts.loc[df_binary[s] == 1, "Duración"].sum()
            resumen.append({"Principal pérdida": s, "#Stops": stops, "Duración Total": dur_total})

        df_resumen = pd.DataFrame(resumen)
        df_total = pd.DataFrame([{
            "Total #Stops": df_resumen["#Stops"].sum(),
            "Total Duración": df_resumen["Duración Total"].sum()
        }])

        # Mostrar
        st.subheader("Sub‑palabras finales")
        st.dataframe(pd.DataFrame({
            "Principal pérdida": list(seeds_to_prefixes.keys()),
            "Sub‑palabras": [", ".join(v) for v in seeds_to_prefixes.values()]
        }))
        st.subheader("Matriz binaria")
        st.dataframe(style_ties(df_binary_total, tie_mask))
        st.subheader("Resumen por principal pérdida")
        st.dataframe(df_resumen)
        st.subheader("Totales globales")
        st.dataframe(df_total)

        # Descargar Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as xls:
            pd.DataFrame({
                "Principal pérdida": list(seeds_to_prefixes.keys()),
                "Sub‑palabras": [", ".join(v) for v in seeds_to_prefixes.values()]
            }).to_excel(xls, index=False, sheet_name="Subpalabras")
            df_binary_total.to_excel(xls, index=False, sheet_name="Matriz_binaria")
            df_resumen.to_excel(xls, index=False, sheet_name="Resumen")
            df_total.to_excel(xls, index=False, sheet_name="Totales")
            df_counts.to_excel(xls, index=False, sheet_name="Detalle_filas")

        output.seek(0)
        st.download_button(
            "Descargar Excel",
            data=output,
            file_name="clasificacion.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
