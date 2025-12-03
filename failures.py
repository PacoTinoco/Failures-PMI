
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
st.title("Clasificación de Principales Pérdidas y Sub‑palabras (CSV/XLSX encabezado en fila 9)")

# =========================
# Utilidades
# =========================
def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", strip_accents(text.lower()))).strip()

def discover_subprefixes(pairs, seed, top_m=10, min_prefix_len=3, stopwords=None):
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
    counts = {seed: 0 for seed in seeds_to_prefixes}
    for seed, prefixes in seeds_to_prefixes.items():
        for tok in tokens:
            if tok == seed or any(tok.startswith(p) for p in prefixes):
                counts[seed] += 1
    return counts

def enforce_unique_subs(seeds_to_prefixes, pairs):
    pref_counts = {}
    for seed, prefs in seeds_to_prefixes.items():
        for p in prefs:
            pref_counts.setdefault(p, {})
            pref_counts[p].setdefault(seed, 0)
    for _, comment in pairs:
        tokens = normalize_text(comment).split()
        for p in pref_counts:
            if any(tok.startswith(p) for tok in tokens):
                for seed in seeds_to_prefixes:
                    if p in seeds_to_prefixes[seed]:
                        pref_counts[p][seed] += 1
    owner = {p: max(pref_counts[p], key=pref_counts[p].get) for p in pref_counts}
    unique = {s: [] for s in seeds_to_prefixes}
    for p, s in owner.items():
        unique[s].append(p)
    return unique

def style_ties(df, tie_mask):
    def _row_style(row):
        return ['background-color: #ffb3b3'] * len(row) if tie_mask.loc[row.name] else [''] * len(row)
    return df.style.apply(_row_style, axis=1)

STOPWORDS = {"de","la","el","y","en","por","se","al","a","no","lo","una","un","otro","otros","limpieza","suciedad"}

# =========================
# Estado inicial
# =========================
if "pairs" not in st.session_state:
    st.session_state.pairs = []
if "seeds" not in st.session_state:
    st.session_state.seeds = []
if "sub_prefixes" not in st.session_state:
    st.session_state.sub_prefixes = {}
if "duration_col" not in st.session_state:
    st.session_state.duration_col = "Duration"
if "comment_col" not in st.session_state:
    st.session_state.comment_col = "Operator Comment"

# =========================
# Controles (barra lateral)
# =========================
st.sidebar.header("Opciones")
top_m_subs = st.sidebar.slider("Top M sub‑palabras (auto)", 5, 20, 10)
min_prefix_len = st.sidebar.slider("Longitud mínima de prefijo", 2, 6, 3)
enforce_unique = st.sidebar.checkbox("Enforzar unicidad de sub‑palabras entre seeds", True)
mostrar_matriz_conteos = st.sidebar.checkbox("Mostrar matriz de conteos", False)

# =========================
# Paso 1: Subir CSV/XLSX y leer encabezado desde fila 9
# =========================
st.markdown("### 1) Sube el archivo **CSV** o **XLSX** (encabezados en fila 9)")
uploaded_file = st.file_uploader("Selecciona el archivo", type=["csv", "xlsx"])

df = None
if uploaded_file:
    fname = uploaded_file.name.lower()
    try:
        if fname.endswith(".csv"):
            df = pd.read_csv(uploaded_file, skiprows=8)  # encabezado en fila 9
        elif fname.endswith(".xlsx"):
            xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
            sheet = st.selectbox("Hoja de Excel:", xls.sheet_names, index=0)
            df = pd.read_excel(uploaded_file, sheet_name=sheet, engine="openpyxl", skiprows=8)
        else:
            st.error("Formato no soportado. Usa CSV o XLSX.")
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")

    if df is not None:
        st.write("Vista previa de las primeras filas:")
        st.dataframe(df.head(10), use_container_width=True)

        # Validación + mapeo manual si falta
        cols = list(df.columns)
        has_duration = "Duration" in cols
        has_comment = "Operator Comment" in cols

        if not (has_duration and has_comment):
            st.warning("El archivo no tiene exactamente las columnas `Duration` y `Operator Comment`. Selecciona manualmente:")
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.duration_col = st.selectbox("Columna para **Duration**:", cols, index=0)
            with c2:
                st.session_state.comment_col = st.selectbox("Columna para **Operator Comment**:", cols, index=0)
        else:
            st.session_state.duration_col = "Duration"
            st.session_state.comment_col = "Operator Comment"

        # Limpieza
        df[st.session_state.duration_col] = pd.to_numeric(df[st.session_state.duration_col], errors="coerce")
        df = df.dropna(subset=[st.session_state.duration_col, st.session_state.comment_col])
        df = df[df[st.session_state.duration_col] > 0]
        df[st.session_state.comment_col] = df[st.session_state.comment_col].astype(str).str.strip()
        df = df[df[st.session_state.comment_col] != ""]

        st.session_state.pairs = list(zip(df[st.session_state.duration_col], df[st.session_state.comment_col]))
        st.success(f"Filas válidas: {len(st.session_state.pairs)}")

# =========================
# Paso 2: Seeds y sub‑palabras automáticas
# =========================
if st.session_state.pairs:
    st.markdown("### 2) Ingresa las principales pérdidas (separadas por coma)")
    seeds_text = st.text_input("Principales pérdidas:", value=", ".join(st.session_state.seeds) if st.session_state.seeds else "")
    if st.button("Descubrir sub‑palabras automáticas"):
        seeds = [normalize_text(s) for s in seeds_text.split(",") if s.strip()]
        if not seeds:
            st.error("Debes ingresar al menos una principal pérdida.")
        else:
            st.session_state.seeds = seeds
            auto_prefixes = {s: discover_subprefixes(st.session_state.pairs, s, top_m_subs, min_prefix_len, STOPWORDS) for s in seeds}
            st.session_state.sub_prefixes = {s: auto_prefixes.get(s, [])[:] for s in seeds}
            st.success("Sub‑palabras automáticas generadas. Edita abajo.")

# =========================
# Paso 3: Edición manual persistente
# =========================
if st.session_state.seeds:
    st.markdown("### 3) Edita/agrega sub‑palabras manualmente")
    with st.form("form_subs"):
        edited = {}
        for s in st.session_state.seeds:
            default_val = ", ".join(st.session_state.sub_prefixes.get(s, []))
            edited[s] = st.text_input(f"Sub‑palabras para **{s}**:", value=default_val)
        saved = st.form_submit_button("Guardar cambios")
        if saved:
            final_subs = {s: [normalize_text(p) for p in edited[s].split(",") if p.strip()] for s in st.session_state.seeds}
            if enforce_unique:
                final_subs = enforce_unique_subs(final_subs, st.session_state.pairs)
            st.session_state.sub_prefixes = final_subs
            st.success("Sub‑palabras guardadas.")

# =========================
# Paso 4: Generar análisis
# =========================
if st.button("Generar análisis"):
    if not st.session_state.pairs or not st.session_state.seeds:
        st.error("Carga datos y define principales pérdidas.")
        st.stop()

    seeds_to_prefixes = st.session_state.sub_prefixes
    rows = []
    for dur, comment in st.session_state.pairs:
        tokens = normalize_text(comment).split()
        counts = count_matches(tokens, seeds_to_prefixes)
        rows.append({"Duración": dur, "Comentario": comment, **counts})
    df_counts = pd.DataFrame(rows)

    seed_cols = st.session_state.seeds
    for s in seed_cols:
        if s not in df_counts.columns:
            df_counts[s] = 0

    df_counts["MaxScore"] = df_counts[seed_cols].max(axis=1)
    df_counts["Ganador"] = df_counts.apply(lambda r: "Empate" if list(r[seed_cols]).count(r["MaxScore"]) > 1 and r["MaxScore"] > 0 else (next((s for s in seed_cols if r[s]==r["MaxScore"]), "Sin coincidencias")), axis=1)
    tie_mask = pd.concat([df_counts["Ganador"]=="Empate", pd.Series([False], index=[len(df_counts)])])

    # Matriz binaria
    df_binary = df_counts[["Duración","Comentario"]].copy()
    for s in seed_cols: df_binary[s] = "-"
    for i,r in df_counts.iterrows():
        if r["Ganador"] in seed_cols: df_binary.at[i,r["Ganador"]] = 1
    totals_binary = {"Duración":"","Comentario":"TOTAL"}
    for s in seed_cols: totals_binary[s]=(df_binary[s]==1).sum()
    df_binary_total = pd.concat([df_binary,pd.DataFrame([totals_binary])],ignore_index=True)

    # Resumen
    resumen = []
    for s in seed_cols:
        stops = (df_binary[s]==1).sum()
        dur_total = df_counts.loc[df_binary[s]==1,"Duración"].sum()
        resumen.append({"Principal pérdida":s,"#Stops":stops,"Duración Total":dur_total})
    df_resumen = pd.DataFrame(resumen)
    df_total = pd.DataFrame([{"Total #Stops":df_resumen["#Stops"].sum(),"Total Duración":df_resumen["Duración Total"].sum()}])

    # Mostrar
    st.subheader("Sub‑palabras finales")
    st.dataframe(pd.DataFrame({"Principal pérdida":list(seeds_to_prefixes.keys()),"Sub‑palabras":[", ".join(v) for v in seeds_to_prefixes.values()]}))
    st.subheader("Matriz binaria")
    st.dataframe(style_ties(df_binary_total,tie_mask))
    st.subheader("Resumen por principal pérdida")
    st.dataframe(df_resumen)
    st.subheader("Totales globales")
    st.dataframe(df_total)

    # Descargar Excel
    output = BytesIO()
    with pd.ExcelWriter(output,engine="openpyxl") as xls:
        pd.DataFrame({"Principal pérdida":list(seeds_to_prefixes.keys()),"Sub‑palabras":[", ".join(v) for v in seeds_to_prefixes.values()]}).to_excel(xls,index=False,sheet_name="Subpalabras")
        df_binary_total.to_excel(xls,index=False,sheet_name="Matriz_binaria")
        df_resumen.to_excel(xls,index=False,sheet_name="Resumen")
        df_total.to_excel(xls,index=False,sheet_name="Totales")
        if mostrar_matriz_conteos:
            df_counts.to_excel(xls,index=False,sheet_name="Matriz_conteos")
    output.seek(0)
    st.download_button("Descargar Excel",data=output,file_name="clasificacion.xlsx")
