
# app.py
# Streamlit app: Mapa de Roles Preventivos – Radar por Rol
# Run: streamlit run app.py

import io
import re
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mapa de Roles Preventivos – Radar", page_icon="🛠️", layout="wide")

CSV_URL = st.secrets.get("CSV_URL", "https://docs.google.com/spreadsheets/d/e/2PACX-1vTKRi88h4vkeGJhY3nvSiYANhVrg2XERmMQqSQ5k-d-DPwCKqcNbVaX_glEv-BgMhO0wjeDEYGj_Ozr/pub?gid=129987530&single=true&output=csv")

ROLE_COLUMNS = [
    "Roles [Jefe de obra]",
    "Roles [Prevencionista]",
    "Roles [Capataz]",
    "Roles [Subcontrato]",
    "Roles [Trabajadores]",
    # "Roles [Fila 5]" // si aparece, se ignora
]

ROLE_NAMES = {
    "Roles [Jefe de obra]" : "Jefe de obra",
    "Roles [Prevencionista]": "Prevencionista",
    "Roles [Capataz]": "Capataz",
    "Roles [Subcontrato]": "Subcontrato",
    "Roles [Trabajadores]": "Trabajadores"
}

FUNCTIONS = ["Planifica", "Autoriza", "Ejecuta", "Supervisa", "Comunica"]
SYNONYMS = {"Informa": "Comunica"}  # Mapeo a vocabulario único

@st.cache_data(show_spinner=False)
def fetch_csv(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url, dtype=str, encoding="utf-8")
    except Exception:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

def split_functions(cell: str):
    if not isinstance(cell, str) or not cell.strip():
        return []
    parts = [p.strip() for p in cell.split(",")]
    out = []
    for p in parts:
        if not p:
            continue
        p = SYNONYMS.get(p, p)
        if p in FUNCTIONS:
            out.append(p)
    return out

def normalize(df: pd.DataFrame, drop_subcontrato: bool = True) -> pd.DataFrame:
    keep_cols = [
        "Marca temporal",
        "Selecciona tu curso",
        "Selecciona tu comunidad",
        "Tarea critica a analizar",
        "Nombre",
    ] + [c for c in ROLE_COLUMNS if c in df.columns]

    df2 = df[[c for c in keep_cols if c in df.columns]].copy()

    if drop_subcontrato and "Roles [Subcontrato]" in df2.columns:
        df2 = df2.drop(columns=["Roles [Subcontrato]"])

    role_cols_present = [c for c in ROLE_COLUMNS if c in df2.columns]
    melted = df2.melt(
        id_vars=[c for c in ["Marca temporal","Selecciona tu curso","Selecciona tu comunidad","Tarea critica a analizar","Nombre"] if c in df2.columns],
        value_vars=role_cols_present,
        var_name="Rol_col",
        value_name="Valor"
    )

    rows = []
    for _, r in melted.iterrows():
        for func in split_functions(r["Valor"]):
            rows.append({
                "Curso": r.get("Selecciona tu curso"),
                "Comunidad": r.get("Selecciona tu comunidad"),
                "Tarea": r.get("Tarea critica a analizar"),
                "Nombre": r.get("Nombre"),
                "Rol": ROLE_NAMES.get(r["Rol_col"], r["Rol_col"]),
                "Funcion": func
            })
    norm = pd.DataFrame(rows)
    return norm

def pivot_counts(norm: pd.DataFrame, curso: str = "", comunidad: str = "", tarea: str = "") -> pd.DataFrame:
    df = norm.copy()
    if curso: df = df[df["Curso"] == curso]
    if comunidad: df = df[df["Comunidad"] == comunidad]
    if tarea: df = df[df["Tarea"] == tarea]

    if df.empty:
        return pd.DataFrame(index=list(ROLE_NAMES.values()), columns=FUNCTIONS).fillna(0).astype(int)

    pv = (
        df
        .groupby(["Rol","Funcion"])
        .size()
        .unstack("Funcion")
        .reindex(columns=FUNCTIONS)
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    for rn in ROLE_NAMES.values():
        if rn not in pv.index:
            pv.loc[rn] = 0
    pv = pv.reindex(index=list(ROLE_NAMES.values()))
    return pv

def radar_angles(n):
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    theta = np.concatenate([theta, theta[:1]])
    return theta

def plot_radar(pivot_df: pd.DataFrame, roles_to_plot):
    values_max = int(pivot_df.to_numpy().max()) if not pivot_df.empty else 1
    values_max = max(1, values_max)
    angles = radar_angles(len(FUNCTIONS))

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(FUNCTIONS)
    ax.set_yticks(range(1, values_max+1))
    ax.set_yticklabels([str(x) for x in range(1, values_max+1)])
    ax.set_rlabel_position(0)

    for rol in roles_to_plot:
        if rol in pivot_df.index:
            vals = pivot_df.loc[rol, FUNCTIONS].tolist()
            vals = vals + vals[:1]
            ax.plot(angles, vals, linewidth=2, linestyle='solid', label=rol)
            ax.fill(angles, vals, alpha=0.10)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    return fig

st.title("🛠️ Mapa de Roles Preventivos – Radar por Rol")

with st.sidebar:
    st.markdown("### Fuente de datos")
    st.write("El app carga un CSV publicado de Google Sheets.")
    st.text_input("CSV URL", value=CSV_URL, key="csv_url")

    drop_sub = st.checkbox("Excluir 'Subcontrato' del análisis", value=True,
                           help="Actívalo para replicar el radar original sin incluir Subcontrato.")

raw = fetch_csv(st.session_state["csv_url"])
norm = normalize(raw, drop_subcontrato=drop_sub)

cursos = [""] + sorted([x for x in norm["Curso"].dropna().unique().tolist()])
comus  = [""] + sorted([x for x in norm["Comunidad"].dropna().unique().tolist()])
tareas = [""] + sorted([x for x in norm["Tarea"].dropna().unique().tolist()])

fc1, fc2, fc3 = st.columns([1,1,2])
with fc1:
    curso_sel = st.selectbox("Selecciona tu curso (vacío = todos)", cursos, index=1 if len(cursos)>1 else 0)
with fc2:
    comu_sel = st.selectbox("Selecciona tu comunidad (vacío = todas)", comus)
with fc3:
    tarea_sel = st.selectbox("Tarea crítica (vacío = todas)", tareas, index=1 if len(tareas)>1 else 0)

pv = pivot_counts(norm, curso_sel, comu_sel, tarea_sel)

roles_all = pv.index.tolist()
default_roles = [r for r in roles_all if r != "Subcontrato"]
roles_sel = st.multiselect("Roles a mostrar", roles_all, default=default_roles if default_roles else roles_all)

c1, c2 = st.columns([3,2])
with c1:
    st.subheader(tarea_sel or "Todas las tareas")
    fig = plot_radar(pv, roles_sel if roles_sel else roles_all)
    st.pyplot(fig, use_container_width=True)

with c2:
    st.subheader("Tabla (conteos)")
    st.dataframe(pv.loc[roles_sel] if roles_sel else pv, use_container_width=True)

st.divider()
col_a, col_b = st.columns(2)
with col_a:
    st.download_button("⬇️ Descargar datos normalizados (CSV)",
                       data=norm.to_csv(index=False).encode("utf-8-sig"),
                       file_name="normalizado_mapa_roles.csv",
                       mime="text/csv")

with col_b:
    st.download_button("⬇️ Descargar pivote filtrado (CSV)",
                       data=pv.to_csv().encode("utf-8-sig"),
                       file_name="pivote_roles_funciones.csv",
                       mime="text/csv")

st.caption("Tip: Usa `.streamlit/secrets.toml` con `CSV_URL=\"...\"` para cambiar la fuente sin tocar el código.")
