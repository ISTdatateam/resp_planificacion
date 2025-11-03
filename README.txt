
# Mapa de Roles Preventivos – Radar (Streamlit)

## Ejecutar
1. (Opcional) Crea y activa un entorno virtual.
2. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Corre la app:
   ```bash
   streamlit run app.py
   ```

## Configuración
- La app carga un CSV publicado de Google Sheets (URL en la barra lateral).
- Puedes configurar la URL en `.streamlit/secrets.toml`:
  ```toml
  CSV_URL = "https://docs.google.com/spreadsheets/d/e/.../pub?gid=...&single=true&output=csv"
  ```

## Pipeline
- Normaliza columnas de roles (Jefe de obra, Prevencionista, Capataz, Subcontrato, Trabajadores).
- Divide respuestas múltiples por coma y mapea `Informa`→`Comunica`.
- Pivotea a matriz Rol × (Planifica, Autoriza, Ejecuta, Supervisa, Comunica).
- Radar con filtros por Curso, Comunidad y Tarea.
