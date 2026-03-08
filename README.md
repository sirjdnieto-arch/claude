# 🎯 Trinidad System
**Análisis cuantitativo automatizado de mercados financieros.**
Sistema de 3 motores que se ejecuta diariamente via GitHub Actions y publica un dashboard en GitHub Pages.
## 🏗️ Arquitectura
```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  yfinance    │────▶│  L'CRACK v7.2    │────▶│                 │
│  (37 ETFs)   │     │  HMM Ensemble x5 │     │   Dashboard     │
└─────────────┘     │  + Percentiles    │     │   GitHub Pages  │
                    └──────────────────┘     │                 │
┌─────────────┐     ┌──────────────────┐     │  - Régimen HMM  │
│  yfinance    │────▶│  PISTOLERO v3.1  │────▶│  - Sectores     │
│  (~600 stocks│     │  34 Features     │     │  - Top Stocks   │
│  + scraping) │     │  Quality Momentum│     │  - Favoritos    │
└─────────────┘     └──────────────────┘     └─────────────────┘
```
## 📊 Motores
### L'CRACK v7.2 — Análisis Sectorial
- **HMM Ensemble**: 5 modelos con semillas distintas, voto por mayoría
- **Suavizado temporal**: Ventana de 5 días para evitar oscilaciones
- **27 sectores** via ETFs (tecnología, energía, salud, cripto, etc.)
- **Clasificación por percentiles**: Líder / Mejorando / Neutro / Debilitándose / Rezagado
### PISTOLERO v3.1 — Stock Picking
- **~600 acciones** de S&P500 + Nasdaq100 + DAX + IBEX35
- **34 features técnicas** (momentum, volatilidad, volumen, tendencia)
- **7 factores de ranking** adaptativos al régimen del mercado
- **Pesos dinámicos**: En BEAR pesa más baja volatilidad, en BULL más momentum
### SARA v4.7 — Análisis Fundamental (próximamente)
- Piotroski F-Score, Altman Z-Score
- Valoración multidimensional
- Fuente: FMP (Financial Modeling Prep)
## ⚙️ Configuración
### 1. Fork/Clone del repo
```bash
git clone https://github.com/TU_USUARIO/trinidad-system.git
```
### 2. Habilitar GitHub Pages
- Settings → Pages → Source: **GitHub Actions**
### 3. (Opcional) Añadir secrets
- `FMP_API_KEY`: Para SARA cuando se integre
### 4. Ejecución manual
- Actions → "Trinidad System - Daily Analysis" → Run workflow
## 📅 Programación
| Motor | Frecuencia | Horario |
|-------|-----------|---------|
| L'CRACK | Diario (L-V) | 08:00 UTC |
| PISTOLERO | Diario (L-V) | 08:00 UTC |
| SARA | Semanal (próximamente) | — |
## 📁 Estructura
```
├── main.py                 # Motor principal (L'CRACK + PISTOLERO)
├── generate_dashboard.py   # Genera el HTML del dashboard
├── requirements.txt        # Dependencias Python
├── .github/
│   └── workflows/
│       └── daily_run.yml   # GitHub Action (cron diario)
├── output/
│   ├── results.json        # Datos crudos (generado)
│   └── index.html          # Dashboard (generado)
└── README.md
```
## ⚠️ Disclaimer
Este sistema es para uso personal y educativo. No constituye asesoramiento financiero.
Los resultados pasados no garantizan resultados futuros.
