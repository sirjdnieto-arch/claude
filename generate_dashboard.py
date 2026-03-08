#!/usr/bin/env python3
"""
TRINIDAD SYSTEM — Dashboard Generator
Lee results.json y genera index.html para GitHub Pages
"""
import json
from pathlib import Path
from datetime import datetime
def generate_dashboard():
    output_dir = Path("output")
    results_file = output_dir / "results.json"
    if not results_file.exists():
        print("ERROR: No se encontro results.json")
        return
    with open(results_file) as f:
        data = json.load(f)
    lcrack = data.get("lcrack", {})
    pistolero = data.get("pistolero", {})
    generated_at = data.get("generated_at", "N/A")
    # Parse data
    regime = lcrack.get("regime", 1)
    regime_label = lcrack.get("regime_label", "?")
    consensus = lcrack.get("consensus", 0)
    ret_z = lcrack.get("ret_z", 0)
    vol_z = lcrack.get("vol_z", 0)
    z_risk = lcrack.get("z_risk", 0)
    sectors = lcrack.get("sectors", [])
    details = lcrack.get("details", {})
    stocks = pistolero.get("stocks", [])
    total_analyzed = pistolero.get("total_analyzed", 0)
    # Regime colors
    regime_colors = {0: "#ef4444", 1: "#f59e0b", 2: "#22c55e"}
    regime_bg = {0: "#fef2f2", 1: "#fffbeb", 2: "#f0fdf4"}
    regime_icons = {0: "&#x26A0;", 1: "&#x25CB;", 2: "&#x2714;"}
    regime_color = regime_colors.get(regime, "#6b7280")
    regime_bg_color = regime_bg.get(regime, "#f9fafb")
    regime_icon = regime_icons.get(regime, "?")
    # Build sector rows
    sector_rows = ""
    status_colors = {
        "LIDER INSTITUCIONAL": "#065f46",
        "LIDER TECNICO": "#047857",
        "MEJORANDO": "#059669",
        "NEUTRO": "#6b7280",
        "DEBILITANDOSE": "#dc2626",
        "REZAGADO": "#991b1b",
        "SALIENDO": "#7f1d1d",
        "SALIDA SUGERIDA": "#b91c1c",
        "DISTRIBUCION": "#b91c1c",
    }
    status_emojis = {
        "LIDER INSTITUCIONAL": "&#x1F40B;",
        "LIDER TECNICO": "&#x1F680;",
        "MEJORANDO": "&#x1F4C8;",
        "NEUTRO": "&#x26AA;",
        "DEBILITANDOSE": "&#x1F4C9;",
        "REZAGADO": "&#x26A0;",
        "SALIENDO": "&#x1F6AA;",
        "SALIDA SUGERIDA": "&#x1F6AB;",
        "DISTRIBUCION": "&#x1F6D1;",
    }
    for i, s in enumerate(sectors):
        status = s.get("Status", "?")
        color = status_colors.get(status, "#6b7280")
        emoji = status_emojis.get(status, "")
        gfz = s.get("GF_Z", 0)
        gfz_color = "#22c55e" if gfz > 1 else "#ef4444" if gfz < -1 else "#6b7280"
        sector_rows += f"""
        <tr>
            <td style="font-weight:600">{i+1}</td>
            <td style="font-weight:600">{s.get('Sector','')}</td>
            <td><code>{s.get('ETF','')}</code></td>
            <td style="color:{gfz_color};font-weight:600">{gfz:+.2f}</td>
            <td>{s.get('Rot_Z', 0):+.2f}</td>
            <td>{s.get('RVOL', 0):.2f}</td>
            <td style="color:{color};font-weight:700">{emoji} {status}</td>
            <td style="font-size:0.8em;color:#666">{s.get('Notas','')}</td>
        </tr>"""
    # Build stock rows (top 30)
    stock_rows = ""
    for i, s in enumerate(stocks[:30]):
        rsi = s.get("rsi", 50)
        rsi_color = "#ef4444" if rsi > 75 else "#3b82f6" if rsi < 25 else "#6b7280"
        rsi_tag = " &#x1F525;" if rsi > 70 else " &#x2744;" if rsi < 30 else ""
        score = s.get("score", 0)
        score_color = "#22c55e" if score > 0.7 else "#f59e0b" if score > 0.5 else "#ef4444"
        fav_tag = " &#x2B50;" if s.get("is_favorito") else ""
        stock_rows += f"""
        <tr>
            <td style="font-weight:600">{i+1}</td>
            <td style="font-weight:700">{s.get('ticker','')}{fav_tag}</td>
            <td>{s.get('sector', 'N/A')}</td>
            <td><code>{s.get('index', '')}</code></td>
            <td style="color:{score_color};font-weight:700">{score:.3f}</td>
            <td style="color:{rsi_color};font-weight:600">{rsi:.1f}{rsi_tag}</td>
            <td>{s.get('close', 0):.2f}</td>
        </tr>"""
    # Favoritos
    favs = [s for s in stocks if s.get("is_favorito")]
    favs.sort(key=lambda x: x.get("score", 0), reverse=True)
    fav_rows = ""
    for s in favs:
        rank = next((i+1 for i, st in enumerate(stocks) if st['ticker'] == s['ticker']), 0)
        score = s.get("score", 0)
        score_color = "#22c55e" if score > 0.7 else "#f59e0b" if score > 0.5 else "#ef4444"
        rsi = s.get("rsi", 50)
        rsi_color = "#ef4444" if rsi > 75 else "#3b82f6" if rsi < 25 else "#6b7280"
        fav_rows += f"""
        <tr>
            <td style="font-weight:700">&#x2B50; {s.get('ticker','')}</td>
            <td>{s.get('sector', 'N/A')}</td>
            <td style="color:{score_color};font-weight:700">{score:.3f}</td>
            <td>{rank}/{total_analyzed}</td>
            <td style="color:{rsi_color}">{rsi:.1f}</td>
        </tr>"""
    # Bottom 5
    bottom = stocks[-5:] if len(stocks) >= 5 else stocks
    bottom_rows = ""
    for s in reversed(bottom):
        bottom_rows += f"""
        <tr style="color:#991b1b">
            <td style="font-weight:600">&#x1F6AB; {s.get('ticker','')}</td>
            <td>{s.get('sector', 'N/A')}</td>
            <td>{s.get('score', 0):.3f}</td>
            <td>{s.get('rsi', 0):.1f}</td>
        </tr>"""
    # Ensemble detail
    votos = details.get("votos", [])
    votos_html = ""
    for v in votos:
        c = "#22c55e" if v == "BULL" else "#ef4444" if v == "BEAR" else "#f59e0b"
        votos_html += f'<span style="background:{c};color:white;padding:2px 8px;border-radius:4px;margin:2px;font-size:0.85em">{v}</span>'
    window = details.get("window", [])
    window_html = ""
    for w in window:
        c = "#22c55e" if w == "BULL" else "#ef4444" if w == "BEAR" else "#f59e0b"
        window_html += f'<span style="background:{c};color:white;padding:2px 6px;border-radius:3px;margin:1px;font-size:0.8em">{w}</span>'
    # Format timestamp
    try:
        ts = datetime.fromisoformat(generated_at)
        ts_formatted = ts.strftime("%d %b %Y, %H:%M UTC")
    except Exception:
        ts_formatted = str(generated_at)
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trinidad System Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 30px 20px;
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-radius: 16px;
            margin-bottom: 24px;
            border: 1px solid #475569;
        }}
        .header h1 {{
            font-size: 2.2em;
            background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .header .timestamp {{
            color: #94a3b8;
            font-size: 0.95em;
        }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
        .card {{
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #334155;
        }}
        .card h3 {{
            font-size: 0.85em;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        .card .value {{
            font-size: 1.8em;
            font-weight: 800;
        }}
        .card .sub {{ color: #64748b; font-size: 0.85em; margin-top: 4px; }}
        .regime-card {{
            background: {regime_bg_color};
            border: 2px solid {regime_color};
        }}
        .regime-card .value {{ color: {regime_color}; }}
        .section {{
            background: #1e293b;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid #334155;
        }}
        .section h2 {{
            font-size: 1.3em;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1px solid #334155;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        th {{
            text-align: left;
            padding: 10px 8px;
            color: #94a3b8;
            font-weight: 600;
            border-bottom: 2px solid #334155;
            font-size: 0.85em;
            text-transform: uppercase;
        }}
        td {{
            padding: 8px;
            border-bottom: 1px solid #1e293b;
        }}
        tr:hover {{ background: #334155; }}
        .ensemble-detail {{
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
            margin-top: 12px;
        }}
        .ensemble-label {{
            color: #94a3b8;
            font-size: 0.85em;
            min-width: 80px;
        }}
        code {{
            background: #334155;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #475569;
            font-size: 0.85em;
        }}
        @media (max-width: 768px) {{
            .grid {{ grid-template-columns: 1fr; }}
            body {{ padding: 10px; }}
            table {{ font-size: 0.8em; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>&#x1F3AF; Trinidad System</h1>
        <p style="color:#cbd5e1;font-size:1.1em">L'CRACK v7.2 + PISTOLERO v3.1</p>
        <p class="timestamp">Ultima actualizacion: {ts_formatted}</p>
    </div>
    <!-- KPIs -->
    <div class="grid">
        <div class="card regime-card">
            <h3>Regimen de Mercado</h3>
            <div class="value">{regime_icon} {regime_label}</div>
            <div class="sub">Consenso: {consensus}% (Ensemble x5)</div>
        </div>
        <div class="card">
            <h3>Z-Risk</h3>
            <div class="value" style="color:{'#22c55e' if z_risk > 0 else '#ef4444' if z_risk < -1 else '#f59e0b'}">{z_risk:+.2f}</div>
            <div class="sub">Retorno: {ret_z:+.2f}&#x03C3; | Volatilidad: {vol_z:+.2f}&#x03C3;</div>
        </div>
        <div class="card">
            <h3>Universo Analizado</h3>
            <div class="value" style="color:#60a5fa">{total_analyzed}</div>
            <div class="sub">{len(sectors)} sectores + {total_analyzed} acciones</div>
        </div>
    </div>
    <!-- HMM Detail -->
    <div class="section">
        <h2>&#x1F3DB; Motor HMM — Detalle del Ensemble</h2>
        <div class="ensemble-detail">
            <span class="ensemble-label">Votos (x5):</span>
            {votos_html}
        </div>
        <div class="ensemble-detail" style="margin-top:8px">
            <span class="ensemble-label">Ultimos 5d:</span>
            {window_html}
        </div>
        <div style="margin-top:12px;color:#94a3b8;font-size:0.85em">
            BEAR: {details.get('bear',0)} | LATERAL: {details.get('lateral',0)} | BULL: {details.get('bull',0)}
            &nbsp;&mdash;&nbsp; Probabilidad media: {details.get('prob_media',0)}%
        </div>
    </div>
    <!-- Sectores -->
    <div class="section">
        <h2>&#x1F4CA; L'CRACK — Mapa Sectorial ({len(sectors)} sectores)</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th><th>Sector</th><th>ETF</th>
                    <th>GF_Z</th><th>Rot_Z</th><th>RVOL</th>
                    <th>Status</th><th>Notas</th>
                </tr>
            </thead>
            <tbody>
                {sector_rows}
            </tbody>
        </table>
    </div>
    <!-- Top Stocks -->
    <div class="section">
        <h2>&#x1F3AF; PISTOLERO — Top 30 Stocks</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th><th>Ticker</th><th>Sector</th>
                    <th>Indice</th><th>Score</th><th>RSI</th><th>Precio</th>
                </tr>
            </thead>
            <tbody>
                {stock_rows}
            </tbody>
        </table>
    </div>
    <!-- Favoritos -->
    <div class="section">
        <h2>&#x2B50; Favoritos</h2>
        <table>
            <thead>
                <tr><th>Ticker</th><th>Sector</th><th>Score</th><th>Rank</th><th>RSI</th></tr>
            </thead>
            <tbody>
                {fav_rows}
            </tbody>
        </table>
    </div>
    <!-- Bottom 5 -->
    <div class="section">
        <h2>&#x1F6AB; Evitar (Bottom 5)</h2>
        <table>
            <thead>
                <tr><th>Ticker</th><th>Sector</th><th>Score</th><th>RSI</th></tr>
            </thead>
            <tbody>
                {bottom_rows}
            </tbody>
        </table>
    </div>
    <div class="footer">
        <p>Trinidad System &mdash; L'CRACK v7.2 + PISTOLERO v3.1</p>
        <p>Datos: yfinance | Ejecucion: GitHub Actions | Dashboard: GitHub Pages</p>
        <p>Generado: {ts_formatted}</p>
    </div>
</body>
</html>"""
    # Write
    output_file = output_dir / "index.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard generado: {output_file}")
    print(f"  Sectores: {len(sectors)}")
    print(f"  Stocks: {len(stocks)}")
    print(f"  Favoritos: {len(favs)}")
if __name__ == "__main__":
    generate_dashboard()
