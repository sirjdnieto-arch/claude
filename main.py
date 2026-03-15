#!/usr/bin/env python3
"""
TRINIDAD SYSTEM — Motor Principal
L'CRACK v7.3 (Sectorial + HMM + Score Compuesto) + PISTOLERO v3.2 (Factores Anticipatorios)
Ejecución diaria via GitHub Actions
CHANGELOG v7.3 vs v7.2:
  - Score compuesto: 60% HMM + 40% Z-Risk con escala centrada (-1/0/+1)
  - Fix ventana temporal: consenso de 5 modelos por día (no solo semilla 42)
  - Estabilidad GF_Z: fast rolling 5→10, min_periods=40 en denominador
  - Estabilidad Rot_Z: ventana 7→14 días
  - Estabilidad RVOL: media 5 días en vez de un solo día
"""
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from io import StringIO
try:
    from hmmlearn import hmm as _hmm_lib
except ImportError:
    _hmm_lib = None
# ============================================================
# DATAPROVIDER — Fuente de datos
# ============================================================
class DataProvider:
    def __init__(self):
        self._cache = {}
        self._features_cache = {}
    def get_prices(self, ticker, period="6y", start_date=None):
        cache_key = f"{ticker}_{period}"
        if cache_key in self._cache:
            df = self._cache[cache_key]
            if start_date and len(df) > 0:
                return df[df.index >= pd.to_datetime(start_date)]
            return df
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, auto_adjust=True)
            if df.empty:
                return None
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in df.columns:
                    df[col] = np.nan
            self._cache[cache_key] = df
            if start_date and len(df) > 0:
                return df[df.index >= pd.to_datetime(start_date)]
            return df
        except:
            return None
    def get_prices_bulk(self, tickers, period="6y"):
        to_download = [t for t in tickers if f"{t}_{period}" not in self._cache]
        cached = {t: self._cache[f"{t}_{period}"] for t in tickers if f"{t}_{period}" in self._cache}
        if to_download:
            print(f"    Descargando {len(to_download)} tickers...")
            try:
                data = yf.download(to_download, period=period, auto_adjust=True,
                                   group_by='ticker', threads=True, progress=False)
                if len(to_download) == 1:
                    ticker = to_download[0]
                    df = data.copy()
                    df.columns = [c.lower().replace(' ', '_') if isinstance(c, str)
                                  else c[0].lower().replace(' ', '_') for c in df.columns]
                    if not df.empty:
                        self._cache[f"{ticker}_{period}"] = df
                        cached[ticker] = df
                else:
                    for ticker in to_download:
                        try:
                            if ticker in data.columns.get_level_values(0):
                                df = data[ticker].copy()
                                df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                                df = df.dropna(how='all')
                                if not df.empty:
                                    self._cache[f"{ticker}_{period}"] = df
                                    cached[ticker] = df
                        except:
                            continue
            except Exception as e:
                print(f"    Bulk download falló: {e}")
                for ticker in to_download:
                    df = self.get_prices(ticker, period=period)
                    if df is not None:
                        cached[ticker] = df
        return cached
    def compute_features(self, ticker, period="2y"):
        cache_key = f"feat_{ticker}"
        if cache_key in self._features_cache:
            return self._features_cache[cache_key]
        df = self.get_prices(ticker, period=period)
        if df is None or len(df) < 60:
            return None
        close = df['close']
        volume = df['volume']
        high = df['high']
        low = df['low']
        feat = pd.DataFrame(index=df.index)
        # RETORNOS
        for d in [1, 5, 10, 21, 63, 126]:
            feat[f'ret_{d}d'] = close.pct_change(d)
        # MOMENTUM CRUZADO
        feat['mom_5_20'] = close.pct_change(5) - close.pct_change(21)
        feat['mom_10_50'] = close.pct_change(10) - close.pct_change(50)
        feat['mom_20_120'] = close.pct_change(21) - close.pct_change(126)
        # DISTANCIA A MEDIAS
        for w in [20, 50, 200]:
            ma = close.rolling(w, min_periods=max(w // 2, 10)).mean()
            feat[f'dist_ma{w}'] = (close - ma) / ma
        # VOLATILIDAD REALIZADA
        daily_ret = close.pct_change()
        for w in [5, 10, 21, 63]:
            feat[f'vol_real_{w}'] = daily_ret.rolling(w).std() * np.sqrt(252)
        feat['vol_ratio_5_20'] = feat['vol_real_5'] / feat['vol_real_21'].replace(0, np.nan)
        feat['vol_ratio_20_60'] = feat['vol_real_21'] / feat['vol_real_63'].replace(0, np.nan)
        # VOLUMEN RELATIVO
        for w in [5, 10, 20]:
            vol_ma = volume.rolling(w).mean()
            feat[f'vrel_{w}d'] = volume / vol_ma.replace(0, np.nan)
        # RSI
        for period_rsi in [7, 14, 28]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period_rsi).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period_rsi).mean()
            rs = gain / loss.replace(0, np.nan)
            feat[f'rsi_{period_rsi}'] = 100 - (100 / (1 + rs))
        # RANGO Y ATR
        feat['daily_range'] = (high - low) / close
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': (high - close.shift(1)).abs(),
            'lc': (low - close.shift(1)).abs()
        }).max(axis=1)
        feat['atr_14'] = tr.rolling(14).mean() / close
        # MAX/MIN RATIO
        feat['max20_ratio'] = close / close.rolling(20).max()
        feat['min20_ratio'] = close / close.rolling(20).min()
        # SLOPES
        for w in [20, 60]:
            slopes = []
            for i in range(len(close)):
                if i < w:
                    slopes.append(np.nan)
                else:
                    y = close.iloc[i - w:i].values
                    x = np.arange(w)
                    if not np.any(np.isnan(y)):
                        slope = np.polyfit(x, y, 1)[0]
                        slopes.append(slope / (np.mean(y) + 1e-10))
                    else:
                        slopes.append(np.nan)
            feat[f'slope_{w}'] = pd.Series(slopes, index=close.index)
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = (ema12 - ema26) / close
        macd_signal = macd_line.ewm(span=9).mean()
        feat['macd'] = macd_line
        feat['macd_signal'] = macd_signal
        feat['macd_hist'] = macd_line - macd_signal
        # OBV SLOPE
        obv = (np.sign(daily_ret) * volume).cumsum()
        obv_norm = obv / obv.rolling(20).mean().replace(0, np.nan)
        feat['obv_slope'] = obv_norm.pct_change(5)
        # VOL-PRICE CORRELATION
        feat['vol_price_corr'] = daily_ret.rolling(20).corr(volume.pct_change())
        # --- FEATURES ANTICIPATORIOS (v3.2) ---
        feat['accel_5v20'] = feat['ret_5d'] - feat['ret_21d']
        feat['macd_hist_delta'] = feat['macd_hist'] - feat['macd_hist'].shift(5)
        feat['dist_min20'] = close / close.rolling(20).min() - 1
        feat['dist_max20'] = 1 - close / close.rolling(20).max()
        # Vol trend
        vol_slopes = []
        for i in range(len(volume)):
            if i < 10:
                vol_slopes.append(np.nan)
            else:
                y = volume.iloc[i-10:i].values
                if not np.any(np.isnan(y)) and np.mean(y) > 0:
                    x = np.arange(10)
                    vol_slopes.append(np.polyfit(x, y / np.mean(y), 1)[0])
                else:
                    vol_slopes.append(np.nan)
        feat['vol_trend'] = pd.Series(vol_slopes, index=close.index)
        # Squeeze
        feat['vol_squeeze'] = feat['vol_real_5'] / feat['vol_real_63'].replace(0, np.nan)
        feat = feat.replace([np.inf, -np.inf], np.nan)
        self._features_cache[cache_key] = feat
        return feat
# ============================================================
# L'CRACK v7.3 — HMM + SCORE COMPUESTO + ESTABILIDAD
# ============================================================
class LCrackV73:
    LCRACK_ETFS = {
        "SPY": "SP500", "QQQ": "Nasdaq100", "IWM": "Small Caps",
        "UUP": "Dolar", "BIL": "Efectivo", "TLT": "Bonos LP",
        "SHY": "Bonos CP", "TIP": "TIPS", "HYG": "High Yield", "LQD": "Inv Grade",
        "XLK": "Tecnologia", "AIQ": "IA", "SMH": "Semiconductores",
        "BOTZ": "IA-Robotica", "XLC": "Comunicaciones", "HACK": "Ciberseguridad",
        "XLE": "Energia", "URA": "Uranio", "XME": "Mineria",
        "GLD": "Oro", "SLV": "Plata", "DBA": "Agricolas",
        "XLF": "Bancos", "KIE": "Seguros", "XLI": "Industria",
        "IYT": "Transporte", "ITA": "Defensa",
        "XLP": "Consumo Basico", "XLY": "Consumo Lujo", "XLV": "Salud",
        "XLU": "Utilities", "XLRE": "Inmuebles",
        "EZU": "Europa", "FXI": "China", "EEM": "Emergentes",
        "VTV": "Value", "BTC-USD": "Crypto",
    }
    MACRO_FILTERS = ["SP500", "Nasdaq100", "Small Caps", "Dolar", "Efectivo",
                     "Bonos LP", "Bonos CP", "TIPS", "High Yield", "Inv Grade"]
    RISK_ON_SECTORS = ["Tecnologia", "IA", "Semiconductores", "Crypto",
                       "Small Caps", "IA-Robotica", "Consumo Lujo"]
    DEFENSIVE_SECTORS = ["Consumo Basico", "Utilities", "Salud", "Oro", "Plata"]
    HMM_SEEDS = [42, 123, 256, 789, 1024]
    CONFIDENCE_THRESHOLD = 0.60
    SMOOTHING_WINDOW = 5
    def __init__(self, data_provider):
        self.dp = data_provider
        self.df_p = pd.DataFrame()
        self.df_v = pd.DataFrame()
        self.regime = 1
        self.regime_confidence = 0.0
        self.z_risk = 0.0
        self.hmm_debug = {}
    def _load_prices(self):
        tickers = list(self.LCRACK_ETFS.keys())
        all_data = self.dp.get_prices_bulk(tickers, period="6y")
        raw, vols = {}, {}
        missing = []
        for ticker, sector in self.LCRACK_ETFS.items():
            if sector in raw:
                continue
            df = all_data.get(ticker)
            if df is None or len(df) < 100:
                missing.append(f"{ticker} ({sector})")
                continue
            raw[sector] = df["close"].rename(sector)
            if "volume" in df.columns:
                vols[sector] = df["volume"].rename(sector)
        if missing:
            print(f"    ETFs sin datos: {', '.join(missing)}")
        if not raw:
            raise ValueError("ERROR: Sin datos de ningun ETF.")
        self.df_p = pd.DataFrame(raw).ffill().dropna(how="all")
        self.df_v = pd.DataFrame(vols).reindex(self.df_p.index).ffill().fillna(0) if vols else pd.DataFrame()
        for col in self.df_p.columns:
            v0 = self.df_p[col].first_valid_index()
            if v0 and self.df_p[col].loc[v0] != 0:
                self.df_p[col] = self.df_p[col] / self.df_p[col].loc[v0] * 100
        print(f"    Universo cargado: {len(self.df_p.columns)} sectores, "
              f"{len(self.df_p)} dias de datos")
    # --------------------------------------------------------
    # HMM ENSEMBLE (sin cambios en la logica core,
    #               solo fix en window_labels)
    # --------------------------------------------------------
    def _detect_regime_hmm_stable(self):
        spy = self.df_p.get("SP500")
        if spy is None or _hmm_lib is None:
            return 1, 0.0, 0.0, 0.0, {}
        rets = spy.pct_change().dropna()
        vol = rets.rolling(21).std().dropna()
        common = rets.index.intersection(vol.index)
        if len(common) < 200:
            return 1, 0.0, 0.0, 0.0, {}
        X = np.column_stack([rets.loc[common].values, vol.loc[common].values])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        ensemble_votes = []
        ensemble_smoothed = []
        ensemble_probs = []
        for seed in self.HMM_SEEDS:
            try:
                model = _hmm_lib.GaussianHMM(
                    n_components=3, covariance_type="full",
                    n_iter=500, min_covar=0.001, random_state=seed
                )
                model.fit(X_scaled)
                states = model.predict(X_scaled)
                means = [X_scaled[states == s, 0].mean() for s in range(3)]
                order = np.argsort(means)
                remap = {order[0]: 0, order[1]: 1, order[2]: 2}
                mapped_states = np.array([remap[s] for s in states])
                last_n = mapped_states[-self.SMOOTHING_WINDOW:]
                counts = np.bincount(last_n, minlength=3)
                smoothed_regime = np.argmax(counts)
                probs = model.predict_proba(X_scaled)
                last_state_original = states[-1]
                prob_last = probs[-1, last_state_original]
                ensemble_votes.append(smoothed_regime)
                ensemble_smoothed.append(last_n.tolist())
                ensemble_probs.append(prob_last)
            except Exception:
                ensemble_votes.append(1)
                ensemble_smoothed.append([1] * self.SMOOTHING_WINDOW)
                ensemble_probs.append(0.33)
        vote_counts = np.bincount(ensemble_votes, minlength=3)
        majority_regime = np.argmax(vote_counts)
        majority_pct = vote_counts[majority_regime] / len(ensemble_votes)
        if majority_pct < self.CONFIDENCE_THRESHOLD:
            final_regime = 1
            confidence = majority_pct
        else:
            final_regime = majority_regime
            confidence = majority_pct
        avg_prob = np.mean(ensemble_probs)
        last_obs = X_scaled[-1]
        labels_map = {0: "BEAR", 1: "LATERAL", 2: "BULL"}
        vote_labels = [labels_map[v] for v in ensemble_votes]
        # ── FIX v7.3: Consenso temporal de los 5 modelos por dia ──
        # Antes: solo ensemble_smoothed[0] (semilla 42)
        # Ahora: voto mayoritario de las 5 semillas para cada dia
        window_consensus = []
        if ensemble_smoothed:
            for day_idx in range(self.SMOOTHING_WINDOW):
                day_votes = [ensemble_smoothed[seed_idx][day_idx]
                             for seed_idx in range(len(ensemble_smoothed))]
                day_counts = np.bincount(day_votes, minlength=3)
                window_consensus.append(int(np.argmax(day_counts)))
        window_labels = [labels_map[s] for s in window_consensus]
        # Detalle por semilla (para debug avanzado)
        window_per_seed = {}
        for i, seed in enumerate(self.HMM_SEEDS):
            window_per_seed[str(seed)] = [labels_map[s] for s in ensemble_smoothed[i]]
        debug = {
            "votos": vote_labels,
            "bear": int(vote_counts[0]),
            "lateral": int(vote_counts[1]),
            "bull": int(vote_counts[2]),
            "prob_media": round(float(avg_prob * 100), 1),
            "window": window_labels,            # Consenso 5 modelos
            "window_per_seed": window_per_seed,  # Detalle por semilla
            "consenso_pct": round(float(confidence * 100), 1),
        }
        return final_regime, confidence, float(last_obs[0]), float(last_obs[1]), debug
    # --------------------------------------------------------
    # Z-RISK (sin cambios)
    # --------------------------------------------------------
    def _calc_zrisk(self):
        try:
            hyg = self.df_p.get("High Yield")
            lqd = self.df_p.get("Inv Grade")
            spread = (hyg / lqd).pct_change(21) if hyg is not None and lqd is not None else pd.Series(0, index=self.df_p.index)
            current_returns = self.df_p.pct_change(21).iloc[-1]
            risk_perf = current_returns[[s for s in self.RISK_ON_SECTORS if s in current_returns]].mean()
            def_perf = current_returns[[s for s in self.DEFENSIVE_SECTORS if s in current_returns]].mean()
            spread_z = (spread.iloc[-1] - spread.rolling(63).mean().iloc[-1]) / (spread.rolling(63).std().iloc[-1] + 1e-6)
            bias_z = (risk_perf - def_perf) * 10
            return float(np.clip(bias_z - spread_z, -3, 3))
        except:
            return 0.0
    # --------------------------------------------------------
    # SCORE COMPUESTO — NUEVO v7.3
    # Combina HMM (60%) + Z-Risk (40%) con escala centrada
    # --------------------------------------------------------
    def get_final_regime(self):
        """
        Score compuesto: 60% HMM + 40% Z-Risk.
        Escala centrada para evitar el bug de BEAR(0) × conf = 0:
          BEAR = -1,  LATERAL = 0,  BULL = +1
        Resultado:
          final_centered en [-1, +1] → se reconvierte a 0/1/2
        """
        # 1. HMM puro
        regime_hmm, conf_hmm, ret_z, vol_z, debug = self._detect_regime_hmm_stable()
        # 2. Z-Risk
        z_risk = self._calc_zrisk()
        # 3. Escala centrada
        hmm_centered = regime_hmm - 1                          # -1, 0, +1
        hmm_score = hmm_centered * conf_hmm                    # -1.0 a +1.0
        zrisk_norm = float(np.clip(z_risk / 3.0, -1.0, 1.0))  # -1.0 a +1.0
        # 4. Score compuesto
        final_centered = 0.6 * hmm_score + 0.4 * zrisk_norm
        final_regime = int(np.clip(round(final_centered + 1), 0, 2))
        # 5. Confianza ajustada por conflicto HMM vs resultado final
        conflict = abs(regime_hmm - final_regime)
        if conflict == 0:
            final_conf = conf_hmm                # Sin conflicto: confianza intacta
        elif conflict == 1:
            final_conf = conf_hmm * 0.70         # 1 nivel de diferencia: -30%
        else:
            final_conf = conf_hmm * 0.50         # 2 niveles: -50%
        # 6. Logging
        regime_labels = {0: "BEAR", 1: "LATERAL", 2: "BULL"}
        print(f"\n  SCORE COMPUESTO (60% HMM + 40% Z-Risk):")
        print(f"    HMM puro    : {regime_labels[regime_hmm]} ({conf_hmm:.0%})")
        print(f"    Z-Risk      : {z_risk:+.2f} -> bias normalizado {zrisk_norm:+.2f}")
        print(f"    hmm_score   : {hmm_score:+.3f}")
        print(f"    final_score : {final_centered:+.3f} -> {regime_labels[final_regime]}")
        print(f"    Confianza   : {final_conf:.0%}" +
              (f" (reducida por conflicto)" if conflict > 0 else ""))
        if regime_hmm != final_regime:
            print(f"    ** OVERRIDE : HMM decia {regime_labels[regime_hmm]}"
                  f" pero Z-Risk ({z_risk:+.2f}) corrigio a"
                  f" {regime_labels[final_regime]} **")
        # 7. Añadir detalles compuestos al debug
        debug['composite'] = {
            'hmm_raw': regime_hmm,
            'hmm_label': regime_labels[regime_hmm],
            'hmm_conf': round(conf_hmm * 100, 1),
            'z_risk': round(z_risk, 2),
            'zrisk_norm': round(zrisk_norm, 3),
            'hmm_score': round(hmm_score, 3),
            'final_score': round(final_centered, 3),
            'final_regime': final_regime,
            'final_label': regime_labels[final_regime],
            'final_conf': round(final_conf * 100, 1),
            'override': regime_hmm != final_regime,
        }
        return final_regime, final_conf, ret_z, vol_z, debug, z_risk
    # --------------------------------------------------------
    # METRICAS SECTORIALES — ESTABILIZADAS v7.3
    # --------------------------------------------------------
    def _calc_metrics(self):
        # ── GF_Z: ACELERACION ──
        # ANTES: pct_change(5).rolling(5) → ~8 dias efectivos, 1 dia = 12% de la muestra
        # AHORA: pct_change(5).rolling(10) → ~14 dias efectivos, 1 dia = 7% de la muestra
        # Denominador: min_periods=40 (antes 20) para std mas estable
        gf_fast = self.df_p.pct_change(5).rolling(10, min_periods=5).mean()
        gf_slow = self.df_p.pct_change(20).rolling(20).mean()
        gf_accel = (gf_fast - gf_slow) * 100
        gf_vol = gf_accel.rolling(63, min_periods=40).std()
        gf_z = gf_accel.iloc[-1] / gf_vol.iloc[-1].replace(0, np.nan)
        gf_z = gf_z.fillna(0)
        # ── ROT_Z: ROTACION RELATIVA ──
        # ANTES: rolling(7) → std de 7 datos es muy ruidosa
        # AHORA: rolling(14) → std mas estable, 1 dia = 7% de la muestra
        w = self.df_p.divide(self.df_p.sum(axis=1), axis=0)
        w_mean = w.rolling(14, min_periods=7).mean()
        w_std = w.rolling(14, min_periods=7).std()
        delta_w = w.iloc[-1] - w_mean.iloc[-1]
        rot_z = delta_w / (w_std.iloc[-1] + 0.0001)
        # ── RVOL: VOLUMEN RELATIVO ──
        # ANTES: vol_hoy / vol_avg_20 → un viernes raro y salta todo
        # AHORA: vol_avg_5 / vol_avg_20 → suaviza dias anomalos
        rvol = pd.Series(1.0, index=self.df_p.columns)
        if not self.df_v.empty:
            vol_avg_5 = self.df_v.rolling(5, min_periods=3).mean().iloc[-1]
            vol_avg_20 = self.df_v.rolling(20, min_periods=10).mean().iloc[-1]
            rvol = vol_avg_5 / (vol_avg_20 + 1e-6)
        return gf_z, rot_z, rvol
    def _check_exit_signals(self, gf_z, rvol):
        if gf_z > 1.5 and rvol < 0.8:
            return "SALIDA SUGERIDA", "Precio sube sin volumen (Trampa)."
        if abs(gf_z) < 0.5 and rvol > 2.0:
            return "DISTRIBUCION", "Alto volumen sin precio. Instituciones vendiendo."
        return None, None
    def _assign_status_by_percentile(self, df):
        df['_score'] = (
            0.50 * df['GF_Z'].rank(pct=True) +
            0.25 * df['Rot_Z'].rank(pct=True) +
            0.25 * df['RVOL'].rank(pct=True)
        )
        statuses, notes, emojis = [], [], []
        for i, r in df.iterrows():
            exit_status, exit_note = self._check_exit_signals(r['GF_Z'], r['RVOL'])
            if exit_status:
                statuses.append(exit_status)
                notes.append(exit_note)
                emojis.append("exit")
                continue
            pct = r['_score']
            if pct >= 0.85:
                if r['Rot_Z'] > 0.5 and r['RVOL'] > 1.0:
                    statuses.append("LIDER INSTITUCIONAL")
                    notes.append("Confluencia: momentum + rotacion + volumen.")
                    emojis.append("whale")
                elif r['GF_Z'] > 0 and r['RVOL'] > 1.0:
                    statuses.append("LIDER TECNICO")
                    notes.append("Aceleracion con volumen.")
                    emojis.append("rocket")
                else:
                    statuses.append("LIDER TECNICO")
                    notes.append("Top del ranking relativo.")
                    emojis.append("rocket")
            elif pct >= 0.65:
                statuses.append("MEJORANDO")
                notes.append("Ganando fuerza relativa.")
                emojis.append("up")
            elif pct >= 0.35:
                statuses.append("NEUTRO")
                notes.append("Sin senal clara.")
                emojis.append("neutral")
            elif pct >= 0.15:
                statuses.append("DEBILITANDOSE")
                notes.append("Perdiendo fuerza relativa.")
                emojis.append("down")
            else:
                if r['GF_Z'] < -1.5 and r['Rot_Z'] < -0.5:
                    statuses.append("SALIENDO")
                    notes.append("Salida activa de capital.")
                    emojis.append("exit")
                else:
                    statuses.append("REZAGADO")
                    notes.append("Ultimo del ranking.")
                    emojis.append("warning")
        df['Status'] = statuses
        df['Notas'] = notes
        df['Emoji'] = emojis
        df = df.drop(columns=['_score'])
        return df
    # --------------------------------------------------------
    # RUN — Usa get_final_regime() en vez de llamadas separadas
    # --------------------------------------------------------
    def run(self):
        print("=" * 75)
        print("  L'CRACK v7.3 — HMM + Score Compuesto + Estabilidad")
        print("=" * 75)
        if self.df_p.empty:
            self._load_prices()
        # ── REGIMEN: Score compuesto (HMM 60% + Z-Risk 40%) ──
        regime, confidence, ret_z, vol_z, debug, z_risk = self.get_final_regime()
        self.regime = regime
        self.regime_confidence = confidence
        self.z_risk = z_risk
        self.hmm_debug = debug
        regime_labels = {0: "BEAR (Miedo)", 1: "LATERAL (Indecision)", 2: "BULL (Optimismo)"}
        hmm_raw = debug.get('composite', {}).get('hmm_raw', regime)
        hmm_label = debug.get('composite', {}).get('hmm_label', '?')
        override = debug.get('composite', {}).get('override', False)
        print(f"\n  REGIMEN FINAL")
        print(f"  Estado Compuesto : {regime_labels.get(regime, '?')}")
        print(f"  Confianza Final  : {confidence*100:.0f}%")
        print(f"  HMM Puro         : {hmm_label} ({debug.get('consenso_pct', 0)}%)")
        print(f"  Z-Risk Sistemico : {z_risk:+.2f}")
        print(f"  Input Retorno    : {ret_z:+.2f} Sigmas")
        print(f"  Input Volatil.   : {vol_z:+.2f} Sigmas")
        if override:
            print(f"  ** OVERRIDE ACTIVO: Z-Risk corrigio el regimen **")
        # ── METRICAS SECTORIALES (estabilizadas) ──
        gf_z_s, rot_z_s, rvol_s = self._calc_metrics()
        investable = [s for s in self.df_p.columns if s not in self.MACRO_FILTERS]
        rows = []
        for sector in investable:
            gfz = float(gf_z_s.get(sector, 0))
            rotz = float(rot_z_s.get(sector, 0))
            rv = float(rvol_s.get(sector, 1.0))
            etf_ticker = [k for k, v in self.LCRACK_ETFS.items() if v == sector][0]
            rows.append({
                "Sector": sector, "ETF": etf_ticker,
                "GF_Z": round(gfz, 2), "Rot_Z": round(rotz, 2), "RVOL": round(rv, 2),
            })
        df = pd.DataFrame(rows)
        df = self._assign_status_by_percentile(df)
        df = df.sort_values('GF_Z', ascending=False).reset_index(drop=True)
        # Print tabla
        print(f"\n  {'#':<3} {'Sector':<16} {'ETF':<8} {'GF_Z':>7} {'Rot_Z':>7} {'RVOL':>6} {'Status':<25}")
        print("  " + "-" * 90)
        for i, r in df.iterrows():
            emoji_map = {"whale": "D ", "rocket": "* ", "up": "+ ", "neutral": "  ",
                         "down": "- ", "exit": "X ", "warning": "! ", "danger": "! "}
            prefix = emoji_map.get(r.get('Emoji', ''), '  ')
            print(f"{prefix}{i+1:<3} {r['Sector']:<16} {r['ETF']:<8} {r['GF_Z']:>7.2f} "
                  f"{r['Rot_Z']:>7.2f} {r['RVOL']:>6.2f} {r['Status']:<25}")
            show_note = (
                "LIDER" in r['Status'] or "SALIDA" in r['Status'] or
                "DISTRIBUCION" in r['Status'] or "SALIENDO" in r['Status'] or
                abs(r['GF_Z']) > 2.0
            )
            if show_note:
                print(f"      > {r['Notas']}")
        print("\n" + "=" * 90)
        # Build output dict for JSON
        sectors_list = []
        for _, r in df.iterrows():
            sectors_list.append({
                "Sector": r['Sector'], "ETF": r['ETF'],
                "GF_Z": r['GF_Z'], "Rot_Z": r['Rot_Z'], "RVOL": r['RVOL'],
                "Status": r['Status'], "Notas": r['Notas'],
                "Emoji": r.get('Emoji', 'neutral'),
            })
        self.regime_data = {
            "regime": int(regime),
            "regime_label": regime_labels.get(regime, "?"),
            "consensus": round(confidence * 100, 1),
            "ret_z": round(ret_z, 2),
            "vol_z": round(vol_z, 2),
            "z_risk": round(z_risk, 2),
            "details": debug,
            "sectors": sectors_list,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.df_result = df
        return self.regime_data
# ============================================================
# PISTOLERO v3.2 — FACTORES ANTICIPATORIOS (sin cambios)
# ============================================================
class PistoleroV32:
    FALLBACK_UNIVERSE = [
        "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
        "MU", "QCOM", "AMD", "INTC", "MRVL", "LRCX", "KLAC", "AMAT", "ADI", "TXN",
        "CRM", "ORCL", "NOW", "ADBE", "PANW", "ZS", "CRWD", "SNOW", "PLTR", "DDOG",
        "JPM", "V", "MA", "GS", "BLK", "MS", "AXP", "C", "BAC", "WFC", "SCHW",
        "UNH", "LLY", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "AMGN", "GILD", "ISRG",
        "CAT", "HON", "GE", "RTX", "LMT", "BA", "DE", "UPS", "FDX", "MMM", "EMR",
        "PG", "COST", "HD", "WMT", "NKE", "SBUX", "MCD", "KO", "PEP", "CL", "EL",
        "XOM", "CVX", "COP", "SLB", "OXY", "EOG", "MPC", "PSX", "VLO",
        "NFLX", "DIS", "CMCSA", "TMUS", "T", "VZ", "CHTR",
        "NEE", "DUK", "SO", "D", "AEP", "AMT", "PLD", "CCI", "EQIX",
        "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE", "MBG.DE", "BMW.DE",
        "MUV2.DE", "ADS.DE", "IFX.DE", "HEN3.DE", "DPW.DE",
        "SAN.MC", "ITX.MC", "IBE.MC", "BBVA.MC", "TEF.MC", "AMS.MC", "FER.MC",
        "REP.MC", "ENG.MC", "GRF.MC",
        "DELL", "COHR", "LITE", "GLW", "B",
    ]
    FAVORITES = ["DELL", "COHR", "LITE", "GLW", "B", "AAPL", "NVDA", "MSFT",
                 "GOOGL", "AVGO", "MU", "ZS", "CAT", "NFLX"]
    WEIGHTS = {
        2: {  # BULL
            'momentum': 0.12, 'tendencia': 0.12, 'baja_vol': 0.08, 'slope': 0.08,
            'aceleracion': 0.15, 'vol_crece': 0.10, 'rebote': 0.08, 'rsi_optimo': 0.12,
            'macd_giro': 0.10, 'squeeze': 0.05,
        },
        1: {  # LATERAL
            'momentum': 0.08, 'tendencia': 0.08, 'baja_vol': 0.07, 'slope': 0.07,
            'aceleracion': 0.12, 'vol_crece': 0.12, 'rebote': 0.12, 'rsi_optimo': 0.15,
            'macd_giro': 0.12, 'squeeze': 0.07,
        },
        0: {  # BEAR
            'momentum': 0.05, 'tendencia': 0.05, 'baja_vol': 0.10, 'slope': 0.05,
            'aceleracion': 0.08, 'vol_crece': 0.10, 'rebote': 0.20, 'rsi_optimo': 0.12,
            'macd_giro': 0.15, 'squeeze': 0.10,
        },
    }
    def __init__(self, data_provider, regime=1):
        self.dp = data_provider
        self.regime = regime
    def _scrape_universe(self):
        headers = {'User-Agent': 'Mozilla/5.0'}
        indices = [
            ('S&P500', 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 0, ''),
            ('Nasdaq100', 'https://en.wikipedia.org/wiki/Nasdaq-100', 4, ''),
            ('DAX', 'https://en.wikipedia.org/wiki/DAX', 4, '.DE'),
            ('IBEX35', 'https://es.wikipedia.org/wiki/IBEX_35', 1, '.MC'),
        ]
        all_tickers = []
        ticker_to_index = {}
        for idx_name, url, table_idx, suffix in indices:
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                tables = pd.read_html(StringIO(resp.text))
                if table_idx >= len(tables):
                    continue
                df = tables[table_idx]
                col = None
                for c in df.columns:
                    c_str = str(c).lower()
                    if any(x in c_str for x in ['symbol', 'ticker', 'componente']):
                        col = c
                        break
                if col is None:
                    continue
                count = 0
                for t in df[col].astype(str):
                    clean = t.split(' ')[0].replace('.', '-').strip()
                    if suffix:
                        base = clean.split('-')[0]
                        clean = base + suffix
                    if clean and len(clean) > 0 and clean != 'nan':
                        all_tickers.append(clean)
                        ticker_to_index[clean] = idx_name
                        count += 1
                print(f"    {idx_name}: {count} tickers")
            except Exception as e:
                print(f"    {idx_name}: scraping fallo ({e})")
                continue
        return list(set(all_tickers)), ticker_to_index
    def _get_sector_map_batch(self, tickers):
        sector_map = {}
        def get_sector(ticker):
            try:
                info = yf.Ticker(ticker).info
                return ticker, info.get('sector', 'N/A')
            except:
                return ticker, 'N/A'
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(get_sector, t): t for t in tickers}
            for future in as_completed(futures):
                ticker, sector = future.result()
                sector_map[ticker] = sector
        return sector_map
    def _rsi_quality_score(self, rsi):
        if 45 <= rsi <= 60:
            return 1.0
        elif 60 < rsi <= 70:
            return 0.85
        elif 70 < rsi <= 80:
            return 0.60
        elif rsi > 80:
            return 0.35
        elif 30 <= rsi < 45:
            return 0.90
        else:
            return 0.70
    def run(self, regime=None, max_tickers=None):
        if regime is not None:
            self.regime = regime
        print("=" * 75)
        print("  PISTOLERO v3.2 — Factores Anticipatorios")
        print("=" * 75)
        regime_labels = {0: "BEAR", 1: "LATERAL", 2: "BULL"}
        print(f"  Regimen: {regime_labels.get(self.regime, '?')}")
        # 1. UNIVERSO
        print(f"\n  Obteniendo universo por scraping...")
        scraped, ticker_to_index = self._scrape_universe()
        if len(scraped) > 50:
            for f in self.FAVORITES:
                if f not in scraped:
                    scraped.append(f)
                    ticker_to_index[f] = "Favorito"
            tickers = scraped
            print(f"  Universo dinamico: {len(tickers)} tickers")
        else:
            print(f"  Scraping insuficiente. Usando fallback.")
            tickers = self.FALLBACK_UNIVERSE.copy()
            for f in self.FAVORITES:
                if f not in tickers:
                    tickers.append(f)
            ticker_to_index = {t: "Fallback" for t in tickers}
            print(f"  Universo fallback: {len(tickers)} tickers")
        if max_tickers and len(tickers) > max_tickers:
            fav_set = set(self.FAVORITES)
            others = [t for t in tickers if t not in fav_set]
            np.random.seed(42)
            selected = list(np.random.choice(others,
                            size=min(max_tickers - len(self.FAVORITES), len(others)),
                            replace=False))
            tickers = list(fav_set) + selected
        # 2. PRECIOS
        print(f"\n  Descargando precios...")
        all_data = self.dp.get_prices_bulk(tickers, period="2y")
        valid_tickers = [t for t in tickers if t in all_data and len(all_data[t]) >= 60]
        failed = len(tickers) - len(valid_tickers)
        print(f"  Tickers validos: {len(valid_tickers)}/{len(tickers)}" +
              (f" ({failed} sin datos)" if failed else ""))
        # 3. SECTORES
        print(f"  Obteniendo sectores ({len(valid_tickers)} tickers)...")
        sector_map = self._get_sector_map_batch(valid_tickers)
        # 4. FEATURES Y FACTORES
        print(f"  Calculando features + 10 factores por ticker...")
        rows = []
        errors = 0
        weights = self.WEIGHTS.get(self.regime, self.WEIGHTS[1])
        for idx, ticker in enumerate(valid_tickers):
            if (idx + 1) % 100 == 0:
                print(f"    ... {idx+1}/{len(valid_tickers)} procesados")
            try:
                feat = self.dp.compute_features(ticker, period="2y")
                if feat is None or len(feat) < 60:
                    errors += 1
                    continue
                last = feat.iloc[-1]
                # FACTOR 1: MOMENTUM AJUSTADO POR RIESGO
                ret_63 = last.get('ret_63d', 0) or 0
                ret_126 = last.get('ret_126d', 0) or 0
                vol_63 = last.get('vol_real_63', 0.3) or 0.3
                f_momentum = (0.6 * ret_63 + 0.4 * ret_126) / vol_63 if vol_63 > 0 else 0
                # FACTOR 2: TENDENCIA
                f_tendencia = last.get('dist_ma200', 0) or 0
                # FACTOR 3: BAJA VOLATILIDAD
                f_baja_vol = -(last.get('vol_real_21', 0.3) or 0.3)
                # FACTOR 4: SLOPE POSITIVO
                f_slope = last.get('slope_20', 0) or 0
                # FACTOR 5: ACELERACION RECIENTE
                f_aceleracion = last.get('accel_5v20', 0) or 0
                if pd.isna(f_aceleracion):
                    r5 = last.get('ret_5d', 0) or 0
                    r21 = last.get('ret_21d', 0) or 0
                    f_aceleracion = r5 - r21
                # FACTOR 6: VOLUMEN CRECIENTE
                f_vol_crece = last.get('vol_trend', 0) or 0
                if pd.isna(f_vol_crece):
                    f_vol_crece = 0
                # FACTOR 7: REBOTE DESDE MINIMOS
                dist_min = last.get('dist_min20', 0.05) or 0.05
                if pd.isna(dist_min):
                    dist_min = 0.05
                f_rebote = max(0, 0.10 - dist_min) * 10
                # FACTOR 8: RSI EN ZONA OPTIMA
                rsi = last.get('rsi_14', 50) or 50
                if pd.isna(rsi):
                    rsi = 50
                if 40 <= rsi <= 60:
                    f_rsi = 1.0
                elif 60 < rsi <= 70:
                    f_rsi = 0.7
                elif rsi > 70:
                    f_rsi = 0.2
                elif 30 <= rsi < 40:
                    f_rsi = 0.8
                else:
                    f_rsi = 0.5
                # FACTOR 9: MACD GIRANDO
                f_macd_giro = last.get('macd_hist_delta', 0) or 0
                if pd.isna(f_macd_giro):
                    f_macd_giro = 0
                # FACTOR 10: SQUEEZE
                vol_sq = last.get('vol_squeeze', 1.0) or 1.0
                if pd.isna(vol_sq):
                    vol_sq = 1.0
                f_squeeze = max(0, 1.0 - vol_sq)
                rows.append({
                    'ticker': ticker,
                    'sector': sector_map.get(ticker, 'N/A'),
                    'index': ticker_to_index.get(ticker, 'N/A'),
                    'f_momentum': f_momentum,
                    'f_tendencia': f_tendencia,
                    'f_baja_vol': f_baja_vol,
                    'f_slope': f_slope,
                    'f_aceleracion': f_aceleracion,
                    'f_vol_crece': f_vol_crece,
                    'f_rebote': f_rebote,
                    'f_rsi': f_rsi,
                    'f_macd_giro': f_macd_giro,
                    'f_squeeze': f_squeeze,
                    'rsi_14': rsi,
                    'slope_20': f_slope,
                    'accel': f_aceleracion,
                    'close': float(all_data[ticker]['close'].iloc[-1]),
                })
            except:
                errors += 1
                continue
        if not rows:
            print("  Sin datos suficientes.")
            return {"stocks": [], "regime": self.regime, "total_analyzed": 0}
        if errors > 0:
            print(f"    {errors} tickers con errores")
        df = pd.DataFrame(rows).set_index('ticker')
        # 5. RANKING MULTI-FACTOR
        def rank_norm(s):
            return s.rank(pct=True).fillna(0.5)
        factor_cols = {
            'momentum': 'f_momentum', 'tendencia': 'f_tendencia',
            'baja_vol': 'f_baja_vol', 'slope': 'f_slope',
            'aceleracion': 'f_aceleracion', 'vol_crece': 'f_vol_crece',
            'rebote': 'f_rebote', 'rsi_optimo': 'f_rsi',
            'macd_giro': 'f_macd_giro', 'squeeze': 'f_squeeze',
        }
        score = pd.Series(0.0, index=df.index)
        for factor_name, col_name in factor_cols.items():
            w = weights[factor_name]
            if factor_name == 'rsi_optimo':
                score += w * df[col_name]
            else:
                score += w * rank_norm(df[col_name])
        rsi_multiplier = df['rsi_14'].apply(self._rsi_quality_score)
        df['pistolero_score'] = score * rsi_multiplier
        df = df.sort_values('pistolero_score', ascending=False)
        # Print
        print(f"\n  {'#':<4} {'Ticker':<9} {'Sector':<24} {'Idx':<10} "
              f"{'Score':>7} {'RSI':>6} {'Accel':>8} {'Slope':>8}")
        print("  " + "-" * 90)
        for idx_num, (ticker, row) in enumerate(df.head(30).iterrows(), 1):
            rsi_val = row['rsi_14']
            rsi_tag = " HOT" if rsi_val > 70 else " COLD" if rsi_val < 30 else ""
            idx_str = str(row.get('index', 'N/A'))[:9]
            print(f"  {idx_num:<4} {ticker:<9} {row['sector']:<24} {idx_str:<10} "
                  f"{row['pistolero_score']:7.3f} {rsi_val:>5.1f}{rsi_tag} "
                  f"{row['accel']:>+7.4f} {row['slope_20']:>+7.4f}")
        # Bottom 5
        print(f"\n  --- BOTTOM 5 (EVITAR) ---")
        for idx_num, (ticker, row) in enumerate(df.tail(5).iterrows(), 1):
            print(f"  {idx_num:<4} {ticker:<9} {row['sector']:<24} "
                  f"{row['pistolero_score']:7.3f} {row['rsi_14']:>5.1f}")
        # Favoritos
        fav_in_df = [t for t in self.FAVORITES if t in df.index]
        if fav_in_df:
            print(f"\n  FAVORITOS:")
            print(f"  {'Ticker':<12} {'Score':>7} {'Rank':>10} {'RSI':>6} {'Sector':<24}")
            print("  " + "-" * 65)
            fav_df = df.loc[fav_in_df].sort_values('pistolero_score', ascending=False)
            total = len(df)
            for ticker, row in fav_df.iterrows():
                rank = list(df.index).index(ticker) + 1
                print(f"  {ticker:<12} {row['pistolero_score']:7.3f} "
                      f"{rank:>4}/{total} {row['rsi_14']:>5.1f} {row['sector']:<24}")
        # Stats
        top30_rsi = df.head(30)['rsi_14']
        print(f"\n  {len(df)} activos analizados.")
        print(f"  Regimen: {regime_labels.get(self.regime, '?')}")
        print(f"  RSI medio Top 30: {top30_rsi.mean():.1f}")
        print(f"  RSI > 70 en Top 30: {(top30_rsi > 70).sum()}")
        print(f"  RSI 40-60 en Top 30: {((top30_rsi >= 40) & (top30_rsi <= 60)).sum()}")
        # Build output for JSON
        stocks_list = []
        for ticker, row in df.iterrows():
            stocks_list.append({
                "ticker": ticker,
                "sector": row['sector'],
                "index": row['index'],
                "score": round(float(row['pistolero_score']), 3),
                "rsi": round(float(row['rsi_14']), 1),
                "accel": round(float(row.get('accel', 0)), 4),
                "slope": round(float(row.get('slope_20', 0)), 4),
                "close": round(float(row.get('close', 0)), 2),
                "is_favorito": ticker in self.FAVORITES,
            })
        return {
            "stocks": stocks_list,
            "regime": int(self.regime),
            "total_analyzed": len(df),
            "timestamp": datetime.utcnow().isoformat(),
        }
# ============================================================
# EJECUCION PRINCIPAL
# ============================================================
def main():
    print("=" * 75)
    print("  TRINIDAD SYSTEM — Ejecucion Diaria")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("  L'CRACK v7.3 + PISTOLERO v3.2")
    print("=" * 75)
    dp = DataProvider()
    # 1. L'CRACK
    print("\n" + "=" * 40)
    print("  PASO 1: L'CRACK SECTORIAL v7.3")
    print("=" * 40)
    lcrack = LCrackV73(dp)
    lcrack_data = lcrack.run()
    regime = lcrack.regime
    # 2. PISTOLERO
    print("\n" + "=" * 40)
    print("  PASO 2: PISTOLERO v3.2")
    print("=" * 40)
    pistolero = PistoleroV32(dp, regime=regime)
    pistolero_data = pistolero.run()
    # 3. Guardar resultados
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    combined = {
        "lcrack": lcrack_data,
        "pistolero": pistolero_data,
        "generated_at": datetime.utcnow().isoformat(),
        "version": "L'CRACK v7.3 + PISTOLERO v3.2",
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\n  Resultados guardados en output/results.json")
    # Resumen
    print("\n" + "=" * 75)
    print("  RESUMEN")
    print("=" * 75)
    regime_labels = {0: "BEAR", 1: "LATERAL", 2: "BULL"}
    composite = lcrack_data.get('details', {}).get('composite', {})
    override_str = " (Z-Risk override)" if composite.get('override', False) else ""
    print(f"  Regimen: {regime_labels.get(regime, '?')} ({lcrack.regime_confidence*100:.0f}% confianza){override_str}")
    if composite.get('override'):
        print(f"    HMM puro decia: {composite.get('hmm_label', '?')}")
        print(f"    Z-Risk ({lcrack.z_risk:+.2f}) lo corrigio a: {composite.get('final_label', '?')}")
    print(f"  Z-Risk: {lcrack.z_risk:+.2f}")
    print(f"  Sectores: {len(lcrack_data.get('sectors', []))}")
    print(f"  Stocks analizados: {pistolero_data.get('total_analyzed', 0)}")
    if pistolero_data.get('stocks'):
        top = pistolero_data['stocks'][0]
        print(f"  Top stock: {top['ticker']} (score: {top['score']})")
    print("  TRINIDAD SYSTEM completado.")
    return combined
if __name__ == "__main__":
    main()
