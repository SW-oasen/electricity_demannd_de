"""
Streamlit web app — Germany hourly energy demand forecast.

Two sections:
  1. Vorhersage (morgen)  — predict the full next day (00:00–23:00 UTC)
  2. Historischer Vergleich — compare predictions vs actual SMARD demand
     over a user-selected date range (max 1 year)

Run with (from workspace root):
    streamlit run src/streamlit_app.py
"""

import sys
import os
# Allow importing sibling modules (fetch_prepare_data, train_model_predict)
# Works whether the app is run from the workspace root or from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import date, timedelta, datetime, timezone

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

from fetch_prepare_data import (
    prepare_future_features,
    fetch_smard_netzlast,
    create_energy_features,
    create_time_based_features,
    prepare_weather_data,
    combine_energy_weather_dataset,
)
from train_model_predict import load_model_from_pickle

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stromverbrauchsprognose Deutschland",
    page_icon="⚡",
    layout="wide",
)

# ── load models once (cached across sessions) ──────────────────────────────────
@st.cache_resource
def load_models():
    _base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    return {
        "LGBM":          load_model_from_pickle(os.path.join(_base, "best_lgbm_model_bayesian.pkl")),
        "Random Forest": load_model_from_pickle(os.path.join(_base, "best_rf_model_bayesian.pkl")),
    }


models = load_models()

# ── page header ────────────────────────────────────────────────────────────────
st.title("⚡ Stromverbrauchsprognose Deutschland")
st.markdown("Stündliche Vorhersage und Vergleich der deutschen Netzlast (SMARD).")

tab_future, tab_hist = st.tabs(["🔮 Vorhersage (morgen)", "📊 Historischer Vergleich"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Future Prediction (tomorrow, full day)
# ══════════════════════════════════════════════════════════════════════════════
with tab_future:
    st.markdown("Vorhersage des Stromverbrauchs für den **nächsten Tag** (00:00–23:00 UTC).")

    now_utc  = datetime.now(timezone.utc)
    tomorrow = date.today() + timedelta(days=1)

    col_info, col_ctrl = st.columns([2, 1])
    with col_info:
        st.markdown(f"**Aktuelle Uhrzeit (UTC):** {now_utc.strftime('%Y-%m-%d %H:%M')}")
        st.markdown(f"**Vorhersagetag:** {tomorrow.isoformat()}")
    with col_ctrl:
        future_model = st.selectbox("Modell", options=list(models.keys()), key="future_model")

    if st.button("Predict for Tomorrow", type="primary", key="btn_future"):
        tomorrow_str = tomorrow.isoformat()

        with st.spinner(f"Features werden vorbereitet für {tomorrow_str} …"):
            try:
                df_future = prepare_future_features(prediction_date=tomorrow_str)
            except Exception as exc:
                st.error(f"Feature-Vorbereitung fehlgeschlagen: {exc}")
                st.stop()

        if df_future.empty:
            st.error("Keine Features zurückgegeben — API-Verbindung prüfen.")
            st.stop()

        with st.spinner(f"{future_model} wird ausgeführt …"):
            X     = df_future.drop(columns=["time", "EnergyDemand"], errors="ignore")
            preds = models[future_model].predict(X)

        st.success(f"Vorhersage abgeschlossen — {tomorrow_str} ({future_model})")

        col_chart, col_table = st.columns([2.5, 1])

        with col_chart:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_future["time"], preds, linewidth=2, color="steelblue",
                    label=f"{future_model} prediction")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.set_xlabel("Stunde (UTC)")
            ax.set_ylabel("Netzlast (MWh)")
            ax.set_title(f"Stromverbrauchsprognose — {tomorrow_str}  [{future_model}]")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            plt.tight_layout()
            st.pyplot(fig)

        with col_table:
            df_result = df_future[["time"]].copy()
            df_result["predicted_MWh"] = preds.round(0).astype(int)
            df_result["Stunde (UTC)"]  = df_result["time"].dt.strftime("%H:%M")
            st.dataframe(
                df_result[["Stunde (UTC)", "predicted_MWh"]].reset_index(drop=True),
                use_container_width=True,
                height=600,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Historical Comparison
# ══════════════════════════════════════════════════════════════════════════════
MAX_RANGE_DAYS = 365

with tab_hist:
    st.markdown(
        "Vorhersage und tatsächlicher Verbrauch (SMARD) im Vergleich. "
        "Maximaler Zeitraum: **1 Jahr**."
    )

    _default_to   = date.today() - timedelta(days=1)
    _default_from = _default_to - timedelta(days=6)
    _max_date     = date.today() - timedelta(days=1)

    col1, col2, col3 = st.columns(3)
    with col1:
        date_from = st.date_input(
            "Von:",
            value=_default_from,
            min_value=date(2019, 1, 1),
            max_value=_max_date,
            key="hist_from",
        )
    with col2:
        date_to = st.date_input(
            "Bis:",
            value=_default_to,
            min_value=date(2019, 1, 1),
            max_value=_max_date,
            key="hist_to",
        )
    with col3:
        hist_model = st.selectbox("Modell", options=list(models.keys()), key="hist_model")

    # ── Range validation ───────────────────────────────────────────────────────
    delta_days = (date_to - date_from).days

    if delta_days < 0:
        st.error('⚠ „Bis"-Datum muss nach dem „Von"-Datum liegen.')
    elif delta_days > MAX_RANGE_DAYS:
        st.warning(
            f"⚠ Gewählter Zeitraum: **{delta_days} Tage** — "
            f"Maximum sind **{MAX_RANGE_DAYS} Tage** (1 Jahr). "
            "Bitte Auswahl einschränken."
        )
    else:
        st.success(f"Zeitraum: {delta_days + 1} Tag(e)  ✓")

        if st.button("Compare Prediction vs Actual", type="primary", key="btn_compare"):
            from_str = str(date_from)
            to_str   = str(date_to)

            # 1. Fetch actual SMARD data ───────────────────────────────────────
            with st.spinner(f"SMARD-Daten werden abgerufen für {from_str} → {to_str} …"):
                try:
                    df_actual = fetch_smard_netzlast(from_str, to_str)
                except Exception as exc:
                    st.error(f"SMARD-Abruf fehlgeschlagen: {exc}")
                    st.stop()

            if df_actual.empty:
                st.error(f"Keine SMARD-Daten verfügbar für {from_str} → {to_str}.")
                st.stop()

            # 2. Build feature matrix ──────────────────────────────────────────
            with st.spinner("Modellfeatures werden berechnet (Energie + Wetter) …"):
                HISTORY_DAYS = 15
                try:
                    hist_start = (
                        pd.to_datetime(from_str) - pd.Timedelta(days=HISTORY_DAYS)
                    ).strftime("%Y-%m-%d")

                    df_energy = fetch_smard_netzlast(hist_start, to_str)
                    df_energy = create_energy_features(df_energy)
                    df_energy = create_time_based_features(
                        df_energy, in_year=pd.to_datetime(to_str).year
                    )
                    df_weather = prepare_weather_data(
                        in_start_date=hist_start, in_end_date=to_str
                    )
                    df_feat = combine_energy_weather_dataset(df_energy, df_weather)
                    df_feat = df_feat.sort_values("time").reset_index(drop=True)

                    from_ts = pd.to_datetime(from_str, utc=True)
                    to_ts   = pd.to_datetime(to_str,   utc=True) + pd.Timedelta(hours=23)
                    df_feat = df_feat[
                        (df_feat["time"] >= from_ts) & (df_feat["time"] <= to_ts)
                    ].reset_index(drop=True)

                except Exception as exc:
                    st.error(f"Feature-Vorbereitung fehlgeschlagen: {exc}")
                    st.stop()

            if df_feat.empty:
                st.error("Keine Feature-Daten für den gewählten Zeitraum.")
                st.stop()

            # 3. Predict ───────────────────────────────────────────────────────
            with st.spinner(f"{hist_model} wird ausgeführt …"):
                X     = df_feat.drop(columns=["time", "EnergyDemand"], errors="ignore")
                preds = models[hist_model].predict(X)

            # 4. Align actual and predicted on shared timestamps ───────────────
            s_pred   = pd.Series(preds, index=df_feat["time"], name="Predicted")
            s_actual = df_actual.set_index("time")["EnergyDemand"].rename("Actual")
            df_plot  = pd.concat([s_actual, s_pred], axis=1).dropna()

            st.success(f"Vergleich abgeschlossen — {from_str} → {to_str} ({hist_model})")

            # 5. Plot ──────────────────────────────────────────────────────────
            days_in_range = delta_days + 1
            if days_in_range <= 3:
                x_fmt = "%m-%d %H:%M"
            elif days_in_range <= 31:
                x_fmt = "%Y-%m-%d"
            else:
                x_fmt = "%Y-%m"

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df_plot.index, df_plot["Actual"],
                    color="steelblue", linewidth=1.5,
                    label="Tatsächlicher Verbrauch (SMARD)")
            ax.plot(df_plot.index, df_plot["Predicted"],
                    color="darkorange", linewidth=1.5, linestyle="--",
                    label=f"Vorhersage ({hist_model})")
            ax.xaxis.set_major_formatter(mdates.DateFormatter(x_fmt))
            ax.set_xlabel("Datum / Uhrzeit (UTC)")
            ax.set_ylabel("Netzlast (MWh)")
            ax.set_title(
                f"Tatsächlicher vs. vorhergesagter Verbrauch — "
                f"{from_str} bis {to_str}  [{hist_model}]"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            plt.tight_layout()
            st.pyplot(fig)

            # 6. Metrics ───────────────────────────────────────────────────────
            mae  = (df_plot["Actual"] - df_plot["Predicted"]).abs().mean()
            rmse = ((df_plot["Actual"] - df_plot["Predicted"]) ** 2).mean() ** 0.5

            m1, m2, m3 = st.columns(3)
            m1.metric("MAE",          f"{mae:,.0f} MWh")
            m2.metric("RMSE",         f"{rmse:,.0f} MWh")
            m3.metric("Datenpunkte",  f"{len(df_plot):,}")
