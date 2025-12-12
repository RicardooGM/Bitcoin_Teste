import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="BTC Decision Model", layout="wide")

st.title("Modelo Sistemático de Compra/Venda — Bitcoin (BTC)")
st.markdown("""
Este app implementa o modelo de decisão: Tendência (MM50/MM200), Momentum (RSI/MACD), Risco (ATR) e métricas on-chain opcionais (via sliders).
O código trata casos com índices desalinhados, converte valores com segurança e adiciona gráficos separados, cálculo de stops baseado em ATR e export de sinais.
""")

# ----- Sidebar: parâmetros -----
st.sidebar.header("Parâmetros")
period_days = st.sidebar.number_input("Dias de histórico", min_value=90, max_value=3650, value=720, step=30)
start_date = (datetime.utcnow() - timedelta(days=period_days)).date()
end_date = datetime.utcnow().date()

ticker = st.sidebar.text_input("Ticker (yfinance)", value="BTC-USD")

# Indicator params
sma_short = st.sidebar.number_input("SMA Curta (dias)", min_value=5, max_value=200, value=50)
sma_long = st.sidebar.number_input("SMA Longa (dias)", min_value=20, max_value=400, value=200)
rsi_period = st.sidebar.number_input("RSI (period)", min_value=5, max_value=50, value=14)
atr_period = st.sidebar.number_input("ATR (period)", min_value=5, max_value=50, value=14)

st.sidebar.markdown("---")
st.sidebar.markdown("**Métricas on-chain (opcionais)** — se você não tiver, deixe como estão ou ajuste manualmente")
mvrv_slider = st.sidebar.slider("MVRV (estimado)", min_value=-10.0, max_value=20.0, value=3.0, step=0.1)
nupl_slider = st.sidebar.slider("NUPL (estimado)", min_value=-1.0, max_value=1.0, value=0.2, step=0.01)
hashrate_trend = st.sidebar.selectbox("Tendência do Hashrate", options=["Alta", "Estável", "Queda"], index=1)

st.sidebar.markdown("---")
capital = st.sidebar.number_input("Capital disponível (R$ ou USD)", min_value=100.0, value=1000.0, step=100.0)
max_risk_pct = st.sidebar.slider("Risco por operação (% do capital)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

# ----- Fetch data -----
@st.cache_data(ttl=600)
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    # se houver colunas duplicadas, desambiguar
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna()
    return df

with st.spinner("Buscando dados..."):
    df = fetch_data(ticker, start_date, end_date)

if df.empty:
    st.error("Nenhum dado retornado. Verifique o ticker e a conexão.")
    st.stop()

# ----- Indicator calculations -----
price = df['Close'].copy()
high = df['High']
low = df['Low']
close = df['Close']

# SMA
df['SMA_short'] = close.rolling(window=sma_short, min_periods=1).mean()
df['SMA_long'] = close.rolling(window=sma_long, min_periods=1).mean()

# MACD (12,26,9)
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# RSI
delta = close.diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
ema_up = up.ewm(com=rsi_period-1, adjust=False).mean()
ema_down = down.ewm(com=rsi_period-1, adjust=False).mean()
rs = ema_up / (ema_down.replace(0, np.nan))
df['RSI'] = 100 - (100 / (1 + rs))

# ATR
tr1 = high - low
tr2 = (high - close.shift()).abs()
tr3 = (low - close.shift()).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['ATR'] = tr.rolling(window=atr_period, min_periods=1).mean()

# ----- Helpers to safely extract scalars -----
latest = df.iloc[-1]

def to_scalar(value):
    try:
        if isinstance(value, (pd.Series, np.ndarray, list, tuple)):
            arr = np.asarray(value)
            if arr.size == 0:
                return np.nan
            v = arr.ravel()[-1]
        else:
            v = value
        if pd.isna(v):
            return np.nan
        return float(v)
    except Exception:
        return np.nan

price_now = to_scalar(latest.get('Close'))
sma_short_now = to_scalar(latest.get('SMA_short'))
sma_long_now = to_scalar(latest.get('SMA_long'))
macd_now = to_scalar(latest.get('MACD'))
macd_signal_now = to_scalar(latest.get('MACD_signal'))
rsi_now = to_scalar(latest.get('RSI'))
atr_now = to_scalar(latest.get('ATR'))

# ----- Scoring logic -----
score = 0
reasons = []

# Tendência
try:
    if (not pd.isna(price_now)) and (not pd.isna(sma_short_now)) and price_now > sma_short_now:
        score += 1
        reasons.append("Preço acima da SMA curta")
    if (not pd.isna(price_now)) and (not pd.isna(sma_long_now)) and price_now > sma_long_now:
        score += 1
        reasons.append("Preço acima da SMA longa")
except Exception:
    pass

# Cruzamento dourado
cross_up = False
if len(df) >= max(sma_long, sma_short) + 2:
    recent = df[['SMA_short', 'SMA_long']].tail(11)
    try:
        prev_short = float(recent['SMA_short'].values[-2])
        prev_long = float(recent['SMA_long'].values[-2])
        last_short = float(recent['SMA_short'].values[-1])
        last_long = float(recent['SMA_long'].values[-1])
        if (prev_short < prev_long) and (last_short > last_long):
            cross_up = True
    except Exception:
        cross_up = False
if cross_up:
    score += 1
    reasons.append("Cruzamento dourado recente")

# Momentum
macd_cross_up = False
if len(df) >= 2:
    try:
        macd_prev = float(df['MACD'].values[-2])
        macd_signal_prev = float(df['MACD_signal'].values[-2])
        if (macd_prev <= macd_signal_prev) and (macd_now > macd_signal_now):
            macd_cross_up = True
    except Exception:
        macd_cross_up = False
if macd_now > macd_signal_now:
    score += 1
    reasons.append("MACD acima da signal")
else: 
    reasons.append("MACD abaixo de signal")
    

# RSI
try:
    rsi_prev = float(df['RSI'].values[-2]) if len(df) >= 2 and not pd.isna(df['RSI'].values[-2]) else np.nan
    if (not pd.isna(rsi_now)) and (rsi_now < 40) and (not pd.isna(rsi_prev)) and (rsi_now > rsi_prev):
        score += 1
        reasons.append("RSI saindo de sobrevenda")
except Exception:
    pass

# On-chain sliders
if mvrv_slider < 3:
    score += 1
    reasons.append(f"MVRV baixo ({mvrv_slider})")
if (nupl_slider > 0) and (nupl_slider < 0.6):
    score += 1
    reasons.append(f"NUPL em otimismo saudável ({nupl_slider})")

# Risco (ATR/hashrate)
if (not pd.isna(atr_now)) and (atr_now / price_now > 0.06):
    score -= 1
    reasons.append("ATR elevado — alta volatilidade")
if hashrate_trend == 'Queda':
    score -= 1
    reasons.append("Hashrate em queda")

# ----- Interpret action -----
if score >= 5:
    action = 'COMPRA'
elif score <= 2:
    action = 'VENDA'
else:
    action = 'NEUTRO'

# ----- Position sizing & ATR-based stops -----
risk_amount = capital * (max_risk_pct / 100.0)

# stops based on ATR (fallback if ATR NaN)
if (not pd.isna(atr_now)) and atr_now > 0:
    stop_1x = atr_now
    stop_1_5x = 1.5 * atr_now
    stop_2x = 2.0 * atr_now
else:
    stop_1x = price_now * 0.03
    stop_1_5x = price_now * 0.045
    stop_2x = price_now * 0.06

# suggested position size using stop_1_5x
stop_distance = stop_1_5x
position_size = 0
if stop_distance > 0:
    position_size = max(0, risk_amount / stop_distance)
position_value = min(position_size * price_now, capital)

# ----- Layout -----
col_left, col_right = st.columns([2, 1])
with col_left:
    st.subheader("Gráfico de Preço (com SMAs)")
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(df.index, df['Close'], label='Close')
    ax1.plot(df.index, df['SMA_short'], label=f'SMA {sma_short}')
    ax1.plot(df.index, df['SMA_long'], label=f'SMA {sma_long}')
    ax1.set_ylabel("Preço")
    ax1.legend(loc="upper left")
    st.pyplot(fig1)

    st.subheader("MACD")
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(df.index, df['MACD'], label='MACD')
    ax2.plot(df.index, df['MACD_signal'], label='Signal')
    ax2.axhline(0, linestyle='--')
    ax2.set_ylabel("MACD")
    ax2.legend(loc="upper left")
    st.pyplot(fig2)

    st.subheader("RSI")
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    ax3.plot(df.index, df['RSI'], label='RSI')
    ax3.axhline(70, linestyle='--')
    ax3.axhline(30, linestyle='--')
    ax3.set_ylabel('RSI')
    ax3.legend(loc='upper left')
    st.pyplot(fig3)

    st.subheader("ATR")
    fig4, ax4 = plt.subplots(figsize=(12, 3))
    ax4.plot(df.index, df['ATR'], label='ATR')
    ax4.set_ylabel('ATR')
    ax4.legend(loc='upper left')
    st.pyplot(fig4)

with col_right:
    st.subheader("Sinal Atual")
    # destaque do sinal com cor
    if action == 'COMPRA':
        st.success(f"Ação sugerida: {action}")
    elif action == 'VENDA':
        st.error(f"Ação sugerida: {action}")
    else:
        st.info(f"Ação sugerida: {action}")

    st.markdown("**Score:** %d" % score)
    st.markdown("**Motivos:**")
    for r in reasons:
        st.write("- ", r)

    st.markdown("---")
    st.subheader("Indicadores (último)")
    ind_df = pd.DataFrame({
        'Indicador': ['Preço', f'SMA_{sma_short}', f'SMA_{sma_long}', 'MACD', 'MACD_signal', 'RSI', 'ATR'],
        'Valor': [price_now, sma_short_now, sma_long_now, macd_now, macd_signal_now, rsi_now, atr_now]
    })
    st.table(ind_df)

    st.markdown("---")
    st.subheader("Gestão de risco (sugerida)")
    st.write(f"Capital: {capital}")
    st.write(f"Risco por operação: {max_risk_pct} % → {risk_amount:.2f}")
    st.write(f"Stop conservador (1×ATR): {stop_1x:.2f}")
    st.write(f"Stop padrão (1.5×ATR): {stop_1_5x:.2f}")
    st.write(f"Stop agressivo (2×ATR): {stop_2x:.2f}")
    st.write(f"Posição aproximada (unidades) usando stop 1.5×ATR: {position_size:.4f}")
    st.write(f"Valor da posição (limitado ao capital): {position_value:.2f}")

# ----- Signals history (últimos 60 dias) -----
st.markdown('---')
st.subheader('Sinais recentes (últimos 60 dias)')

window = 60
history = df.tail(window).copy()

# Compute daily score for history (simplified: same rules applied row-wise)
def compute_row_score(row):
    s = 0
    try:
        close_v = float(row['Close'])
        sma_s_v = float(row['SMA_short']) if not pd.isna(row['SMA_short']) else np.nan
        sma_l_v = float(row['SMA_long']) if not pd.isna(row['SMA_long']) else np.nan
        macd_v = float(row['MACD']) if not pd.isna(row['MACD']) else np.nan
        macd_sig_v = float(row['MACD_signal']) if not pd.isna(row['MACD_signal']) else np.nan
        rsi_v = float(row['RSI']) if not pd.isna(row['RSI']) else np.nan
        atr_v = float(row['ATR']) if not pd.isna(row['ATR']) else np.nan

        if not pd.isna(sma_s_v) and close_v > sma_s_v:
            s += 1
        if not pd.isna(sma_l_v) and close_v > sma_l_v:
            s += 1
        if not pd.isna(macd_v) and not pd.isna(macd_sig_v) and macd_v > macd_sig_v:
            s += 1
        if not pd.isna(rsi_v) and rsi_v < 40:
            s += 1
        if mvrv_slider < 3:
            s += 1
        if (nupl_slider > 0) and (nupl_slider < 0.6):
            s += 1
        if (not pd.isna(atr_v)) and (atr_v / close_v > 0.06):
            s -= 1
    except Exception:
        pass
    return s

history['score'] = history.apply(lambda r: compute_row_score(r), axis=1)
history['signal'] = history['score'].apply(lambda x: 'BUY' if x>=5 else ('SELL' if x<=2 else 'HOLD'))

st.dataframe(history[['Close','SMA_short','SMA_long','MACD','MACD_signal','RSI','ATR','score','signal']].tail(60))

# download dos sinais
csv = history[['Close','SMA_short','SMA_long','MACD','MACD_signal','RSI','ATR','score','signal']].to_csv(index=True)
st.download_button(label="Exportar sinais (CSV)", data=csv, file_name="btc_signals.csv", mime='text/csv')

st.markdown('---')
st.caption('Este é um modelo educacional. Faça seus próprios testes e backtests antes de usar em produção.')

# === Debug: breakdown de contribuições ===
contrib = {
    'preco_vs_sma_short': 1 if (not pd.isna(price_now) and not pd.isna(sma_short_now) and price_now > sma_short_now) else 0,
    'preco_vs_sma_long': 1 if (not pd.isna(price_now) and not pd.isna(sma_long_now) and price_now > sma_long_now) else 0,
    'golden_cross_recent': 1 if cross_up else 0,
    'macd_cross_up': 1 if macd_cross_up else 0,
    'rsi_recovery': 1 if ((not pd.isna(rsi_now)) and (rsi_now < 40) and (not pd.isna(rsi_prev)) and (rsi_now > rsi_prev)) else 0,
    'mvrv_bonus': 1 if (mvrv_slider < 3) else 0,
    'nupl_bonus': 1 if ((nupl_slider > 0) and (nupl_slider < 0.6)) else 0,
    'atr_penalty': -1 if ((not pd.isna(atr_now)) and (atr_now / price_now > 0.06)) else 0,
    'hashrate_penalty': -1 if (hashrate_trend == 'Queda') else 0
}
# soma segura
debug_score = sum(contrib.values())
# mostrar no app (na coluna direita)
st.markdown('**Debug: Contribution breakdown**')
st.table(pd.DataFrame.from_dict(contrib, orient='index', columns=['value']))
st.write(f'Pontuação (recalculada): {debug_score}  — Pontuação usada no app: {score}')
st.write('Valores (último candle):')
st.table(pd.DataFrame({
    'indicator': ['price_now','sma_short_now','sma_long_now','macd_now','macd_signal_now','rsi_now','atr_now'],
    'value': [price_now,sma_short_now,sma_long_now,macd_now,macd_signal_now,rsi_now,atr_now]
}))
