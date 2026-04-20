import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit.components.v1 as components

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn

def plot_hexbin(x, y, data=None, cmap="YlGnBu_r", bins=30, figsize=(10, 8), title=None):
    """
    简洁的六边形分箱图绘制函数
    
    Parameters:
    x, y: 数据列名或数组
    data: DataFrame (可选)
    cmap: 颜色映射
    bins: 六边形数量
    figsize: 图形大小
    title: 标题
    """
    # 处理数据
    if data is not None:
        x_data = data[x]
        y_data = data[y]
        x_label = x
        y_label = y
    else:
        x_data = np.array(x)
        y_data = np.array(y)
        x_label = "x"
        y_label = "y"
        data = pd.DataFrame({x_label: x_data, y_label: y_data})
    
    # 创建图形
    plt.figure(figsize=figsize, dpi=100)
    
    # 计算宽高比
    width, height = figsize
    ratio = int(height * 0.8)
    
    # 绘制六边形分箱图
    g = sns.jointplot(
        data=data, 
        x=x_label, 
        y=y_label,
        kind='hex',
        height=height * 0.9,
        ratio=ratio,
        space=0.1,
        color="#2ca02c",
        gridsize=bins,
        cmap=cmap
    )
    
    # 设置标题
    if title:
        g.fig.suptitle(title, y=1.02)
    
    # 美化样式
    g.fig.set_facecolor('white')
    g.ax_joint.set_facecolor('white')
    
    # 添加相关系数
    corr = np.corrcoef(x_data, y_data)[0, 1]
    g.ax_joint.text(0.05, 0.95, f'r = {corr:.3f}', 
                    transform=g.ax_joint.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return g.fig

# --- 页面配置 ---
st.set_page_config(page_title="音乐流行度深度分析系统", layout="wide", page_icon="🎵")

# --- 1. 注入自定义 CSS ---
st.markdown("""
    <style>
    /* 基础布局优化 */
    .stMainBlockContainer { margin-top: -70px; }
    
    /* 居中主标题 */
    .main-header {
        text-align: center;
        color: #1DB954;
        font-weight: 800;
        font-size: 1.5rem;
        margin-bottom: 20px;
    }
    
    /* 分析报告文本卡片 */
    .report-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #e1e4e8;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        color: #2D3436;
        line-height: 1.8;
    }
    
    /* 预测结果特显 */
    .prediction-box {
        background: #f1f2f6;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #1DB954;
    }
    </style>
    """, unsafe_allow_html=True)

# 顶部固定导航条
components.html("""
<div style="height: 60px; background: #1DB954; line-height: 60px; text-align: center; 
            font-size: 26px; color: white; font-weight: bold; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            position: fixed; top: 0; left: 0; width: 100%; z-index: 9999;">
    🎵 音乐流行度数据深度分析系统
</div>
""", height=0)

st.sidebar.html("<div style='text-align: center; font-weight: bold; font-size: 1.5rem;'>XXX系统</div><hr>")

# --- 2. 资源加载 ---
@st.cache_resource
def load_assets():
    model = joblib.load('music_popularity_rf.pkl')
    features = joblib.load('feature_columns.pkl')
    test = joblib.load('test.pkl')
    explainer = shap.TreeExplainer(model)
    analyzer = SentimentIntensityAnalyzer()
    df = pd.read_csv("Spotify-2000.csv")
    return model, features, explainer, analyzer, test, df

# 如果没有模型文件，系统会提示并停止
try:
    model, feature_columns, explainer, analyzer, test, df = load_assets()
except:
    st.error("❌ 未找到 'music_popularity_rf.pkl' 或 'feature_columns.pkl' 文件。")
    st.stop()
    
# --- 3. 侧边栏按钮交互 ---
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.image("https://img.shetu66.com/2023/07/11/1689059685175876.png", use_container_width=True)
st.sidebar.markdown("### 🛠️ 系统控制面板")

if "menu" not in st.session_state:
    st.session_state["menu"] = "Metrics"

if st.sidebar.button("📊 数据模型效能评估", use_container_width=True):
    st.session_state["menu"] = "Metrics"
if st.sidebar.button("🔍 特征驱动因子解析", use_container_width=True):
    st.session_state["menu"] = "Importance"
if st.sidebar.button("🔮 流行度模拟预测", use_container_width=True):
    st.session_state["menu"] = "Prediction"
if st.sidebar.button("💻 特征分布探索分析", use_container_width=True):
    st.session_state["menu"] = "cumputer"

# --- 4. 模块逻辑 ---

# 模块 1: 模型效能
if st.session_state["menu"] == "Metrics":
    st.markdown("<div class='main-header'>📊 模型效能与预测基准评估</div>", unsafe_allow_html=True)
    st.html("<hr>")
    col1, col2, col3 = st.columns(3, border=True)
    col1.metric("R² Score (解释力)", "0.325", "及格水平")
    col2.metric("MAE (平均绝对误差)", "9.74", "-10% 偏差")
    col3.metric("RMSE (离群波动误差)", "12.04", "偏大")
    st.html("<hr>")
    with st.container(border=True):
        left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("🎯 实际值 vs. 预测值 分布图")

        fig = px.scatter(x=test[0], y=test[1], labels={'x':'实际流行度', 'y':'预测流行度'},
                         template="plotly_white", opacity=0.7)
        fig.add_shape(type="line", x0=20, y0=20, x1=95, y1=95, line=dict(color="Red", dash="dash"))
        fig.update_traces(marker=dict(size=10, color='#1DB954'))
        
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("📝 深度分析结论")
        st.markdown(f"""
        <div>
            <b>1. 整体解释力评估</b><br>
            当前随机森林回归模型的 R² 为 0.325。这意味着模型捕捉到了数据中 32.5% 的波动逻辑。在极其复杂且受主观情绪驱动的音乐市场，这一表现已经初步建立了对流行趋势的预测基准。<br><br>
            <b>2. 误差来源解读</b><br>
            MAE (9.74) 表现稳健，但 RMSE (12.04) 偏高，揭示了数据中存在“黑马效应”——即部分歌曲虽然音频特征平庸，却因外部机遇获得了极高流行度，导致预测产生了离群误差。
        </div>
        """, unsafe_allow_html=True)

# 模块 2: 特征解析
elif st.session_state["menu"] == "Importance":
    st.markdown("<div class='main-header'>🔍 流行度驱动因子梯队分析</div>", unsafe_allow_html=True)
    st.html("<hr>")
    imp_data = {
        "Feature": ["Year", "Length", "Dutch Indie", "Danceability", "Liveness", "Loudness", "BPM", "Valence", "Title Length", "Energy"],
        "Importance": [0.118, 0.090, 0.084, 0.075, 0.074, 0.074, 0.070, 0.066, 0.065, 0.061]
    }
    df_plot = pd.DataFrame(imp_data).sort_values("Importance", ascending=True)
    
    fig_imp = px.bar(df_plot, x='Importance', y='Feature', orientation='h', 
                     color='Importance', color_continuous_scale='Greens',
                     template="plotly_white", title="核心特征全局影响力权重")
    
    with st.container(border=True):
        st.plotly_chart(fig_imp, use_container_width=True)


    t1, t2, t3 = st.columns(3, border=True)
    t1.write("🕒 **第一梯队：时代特征**\n\n年份的影响力位居首位 (0.118)，证明了音乐市场的流行具有极强的时代周期性，老歌长青与新歌霸榜的现象在模型中得到了体现。")
    t2.write("💃 **第二梯队：节奏律动**\n\n舞蹈性 (Danceability) 与响度 (Loudness) 的权重极其接近，构成了现代商业音乐能否在第一时间抓住听众耳朵的核心物理指标。")
    t3.write("🏷️ **第三梯队：市场标签**\n\n荷兰独立流派 (Dutch Indie) 展示了极强的市场确定性，这说明在垂直领域，流派标签带来的预测收益甚至超过了基本的能量感 (Energy)。")

# 模块 3: 模拟预测
elif st.session_state["menu"] == "Prediction":
    st.markdown("<div class='main-header'>🔮 单曲流行度 AI 预测模拟</div>", unsafe_allow_html=True)
    st.html("<hr>")
    with st.container():
        with st.form("predict_form"):
            st.markdown("##### 🎹 请配置待模拟歌曲的属性参数")
            c1, c2, c3 = st.columns(3)
            with c1:
                title = st.text_input("歌曲名称", "Top Hit Song")
                year = st.number_input("预计发行年份", 1950, 2026, 2024)
                bpm = st.slider("BPM (每分钟节拍数)", 50, 200, 118)
            with c2:
                duration = st.number_input("时长 (单位：秒)", 60, 600, 205)
                dance = st.slider("Danceability (舞蹈性)", 0.0, 1.0, 0.65)
                energy = st.slider("Energy (能量感)", 0.0, 1.0, 0.72)
            with c3:
                genre = st.selectbox("核心流派分类", ["dutch indie", "dutch pop", "album rock", "other"])
                loudness = st.slider("Loudness (响度dB)", -60, 0, -7)
                valence = st.slider("Valence (情绪正向度)", 0.0, 1.0, 0.55)
            
            submit = st.form_submit_button("🚀 执行 AI 深度归因预测", use_container_width=True)

    if submit:
        # 特征工程处理
        input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
        input_df['Year'] = year
        input_df['Length (Duration)'] = duration
        input_df['Beats Per Minute (BPM)'] = bpm
        input_df['Danceability'] = dance
        input_df['Energy'] = energy
        input_df['Loudness (dB)'] = loudness
        input_df['Valence'] = valence
        input_df['Title Length'] = len(title)
        input_df['Title Emotion'] = analyzer.polarity_scores(title)["compound"]
        
        genre_col = f"Genre_Grouped_{genre}"
        if genre_col in input_df.columns:
            input_df[genre_col] = 1

        # 运行模型
        prediction = model.predict(input_df)[0]
        shap_values = explainer.shap_values(input_df)

        st.html("<hr>")
        
        with st.container(border=True):
            st.markdown("#### 📊 模拟预测结果分析")
        
            if prediction > 65:
                st.success(f"🔴 预计流行度得分: {prediction:.2f}, 🌟 具备爆款潜力")
            else: 
                st.info(f"🔴 预计流行度得分: {prediction:.2f}, 🎵 垂直细分作品")

            st.html("<div style='text-align: center;'>特征归因路径 (SHAP Force Analysis)</div><hr>")
            
            # 渲染 SHAP 力图
            fig_shap, ax = plt.subplots(figsize=(12, 3), dpi=100)
            shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0], 
                            matplotlib=True, show=False, text_rotation=0)
            st.pyplot(plt.gcf(), use_container_width=True)
            st.html("<div style='text-align: center; color: red;'>注：红色代表推高流行度的因子，蓝色代表降低流行度的因子。</div>")

# 模块 4: 特征分布探索分析
if st.session_state["menu"] == "cumputer":
    st.markdown("<div class='main-header'>💻 特征分布探索分析</div>", unsafe_allow_html=True)
    st.html("<hr>")
    
    df["Length (Duration)"] = df["Length (Duration)"].apply(lambda x: int(str(x).replace(",","")))

    numeric_col = "Year	Beats Per Minute (BPM)	Energy	Danceability	Loudness (dB)	Liveness	Valence	Length (Duration)	Acousticness	Speechiness	Popularity".split("\t")

    col = st.columns(2)
    x1 = col[0].selectbox("X轴", numeric_col)
    y1 = col[1].selectbox("Y轴", numeric_col)
    
    fig = plot_hexbin(x1, y1, data=df, cmap="Reds", bins=30, figsize=(10, 8), title=f"{x1} vs {y1}")
    
    col = st.columns([1,3,1])
    col[1].pyplot(fig, use_container_width=True)
    
with st.sidebar.container(border=True):
    st.html("""
    <div style='text-align: center;color:red;'>系统说明</div><hr>
    <p style="font-size: 10px;">本系统旨在通过探索分析项目数据中影响音乐流行度的特征，核心目的在于通过模型的特征重要性解释音乐流行度，
    但因为音乐流行度的影响因素包括但不限于数据集所示特征，还有很多社会因素影响，导致预测结果具有较大误差。
    但模型特征重要性对于音乐流行度的支持具有很好的指示作用，可以为音乐是否流行提供一个可评价的参考体系。</p>
    <div style="text-align:center; color:red; font-size:10px;">结果只供参考：不可用着推广决策依据!!!</div>
    """)