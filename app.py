from __future__ import annotations

import csv
import io
import html
from datetime import datetime

import streamlit as st

from src.model_manager import ArtifactError, load_model_bundle
from src.prediction import predict_many


st.set_page_config(
    page_title="FeelScope | Emotion AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Visual system ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #f3f5ff;
        --muted: #9ba5c1;
        --panel: rgba(20, 27, 51, .78);
        --line: rgba(172, 185, 230, .14);
        --violet: #a78bfa;
        --mint: #5eead4;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(ellipse at 12% 0%, rgba(114, 80, 220, .23), transparent 34%),
          radial-gradient(ellipse at 90% 14%, rgba(26, 183, 170, .12), transparent 30%),
          linear-gradient(145deg, #080d1b 0%, #0b1021 50%, #0b1224 100%);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(8, 13, 27, .45); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(17, 24, 47, .98), rgba(10, 16, 33, .98));
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.1rem; }
    .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 4rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; color: var(--ink); }
    p, label, span, div { color: inherit; }
    .brand-row { display:flex; align-items:center; gap:12px; margin-bottom:1.7rem; }
    .brand-mark {
        width:46px; height:46px; border-radius:15px; display:flex; align-items:center; justify-content:center;
        font-size:24px; background:linear-gradient(135deg,#8b5cf6,#14b8a6);
        box-shadow:0 10px 32px rgba(139,92,246,.25);
    }
    .brand-name { font:800 1.14rem 'Manrope',sans-serif; color:#f8faff; letter-spacing:-.04em; }
    .brand-caption { font-size:.72rem; color:#9aa6c5; letter-spacing:.13em; text-transform:uppercase; }
    .hero {
        position:relative; overflow:hidden; border:1px solid rgba(181,190,255,.17);
        border-radius:28px; padding:34px 36px 32px;
        background:linear-gradient(112deg,rgba(31,39,77,.94),rgba(20,39,61,.83) 62%,rgba(19,74,76,.6));
        box-shadow:0 22px 70px rgba(0,0,0,.22); margin-bottom:24px;
    }
    .hero:after { content:'✦'; position:absolute; right:5%; top:-52px; font-size:220px; color:rgba(255,255,255,.035); }
    .eyebrow { color:#a9b5d7; font-size:.75rem; font-weight:700; letter-spacing:.17em; text-transform:uppercase; }
    .hero-title { font:800 clamp(2.2rem,5vw,3.5rem)/1.08 'Manrope',sans-serif; color:#fff; letter-spacing:-.065em; margin:.65rem 0 .8rem; }
    .hero-title span { background:linear-gradient(90deg,#c4b5fd,#5eead4); -webkit-background-clip:text; background-clip:text; color:transparent; }
    .hero-copy { color:#c0c9e0; font-size:1rem; line-height:1.65; max-width:690px; margin:0; }
    .hero-pills { display:flex; flex-wrap:wrap; gap:9px; margin-top:22px; }
    .pill { border:1px solid rgba(218,224,255,.16); background:rgba(255,255,255,.06); border-radius:100px; padding:7px 12px; color:#d9def1; font-size:.76rem; }
    .section-kicker { color:#9ea9c7; text-transform:uppercase; font-size:.72rem; font-weight:700; letter-spacing:.15em; margin:0 0 9px; }
    .panel {
        background:linear-gradient(145deg,rgba(23,31,58,.88),rgba(15,23,45,.88));
        border:1px solid var(--line); border-radius:22px; padding:22px 23px;
        box-shadow:0 16px 44px rgba(0,0,0,.13);
    }
    .panel-title { color:#f4f6ff; font:700 1.12rem 'Manrope',sans-serif; margin:0 0 6px; }
    .panel-subtitle { color:#9da8c4; font-size:.86rem; margin:0 0 16px; line-height:1.5; }
    .result-card {
        display:flex; align-items:center; gap:18px; padding:22px; border-radius:20px;
        border:1px solid rgba(167,139,250,.28);
        background:linear-gradient(120deg,rgba(139,92,246,.16),rgba(20,184,166,.09));
    }
    .result-emoji { width:84px; height:84px; flex:0 0 84px; border-radius:24px; display:flex; align-items:center; justify-content:center; font-size:48px; background:rgba(255,255,255,.07); }
    .result-label { font:800 1.75rem 'Manrope',sans-serif; color:#fff; margin:0; text-transform:capitalize; }
    .result-note { color:#aeb8d2; margin:.25rem 0 0; font-size:.87rem; }
    .score-chip { margin-left:auto; align-self:flex-start; border:1px solid rgba(94,234,212,.28); color:#8cf3df; background:rgba(20,184,166,.1); border-radius:14px; padding:9px 12px; font-weight:700; white-space:nowrap; }
    .prob-row { display:grid; grid-template-columns:105px 1fr 54px; gap:11px; align-items:center; margin:13px 0; }
    .prob-label { color:#c7cee1; font-size:.86rem; text-transform:capitalize; }
    .prob-track { width:100%; height:9px; border-radius:99px; background:rgba(169,180,218,.12); overflow:hidden; }
    .prob-fill { height:100%; border-radius:99px; }
    .prob-value { text-align:right; color:#e9edfa; font-size:.82rem; font-variant-numeric:tabular-nums; }
    .muted { color:#9da8c4; }
    .small-note { color:#8f9ab7; font-size:.77rem; line-height:1.5; }
    .stat-box { background:rgba(255,255,255,.035); border:1px solid var(--line); border-radius:17px; padding:15px 16px; }
    .stat-label { color:#9aa6c4; font-size:.75rem; text-transform:uppercase; letter-spacing:.1em; }
    .stat-value { color:#fff; font:800 1.45rem 'Manrope',sans-serif; margin-top:5px; }
    .emotion-tile { background:rgba(255,255,255,.035); border:1px solid var(--line); border-radius:16px; padding:13px; text-align:center; }
    .emotion-tile .emoji { font-size:26px; }
    .emotion-tile .name { color:#e6eafd; font-weight:700; text-transform:capitalize; font-size:.82rem; margin-top:5px; }
    .stButton > button, .stFormSubmitButton > button, [data-testid="stDownloadButton"] button {
        border:1px solid rgba(167,139,250,.35) !important; border-radius:13px !important;
        color:#fff !important; background:linear-gradient(100deg,#7556d8,#277e88) !important;
        font-weight:700 !important; min-height:2.75rem; transition:all .18s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover { border-color:#b7a3ff !important; transform:translateY(-1px); box-shadow:0 8px 22px rgba(103,80,195,.2); }
    div[data-baseweb="textarea"] textarea, div[data-baseweb="input"] input {
        background:rgba(7,13,29,.68) !important; color:#f3f5ff !important;
        border:1px solid rgba(171,184,230,.19) !important; border-radius:14px !important;
    }
    div[data-baseweb="select"] > div { background:rgba(7,13,29,.68); border-color:rgba(171,184,230,.19); border-radius:12px; }
    [data-testid="stMetric"] { background:rgba(255,255,255,.035); border:1px solid var(--line); padding:13px 15px; border-radius:15px; }
    [data-testid="stMetricLabel"] { color:#a8b2cc !important; }
    [data-testid="stMetricValue"] { color:#f4f6ff !important; }
    hr { border-color:rgba(172,185,230,.13) !important; }
    .footer { text-align:center; padding:28px 0 5px; color:#7e89a8; font-size:.76rem; }
    @media (max-width: 700px) {
      .hero { padding:25px 22px; }
      .result-card { align-items:flex-start; flex-wrap:wrap; }
      .score-chip { margin-left:0; }
      .prob-row { grid-template-columns:82px 1fr 48px; gap:7px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


EMOTION_META = {
    "joy": {"emoji": "😂", "color": "#f6c85f", "description": "An upbeat, cheerful, or happy expression."},
    "sadness": {"emoji": "😢", "color": "#78a9ff", "description": "A low, disappointed, or sorrowful tone."},
    "love": {"emoji": "❤️", "color": "#fb7185", "description": "Warmth, affection, or care."},
    "anger": {"emoji": "😡", "color": "#fb806d", "description": "Frustration, irritation, or outrage."},
    "fear": {"emoji": "😨", "color": "#b59cff", "description": "Worry, nervousness, or uncertainty."},
    "surprise": {"emoji": "😲", "color": "#5eead4", "description": "A sudden or unexpected reaction."},
}

EXAMPLES = {
    "Joy": "I finally achieved the goal I had been working towards. I am so happy!",
    "Sadness": "I miss the old days and everything feels a little empty right now.",
    "Love": "I am so grateful to have you in my life. You mean the world to me.",
    "Anger": "I am really frustrated that nobody listened to what I said.",
    "Fear": "I am nervous about what might happen tomorrow. I cannot stop worrying.",
    "Surprise": "Wait, you planned all of this for me? I genuinely did not expect it!",
}


def emotion_meta(label: str) -> dict:
    key = label.strip().lower()
    return EMOTION_META.get(key, {"emoji": "💬", "color": "#a78bfa", "description": "The model's predicted emotion."})


def render_probability_bars(labels: list[str], probabilities: list[float]) -> None:
    ranked = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)
    chunks = []
    for label, probability in ranked:
        meta = emotion_meta(label)
        percentage = max(0.0, min(float(probability) * 100, 100.0))
        chunks.append(
            f'''<div class="prob-row">
                <div class="prob-label">{html.escape(label)}</div>
                <div class="prob-track"><div class="prob-fill" style="width:{percentage:.2f}%;background:{meta['color']};"></div></div>
                <div class="prob-value">{percentage:.1f}%</div>
            </div>'''
        )
    st.markdown("".join(chunks), unsafe_allow_html=True)


def add_to_history(text: str, label: str, confidence: float) -> None:
    history = st.session_state.setdefault("history", [])
    history.insert(0, {
        "text": text[:90] + ("…" if len(text) > 90 else ""),
        "label": label,
        "confidence": confidence,
        "time": datetime.now().strftime("%H:%M"),
    })
    del history[8:]


# ---------- Load trained notebook artifacts ----------
st.markdown(
    """
    <div class="brand-row">
      <div class="brand-mark">🎭</div>
      <div><div class="brand-name">FeelScope</div><div class="brand-caption">Emotion intelligence studio</div></div>
    </div>
    <div class="hero">
      <div class="eyebrow">A little more understanding, one sentence at a time</div>
      <div class="hero-title">Feel the words.<br><span>Reveal the emotion.</span></div>
      <p class="hero-copy">Explore the emotion behind a message with a Bidirectional GRU model trained on the Hugging Face Emotion dataset. Write a sentence, run the analysis, and see how the model distributes its prediction across six emotions.</p>
      <div class="hero-pills"><span class="pill">✦ Bidirectional GRU</span><span class="pill">6 emotion classes</span><span class="pill">50-token sequence</span><span class="pill">Live inference</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    with st.spinner("Waking up the emotion model…"):
        model, tokenizer, labels, config = load_model_bundle()
except ArtifactError as exc:
    st.error("The app is ready, but its trained model artifacts have not been connected yet.")
    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">One last setup step</div>
          <p class="panel-subtitle">Export the trained BiGRU, tokenizer, labels, and preprocessing configuration from your notebook into the <code>models/</code> folder, or connect a public/private Hugging Face model repository.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(str(exc), language="text")
    st.info("Open README.md → ‘Export the trained model’ and follow the steps. Do not retrain inside Streamlit.")
    st.stop()

with st.sidebar:
    st.markdown("<div class='section-kicker'>Model status</div>", unsafe_allow_html=True)
    st.success("BiGRU model loaded")
    st.markdown("<div class='small-note'>Inference only · no training runs on this website</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div class='section-kicker'>Model snapshot</div>", unsafe_allow_html=True)
    st.markdown("""**Architecture**  
Bidirectional GRU × 2  
**Embedding**  
300 dimensions  
**Sequence length**  
50 tokens  
**Vocabulary cap**  
10,000 words""", unsafe_allow_html=False)
    st.divider()
    st.markdown("<div class='section-kicker'>Recent analyses</div>", unsafe_allow_html=True)
    history = st.session_state.get("history", [])
    if history:
        for item in history[:5]:
            meta = emotion_meta(item["label"])
            st.markdown(
                f"{meta['emoji']} **{html.escape(item['label'].title())}** · {item['confidence']*100:.1f}%  \n"
                f"<span class='small-note'>{html.escape(item['text'])}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Your recent predictions will appear here.")
    if history and st.button("Clear history", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()
    st.divider()
    st.markdown("<div class='small-note'>Built with TensorFlow · Keras · Streamlit</div>", unsafe_allow_html=True)


# ---------- Main workspace ----------
analyzer_tab, batch_tab, model_tab = st.tabs(["✨  Emotion studio", "🧪  Batch lab", "🧠  Model notes"])

with analyzer_tab:
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown('<div class="section-kicker">01 / Write something</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">What is the message feeling?</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Enter a sentence or a short message. The model uses the first 50 tokens after preprocessing.</div>', unsafe_allow_html=True)
        sample_choice = st.selectbox("Try an example", ["Choose a sample…", *EXAMPLES.keys()], label_visibility="collapsed")
        if st.button("Load example", use_container_width=False, disabled=sample_choice == "Choose a sample…"):
            st.session_state["single_text"] = EXAMPLES[sample_choice]
        with st.form("emotion_form", clear_on_submit=False):
            text_input = st.text_area(
                "Your text",
                key="single_text",
                height=175,
                max_chars=3000,
                placeholder="Write a thought, a message, or a sentence here…",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Analyze emotion  →", use_container_width=True)
        if submitted:
            if not text_input or not text_input.strip():
                st.warning("Please enter some text before starting the analysis.")
            else:
                with st.spinner("Reading the emotional signals…"):
                    result = predict_many(model, tokenizer, labels, config, [text_input.strip()])[0]
                st.session_state["last_result"] = {"text": text_input.strip(), **result}
                add_to_history(text_input.strip(), result["label"], result["confidence"])

    with right:
        st.markdown('<div class="section-kicker">02 / Emotion palette</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Six emotions, one prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">The classifier chooses from the six labels in the training dataset.</div>', unsafe_allow_html=True)
        class_rows = st.columns(3)
        for index, label in enumerate(labels):
            meta = emotion_meta(label)
            with class_rows[index % 3]:
                st.markdown(
                    f'<div class="emotion-tile"><div class="emoji">{meta["emoji"]}</div><div class="name">{html.escape(label)}</div></div>',
                    unsafe_allow_html=True,
                )

    result = st.session_state.get("last_result")
    if result:
        st.markdown("<br>", unsafe_allow_html=True)
        meta = emotion_meta(result["label"])
        confidence_pct = result["confidence"] * 100
        st.markdown('<div class="section-kicker">03 / Your result</div>', unsafe_allow_html=True)
        st.markdown(
            f'''<div class="result-card">
                <div class="result-emoji">{meta['emoji']}</div>
                <div><div class="eyebrow">Dominant predicted emotion</div>
                <p class="result-label">{html.escape(result['label'])}</p>
                <p class="result-note">{meta['description']}</p></div>
                <div class="score-chip">{confidence_pct:.1f}% score</div>
            </div>''',
            unsafe_allow_html=True,
        )
        detail_left, detail_right = st.columns([1.05, .95], gap="large")
        with detail_left:
            st.markdown('<div class="panel-title">Probability distribution</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-subtitle">How the model scored each available emotion.</div>', unsafe_allow_html=True)
            render_probability_bars(labels, result["probabilities"])
        with detail_right:
            st.markdown('<div class="panel-title">What was analyzed</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="panel"><p style="color:#e7ebfa;line-height:1.7;margin:0">“{html.escape(result["text"])}”</p><hr><div class="small-note">This is a model prediction, not a definitive reading of a person’s feelings. Context, sarcasm, mixed emotions, and cultural differences can affect the result.</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="panel"><div class="panel-title">Your result will appear here ✨</div><div class="panel-subtitle">Submit a sentence to reveal the dominant emotion and the full six-class probability distribution.</div></div>',
            unsafe_allow_html=True,
        )

with batch_tab:
    st.markdown('<div class="section-kicker">Batch workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Analyze multiple messages together</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">Enter one message per line. Empty lines are ignored. You can export the resulting predictions as CSV.</div>', unsafe_allow_html=True)
    batch_text = st.text_area("Messages", height=220, placeholder="I got the job! I am so excited.\nI am worried about the exam tomorrow.\nThat was the sweetest thing anyone has done for me.", key="batch_text")
    if st.button("Analyze all messages  →", key="run_batch", use_container_width=True):
        batch_items = [line.strip() for line in batch_text.splitlines() if line.strip()]
        if not batch_items:
            st.warning("Add at least one non-empty message.")
        elif len(batch_items) > 100:
            st.warning("Please keep each batch to 100 messages or fewer.")
        else:
            with st.spinner(f"Analyzing {len(batch_items)} messages…"):
                batch_results = predict_many(model, tokenizer, labels, config, batch_items)
            st.session_state["batch_results"] = list(zip(batch_items, batch_results))

    batch_results = st.session_state.get("batch_results", [])
    if batch_results:
        rows = []
        for original, item in batch_results:
            meta = emotion_meta(item["label"])
            rows.append({
                "Text": original,
                "Predicted emotion": item["label"],
                "Emoji": meta["emoji"],
                "Score (%)": round(item["confidence"] * 100, 2),
            })
        st.markdown("#### Results")
        st.dataframe(rows, use_container_width=True, hide_index=True)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button(
            "Download results as CSV",
            data=output.getvalue().encode("utf-8-sig"),
            file_name="feelscope_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

with model_tab:
    st.markdown('<div class="section-kicker">Inside the classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">A closer look at the BiGRU</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">The deployed application uses the trained model and the exact fitted tokenizer exported from the notebook. It does not retrain the network at runtime.</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recorded accuracy", "91.70%")
    m2.metric("Recorded loss", "0.2247")
    m3.metric("Embedding size", "300")
    m4.metric("Sequence length", "50")
    st.markdown("### Architecture")
    st.code(
        "Text input\n"
        "  ↓ Tokenizer + post-padding (50 tokens)\n"
        "Embedding(input_dim=10000, output_dim=300)\n"
        "  ↓\n"
        "Bidirectional(GRU(128, return_sequences=True))\n"
        "  ↓ Dropout(0.5)\n"
        "Bidirectional(GRU(64))\n"
        "  ↓ Dropout(0.5)\n"
        "Dense(6, activation='softmax')",
        language="text",
    )
    st.markdown("### Training setup")
    st.markdown(
        "- Dataset: `dair-ai/emotion` from Hugging Face.\n"
        "- Vocabulary cap: 10,000 words; out-of-vocabulary token: `<unk>`.\n"
        "- Sequence handling: 50 tokens, post-padding and post-truncation.\n"
        "- Optimizer and loss: Adam and sparse categorical cross-entropy.\n"
        "- Class weights: balanced weights computed from the training labels.\n"
        "- Early stopping: monitor `val_loss`, patience 3, restore best weights."
    )
    st.warning(
        "Evaluation note: the notebook supplied the test split as validation_data during training and then evaluated on that same split. The displayed 91.70% is the notebook-recorded score, not an untouched holdout estimate."
    )

st.markdown(
    '<div class="footer">FeelScope · Emotion classification is an exploratory prediction, not a psychological or clinical assessment.</div>',
    unsafe_allow_html=True,
)
