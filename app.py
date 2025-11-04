import streamlit as st
from transformers import pipeline
from PIL import Image
from deep_translator import GoogleTranslator
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# ==============================
# ⚙️ Configuração inicial
# ==============================
st.set_page_config(page_title="AI Universal Studio", page_icon="🧠", layout="wide")
st.title("🧠 AI Universal Studio")
st.write("Demonstração de um sistema de IA que aprende a partir de **imagens**, **áudios** e **textos** para gerar **previsões inteligentes** ⚡")

# ==============================
# 🧩 Modelos de IA
# ==============================

# O modelo BLIP (Image Captioning) é carregado uma única vez para eficiência.
@st.cache_resource
def load_caption_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Novo: O modelo Whisper (Automatic Speech Recognition) para transcrição de áudio.
@st.cache_resource
def load_asr_model():
    # Usando o modelo Whisper "tiny" por ser mais leve e rápido para demonstração.
    # Nota: Este modelo requer que dependências de áudio como torchaudio e librosa estejam instaladas.
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

captioner = load_caption_model()
asr_transcriber = load_asr_model()

# ==============================
# 🔁 Sessão compartilhada
# ==============================
for var, default in {
    "keywords": [],
    "categories": [],
    "modelo": None, # O modelo Random Forest treinado
    "vectorizer": None # O CountVectorizer para transformar texto em features
}.items():
    if var not in st.session_state:
        st.session_state[var] = default

# ==============================
# 🧭 Abas
# ==============================
aba = st.tabs([
    "🧩 Etapa 1 - Base de Treinamento",
    "⚙️ Etapa 2 - Treinar Modelo",
    "🔮 Etapa 3 - Fazer Previsão"
])

# ======================================================
# 1️⃣ ETAPA 1 – BASE DE TREINAMENTO
# ======================================================
with aba[0]:
    st.header("🧩 Etapa 1 – Criar base de aprendizado (Palavras + Categorias)")
    st.write("Adicione até **3 exemplos de texto** para ensinar a IA o que significa cada categoria (Baixo, Moderado, Alto risco).")

    entradas = []
    # Cria três pares de input/selectbox para os exemplos de treinamento
    for i in range(3):
        col1, col2 = st.columns([3, 1])
        palavras = col1.text_input(f"📝 Exemplo {i+1} (texto ou frase):", key=f"texto_{i}")
        categoria = col2.selectbox(
            f"🎯 Categoria {i+1}:",
            ["Baixo", "Moderado", "Alto"],
            index=1,
            key=f"cat_{i}"
        )
        if palavras:
            entradas.append({"texto": palavras, "categoria": categoria})

    # Botão para salvar os dados na session_state
    if entradas and st.button("💾 Salvar base de aprendizado"):
        st.session_state.keywords = [e["texto"] for e in entradas]
        st.session_state.categories = [e["categoria"] for e in entradas]
        st.success("✅ Base de aprendizado salva com sucesso!")
        st.dataframe(pd.DataFrame(entradas), use_container_width=True)

# ======================================================
# 2️⃣ ETAPA 2 – TREINAR MODELO
# ======================================================
with aba[1]:
    st.header("⚙️ Etapa 2 – Treinar modelo com base na base de aprendizado")

    if not st.session_state.keywords or not st.session_state.categories:
        st.warning("⚠️ Nenhum dado de aprendizado. Vá para a Etapa 1 primeiro.")
    else:
        # Botão para iniciar o treinamento do modelo de Machine Learning
        if st.button("🚀 Treinar modelo agora"):
            # Inicializa e treina o CountVectorizer (Bag-of-Words)
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(st.session_state.keywords)
            y = st.session_state.categories
            
            # Inicializa e treina o modelo Random Forest
            modelo = RandomForestClassifier(random_state=42) # Adicionando random_state para reprodutibilidade
            modelo.fit(X, y)
            
            # Salva o vetorizador e o modelo na session_state
            st.session_state.vectorizer = vectorizer
            st.session_state.modelo = modelo
            st.success("✅ Modelo treinado com sucesso! Vá para a Etapa 3 para prever.")

        if st.session_state.modelo:
            st.info("✅ Modelo já treinado! Você pode ir para a Etapa 3.")

# ======================================================
# 3️⃣ ETAPA 3 – PREVISÃO (Imagem + Texto + Áudio)
# ======================================================
with aba[2]:
    st.header("🔮 Etapa 3 – Fazer previsão com novos dados (imagem + áudio + texto)")
    st.write("Envie uma **imagem**, **áudio** e/ou **texto descritivo**, e depois clique em **Fazer previsão** para combinar as informações.")

    # Colunas para organizar os uploads de imagem e áudio
    col_img, col_audio = st.columns(2)
    
    with col_img:
        uploaded_img = st.file_uploader("📷 Envie uma imagem (opcional):", type=["jpg", "jpeg", "png"], key="predict_img")
    
    with col_audio:
        uploaded_audio = st.file_uploader("🎤 Envie um arquivo de áudio (opcional):", type=["mp3", "wav", "flac"], key="predict_audio")
        
    texto_input = st.text_area("💬 Texto descritivo (opcional):", key="predict_text")

    desc_img = ""
    # Se uma imagem foi carregada, gere a descrição automaticamente
    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="📸 Imagem enviada", use_container_width=True)
        
        # O captioning é uma operação demorada, usa st.spinner
        with st.spinner("🔍 Gerando descrição automática da imagem..."):
            # 1. Gera o caption em inglês
            caption_en = captioner(image)[0]["generated_text"]
            # 2. Traduz para português para unificar a linguagem de entrada com a base de treino
            desc_img = GoogleTranslator(source="en", target="pt").translate(caption_en)
            st.markdown(f"<small>Descrição da Imagem: *{desc_img}*</small>", unsafe_allow_html=True)
    
    desc_audio = ""
    # Se um áudio foi carregado, gere a transcrição automaticamente
    if uploaded_audio:
        st.audio(uploaded_audio, format=uploaded_audio.type)
        with st.spinner("🎧 Transcrevendo áudio automaticamente..."):
            try:
                # O pipeline ASR aceita o objeto de arquivo carregado
                transcription_result = asr_transcriber(uploaded_audio)
                transcription_text = transcription_result["text"].strip()
                
                # Traduz para português (fonte 'auto' para o ASR)
                desc_audio = GoogleTranslator(source="auto", target="pt").translate(transcription_text)
                st.markdown(f"<small>Transcrição do Áudio: *{desc_audio}*</small>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Erro ao processar áudio. Verifique se o arquivo está no formato correto. Detalhe: {e}")
                desc_audio = "" # Limpa a descrição em caso de erro

    # Combina a descrição da imagem, a transcrição do áudio e o texto de entrada do usuário
    entrada = f"{desc_img} {desc_audio} {texto_input}".strip()
    st.text_area("🧩 Entrada combinada (Dados de Imagem + Áudio + Texto):", value=entrada, height=100)

    # --- Botão para previsão ---
    if st.button("🔍 Fazer previsão"):
        if not st.session_state.modelo or not st.session_state.vectorizer:
            st.error("⚠️ Treine o modelo na Etapa 2 antes de fazer previsões.")
        elif not entrada:
            st.error("⚠️ Insira uma imagem, áudio e/ou texto para prever.")
        else:
            # Transforma a nova entrada usando o vetorizador treinado
            X_novo = st.session_state.vectorizer.transform([entrada])
            # Faz a previsão
            pred = st.session_state.modelo.predict(X_novo)[0]
            
            # Define a cor de exibição com base na previsão
            cor = {"Baixo": "green", "Moderado": "orange", "Alto": "red"}[pred]

            # Exibe o resultado com estilo
            st.markdown("---")
            st.markdown(
                f"<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;'>"
                f"<h3>🧠 Previsão da IA: <span style='color:{cor}'>**{pred}**</span></h3>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown("---")


            # Exibe os exemplos de treinamento que caíram na mesma categoria
            exemplos_relacionados = [
                kw for kw, cat in zip(st.session_state.keywords, st.session_state.categories)
                if cat == pred
            ]
            if exemplos_relacionados:
                st.markdown("📚 **Exemplos que levaram a este resultado no treinamento:**")
                st.info(", ".join(exemplos_relacionados))



