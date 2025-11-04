import streamlit as st
from transformers import pipeline
from PIL import Image
from deep_translator import GoogleTranslator

# ==============================
# ⚙️ Configuração inicial
# ==============================
st.set_page_config(page_title="SmartPost Studio", page_icon="✨", layout="wide")
st.title("✨ SmartPost Studio")
st.write("Um app de IA que gera **legendas criativas**, **traduções**, **hashtags** e **resumos** automaticamente para suas imagens. Ideal para redes sociais e criação de conteúdo!")

# ==============================
# 🔁 Cache de modelos
# ==============================
@st.cache_resource
def load_caption_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_summary_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

captioner = load_caption_model()
summarizer = load_summary_model()

# ==============================
# 📸 Upload da Imagem
# ==============================
uploaded_img = st.file_uploader("📷 Envie uma imagem para gerar a legenda automática:", type=["jpg", "jpeg", "png"])

if uploaded_img:
    image = Image.open(uploaded_img).convert("RGB")
    st.image(image, caption="Imagem carregada", use_container_width=True)

    # Geração de legenda
    with st.spinner("✨ Gerando legenda automática..."):
        caption_en = captioner(image)[0]["generated_text"]
        caption_pt = GoogleTranslator(source="en", target="pt").translate(caption_en)

    st.subheader("📝 Legenda gerada")
    st.success(caption_pt)

    # ==============================
    # 🌍 Traduções
    # ==============================
    st.subheader("🌎 Traduções automáticas")
    col1, col2 = st.columns(2)

    with col1:
        caption_es = GoogleTranslator(source="pt", target="es").translate(caption_pt)
        st.text_area("🇪🇸 Espanhol", caption_es, height=100)

    with col2:
        caption_en2 = GoogleTranslator(source="pt", target="en").translate(caption_pt)
        st.text_area("🇺🇸 Inglês", caption_en2, height=100)

    # ==============================
    # 🔖 Hashtags automáticas
    # ==============================
    st.subheader("🏷️ Hashtags sugeridas")
    palavras = caption_pt.lower().split()
    hashtags = [f"#{p.strip(',.!?')}" for p in palavras if len(p) > 3][:8]
    st.write(" ".join(hashtags))

    # ==============================
    # 🧠 Resumo criativo (opcional)
    # ==============================
    st.subheader("🧠 Resumo criativo")
    resumo = summarizer(caption_en, max_length=30, min_length=5, do_sample=False)[0]["summary_text"]
    resumo_pt = GoogleTranslator(source="en", target="pt").translate(resumo)
    st.info(resumo_pt)

else:
    st.info("📤 Envie uma imagem acima para começar!")




