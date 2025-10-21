import streamlit as st
from transformers import pipeline
from PIL import Image
from deep_translator import GoogleTranslator

# ==============================
# ⚙️ Configuração
# ==============================
st.set_page_config(page_title="SmartPost Studio", page_icon="💡")

@st.cache_resource
def load_model():
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    return captioner

captioner = load_model()

# ==============================
# 🎨 Interface
# ==============================
st.title("🧠 SmartPost Studio")
st.write("Gere legendas automáticas e criativas em português a partir de imagens ✨")

uploaded_file = st.file_uploader("📤 Envie uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Imagem enviada", use_column_width=True)

    if st.button("Gerar Legenda"):
        with st.spinner("Gerando legenda..."):
            # Gera legenda em inglês
            caption_en = captioner(image)[0]["generated_text"]

            # Traduz com Google Translator (Deep Translator)
            caption_pt = GoogleTranslator(source="en", target="pt").translate(caption_en)

        st.subheader("📝 Resultado:")
        st.write(f"**🇺🇸 Inglês:** {caption_en}")
        st.write(f"**🇧🇷 Português:** {caption_pt}")

        texto = f"{caption_pt}\n\n(Original: {caption_en})"
        st.download_button("💾 Baixar Legenda", texto, file_name="legenda.txt")


