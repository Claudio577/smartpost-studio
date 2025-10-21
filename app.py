import streamlit as st
from transformers import pipeline
from PIL import Image

# ==============================
# 🧠 Carregar modelos
# ==============================
st.set_page_config(page_title="SmartPost Studio", page_icon="💡")

@st.cache_resource
def load_models():
    captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    translator = pipeline("translation", model="facebook/m2m100_418M")
    creative_writer = pipeline("text2text-generation", model="google/flan-t5-small")
    return captioner, translator, creative_writer

captioner, translator, creative_writer = load_models()

# ==============================
# 🎨 Interface
# ==============================
st.title("🧠 SmartPost Studio")
st.write("Gere legendas criativas para redes sociais com IA 🚀")

uploaded_file = st.file_uploader("📤 Envie uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Imagem enviada", use_column_width=True)

    if st.button("✨ Gerar Legenda Criativa"):
        with st.spinner("Gerando legenda..."):
            # 1️⃣ Geração da legenda base (em inglês)
            caption_en = captioner(image)[0]['generated_text']

            # 2️⃣ Tradução para português
            caption_pt = translator(caption_en, src_lang="en", tgt_lang="pt")[0]['translation_text']

            # 3️⃣ Geração de legenda criativa
            prompt = f"Crie uma legenda criativa e emocional para Instagram sobre: {caption_pt}. Use emojis e tom amigável."
            creative_caption = creative_writer(prompt, max_length=60)[0]['generated_text']

            # 4️⃣ Geração de hashtags
            hashtags_prompt = f"Gere 5 hashtags populares em português relacionadas a: {caption_pt}"
            hashtags = creative_writer(hashtags_prompt, max_length=40)[0]['generated_text']

        st.subheader("📝 Resultado:")
        st.write(f"**Legenda Criativa:** {creative_caption}")
        st.write(f"**Hashtags:** {hashtags}")
        st.write(f"**Tradução literal:** {caption_pt}")

        # 5️⃣ Opção para baixar o texto
        output = f"{creative_caption}\n\n{hashtags}\n\n(Descrição literal: {caption_pt})"
        st.download_button("💾 Baixar legenda", output, file_name="legenda_smartpost.txt")

