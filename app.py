import streamlit as st
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image

st.set_page_config(page_title="SmartPost Studio", page_icon="💡")

# ==============================
# 🔹 Carregar modelos leves
# ==============================
@st.cache_resource
def load_models():
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    translator = pipeline("translation_en_to_pt", model="Helsinki-NLP/opus-mt-en-pt")
    return caption_model, caption_processor, translator

caption_model, caption_processor, translator = load_models()

# ==============================
# 🎨 Interface
# ==============================
st.title("🧠 SmartPost Studio")
st.write("Gere legendas criativas e traduzidas para redes sociais 🌎")

uploaded_file = st.file_uploader("📤 Envie uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Imagem enviada", use_column_width=True)

    if st.button("✨ Gerar Legenda"):
        with st.spinner("Gerando legenda..."):
            inputs = caption_processor(image, return_tensors="pt")
            output = caption_model.generate(**inputs)
            caption_en = caption_processor.decode(output[0], skip_special_tokens=True)
            caption_pt = translator(caption_en)[0]['translation_text']

        st.subheader("📝 Resultado:")
        st.write(f"**🇺🇸 Inglês:** {caption_en}")
        st.write(f"**🇧🇷 Português:** {caption_pt}")

        st.download_button(
            "💾 Baixar legenda",
            f"{caption_pt}\n\n(Original: {caption_en})",
            file_name="legenda_smartpost.txt"
        )

