import streamlit as st
from transformers import pipeline
from PIL import Image
from deep_translator import GoogleTranslator
import random

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
st.write("Gere legendas, traduções, hashtags e resumos automáticos para suas imagens ✨")

# Se o usuário quiser reiniciar o app
if "nova_imagem" not in st.session_state:
    st.session_state.nova_imagem = False

if st.session_state.nova_imagem:
    st.session_state.clear()
    st.experimental_rerun()

uploaded_file = st.file_uploader("📤 Envie uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Imagem enviada", use_container_width=True)

    if st.button("Gerar Post"):
        with st.spinner("Gerando legenda e análise de imagem..."):
            # ====== Legenda em inglês ======
            caption_en = captioner(image)[0]["generated_text"]

            # ====== Tradução para português ======
            caption_pt = GoogleTranslator(source="en", target="pt").translate(caption_en)

            # ====== Resumo curto ======
            resumo_opcoes = [
                "Um toque criativo para suas redes sociais!",
                "Perfeito para inspirar o dia ✨",
                "Um momento simples que fala muito.",
                "Transforme momentos em conexões 💫",
                "Compartilhe boas vibrações 💛"
            ]
            resumo_curto = random.choice(resumo_opcoes)

            # ====== Hashtags automáticas ======
            palavras = caption_pt.lower().split()
            principais = [p.replace(",", "") for p in palavras if len(p) > 4]
            hashtags = ["#" + p for p in principais[:5]]
            hashtags_base = hashtags + ["#inspiracao", "#fotografia", "#smartpost", "#ia"]

        # ==============================
        # 🧾 Exibição dos resultados
        # ==============================
        st.subheader("📝 Resultados:")
        st.markdown(f"**🇺🇸 Legenda (Inglês):** {caption_en}")
        st.markdown(f"**🇧🇷 Tradução:** {caption_pt}")
        st.markdown(f"**🪶 Resumo curto:** {resumo_curto}")
        st.markdown(f"**🏷️ Hashtags:** {' '.join(hashtags_base)}")

        texto_final = (
            f"{caption_pt}\n\n{resumo_curto}\n\n{' '.join(hashtags_base)}"
            f"\n\n(Original: {caption_en})"
        )

        st.download_button(
            "💾 Baixar Post Completo",
            texto_final,
            file_name="post_gerado.txt"
        )

        # Novo botão: Enviar nova imagem
        st.markdown("---")
        if st.button("🖼️ Enviar nova imagem"):
            st.session_state.nova_imagem = True
            st.experimental_rerun()


