import streamlit as st
from transformers import pipeline
from PIL import Image
from deep_translator import GoogleTranslator
import random

# ==============================
# ⚙️ Configuração inicial
# ==============================
st.set_page_config(page_title="SmartPost Studio", page_icon="💡")

st.title("🧠 SmartPost Studio (versão otimizada 🚀)")
st.write("Gere legendas, traduções, hashtags e resumos automáticos com mais precisão e velocidade ✨")

# ==============================
# 🧩 Carregamento dos modelos
# ==============================
@st.cache_resource
def load_caption_model():
    try:
        # Modelo leve e compatível com Streamlit Cloud
        model_name = "microsoft/git-large-coco"
        captioner = pipeline("image-to-text", model=model_name)
        st.sidebar.success(f"Modelo carregado: {model_name}")
    except Exception as e:
        # Fallback se o modelo principal falhar
        st.sidebar.error(f"Falha ao carregar modelo principal: {e}")
        model_name = "Salesforce/blip-image-captioning-base"
        captioner = pipeline("image-to-text", model=model_name)
        st.sidebar.warning(f"Usando fallback: {model_name}")
    return captioner

@st.cache_resource
def load_refiner_model():
    try:
        refiner = pipeline("text2text-generation", model="google/flan-t5-small")
        st.sidebar.info("Refinador de texto ativo ✅")
        return refiner
    except Exception as e:
        st.sidebar.warning(f"Não foi possível carregar refinador: {e}")
        return None

captioner = load_caption_model()
refiner = load_refiner_model()

# ==============================
# 🎨 Interface
# ==============================
uploaded_file = st.file_uploader("📤 Envie uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Reduz tamanho da imagem para acelerar
    image = image.resize((512, 512))
    st.image(image, caption="📸 Imagem enviada", use_container_width=True)

    if st.button("✨ Gerar Post"):
        with st.spinner("Gerando legenda e análise da imagem..."):
            # ====== Legenda base (em inglês) ======
            caption_en = captioner(image)[0]["generated_text"]

            # ====== Refinar texto (opcional) ======
            if refiner:
                prompt = f"Melhore a legenda em inglês para que soe natural e descritiva: {caption_en}"
                caption_en = refiner(prompt, max_new_tokens=50)[0]["generated_text"]

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
        st.subheader("📝 Resultados")
        st.markdown(f"**🇺🇸 Legenda (Inglês refinada):** {caption_en}")
        st.markdown(f"**🇧🇷 Tradução (Português):** {caption_pt}")
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
            st.session_state.clear()
            st.experimental_rerun()
else:
    st.info("Envie uma imagem para começar 💡")


