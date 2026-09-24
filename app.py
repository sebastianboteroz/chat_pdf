import os
import platform
import traceback
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

# Importaciones modernas de LangChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS (UX/UI)
# ==========================================
st.set_page_config(
    page_title="BrandIntel AI | Análisis de Marcas & Marketing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding: 1.5rem 2rem; }
    .brand-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #4338CA 50%, #6D28D9 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .brand-header h1 {
        color: #FFFFFF !important;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .brand-header p {
        color: #E0E7FF;
        font-size: 1.05rem;
        margin: 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        color: white !important;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.92; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# BARRA LATERAL (CONFIGURACIÓN & MARCA)
# ==========================================
with st.sidebar:
    try:
        image = Image.open('Chat_pdf.png')
        st.image(image, width=280)
    except Exception:
        st.markdown("### 📈 BrandIntel AI")

    st.markdown("## 🎯 BrandIntel AI")
    st.caption("Asistente RAG especializado en auditorías de marca, análisis de competencia y estrategias de mercadeo.")
    
    st.markdown("---")

    st.subheader("🔑 Autenticación")
    ke = st.text_input(
        "Clave de API de OpenAI", 
        type="password",
        placeholder="sk-...",
        help="Tu API Key es requerida para procesar embeddings y consultas estratégicas."
    )
    
    if ke:
        os.environ['OPENAI_API_KEY'] = ke
        st.success("API Key vinculada correctamente")
    else:
        st.warning("Ingresa tu API Key para habilitar la plataforma.")

    st.markdown("---")
    st.caption(f"Entorno: Python v{platform.python_version()} | Engine: LangChain RAG")

# ==========================================
# CUERPO PRINCIPAL
# ==========================================
st.markdown("""
<div class="brand-header">
    <h1>Estrategia de Marca & Analítica de Mercadeo 📊</h1>
    <p>Carga reportes de mercado, briefs publicitarios o planes de medios en PDF para obtener hallazgos estratégicos en segundos.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Cargar Documento de Marca")
    pdf = st.file_uploader(
        "Selecciona un estudio de mercado, brief o reporte (PDF)",
        type="pdf"
    )

with col2:
    st.subheader("💡 Ejemplos de Consultas")
    st.markdown("""
    * *"¿Cuál es la propuesta de valor principal de la marca?"*
    * *"Identifica el buyer persona descrito en el reporte."*
    * *"¿Cuáles son los canales de comunicación clave recomendados?"*
    * *"Resume las fortalezas y debilidades frente a los competidores."*
    """)

st.markdown("---")

# ==========================================
# LÓGICA DE PROCESAMIENTO RAG
# ==========================================
if pdf is not None and ke:
    try:
        with st.spinner("Procesando e indexando el documento de marca..."):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=600,
                chunk_overlap=50,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.success(f"¡Documento indexado con éxito! ({len(chunks)} bloques procesados)")
        
        st.subheader("🔍 Consultar al Asistente de Marca")
        
        with st.form(key="marketing_query_form"):
            user_question = st.text_area(
                "Escribe tu pregunta estratégica:",
                placeholder="Ejemplo: Resume la estrategia de posicionamiento y las métricas clave (KPIs)...",
                height=100
            )
            
            submit_button = st.form_submit_button(label="🚀 Analizar y Generar Respuesta")

        if submit_button:
            if not user_question.strip():
                st.warning("Por favor escribe una consulta antes de procesar.")
            else:
                with st.spinner("Analizando la información estratégica del documento..."):
                    docs = knowledge_base.similarity_search(user_question, k=4)
                    context_text = "\n\n".join([doc.page_content for doc in docs])

                    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")
                    
                    messages = [
                        ("system", "Eres un experto analista de mercadeo y estrategia de marcas. Responde la pregunta del usuario utilizando exclusivamente la información del contexto."),
                        ("user", f"Contexto:\n{context_text}\n\nPregunta: {user_question}")
                    ]
                    
                    response = llm.invoke(messages)

                    st.markdown("---")
                    st.subheader("📋 Hallazgos & Respuesta Estratégica")
                    st.info(response.content)

    except Exception as e:
        st.error("Ocurrió un problema al procesar el archivo o ejecutar la consulta.")
        with st.expander("Ver detalle técnico del error"):
            st.code(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("⚠️ Debes ingresar tu Clave de API de OpenAI en la barra lateral para continuar.")
else:
    st.info("👋 Para comenzar, carga un archivo PDF de mercadeo desde el panel superior.")
