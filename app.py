import os
import platform
import traceback
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

# Importaciones actualizadas y compatibles de LangChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS (UX/UI)
# ==========================================
st.set_page_config(
    page_title="BrandIntel AI | Análisis de Marcas & Marketing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para una interfaz moderna
st.markdown("""
<style>
    .main {
        padding: 2rem 3rem;
    }
    
    .brand-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #4338CA 50%, #6D28D9 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .brand-header h1 {
        color: #FFFFFF !important;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0.5rem;
    }
    
    .brand-header p {
        color: #E0E7FF;
        font-size: 1.1rem;
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
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        opacity: 0.95;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# BARRA LATERAL (CONFIGURACIÓN & MARCA)
# ==========================================
with st.sidebar:
    # Carga de imagen con compatibilidad garantizada
    try:
        image = Image.open('Chat_pdf.png')
        st.image(image, use_column_width=True)
    except Exception:
        try:
            # Reintento con el nombre alternativo si existe
            image = Image.open('brand_logo.png')
            st.image(image, use_column_width=True)
        except Exception:
            # Imagen de respaldo
            st.image(
                "https://images.unsplash.com/photo-1557804506-669a67965ba0?auto=format&fit=crop&w=600&q=80",
                use_column_width=True,
                caption="Brand & Market Intelligence"
            )

    st.markdown("## 🎯 BrandIntel AI")
    st.caption("Asistente RAG especializado en auditorías de marca, análisis de competencia y estrategias de mercadeo.")
    
    st.divider()

    # Credenciales de API
    st.subheader("🔑 Autenticación")
    ke = st.text_input(
        "Clave de API de OpenAI", 
        type="password",
        placeholder="sk-...",
        help="Tu API Key es requerida para procesar embeddings y consultas estratégicas."
    )
    
    if ke:
        os.environ['OPENAI_API_KEY'] = ke
        st.success("API Key vinculada correctamente", icon="✅")
    else:
        st.warning("Ingresa tu API Key para habilitar la plataforma.", icon="⚠️")

    st.divider()
    st.caption(f"Entorno: Python v{platform.python_version()} | Engine: LangChain RAG")

# ==========================================
# CUERPO PRINCIPAL
# ==========================================

# Banner de Encabezado
st.markdown("""
<div class="brand-header">
    <h1>Estrategia de Marca & Analítica de Mercadeo 📊</h1>
    <p>Carga reportes de mercado, briefs publicitarios o planes de medios en PDF para obtener hallazgos estratégicos en segundos.</p>
</div>
""", unsafe_allow_html=True)

# Layout de dos columnas para carga de archivo y métricas rápidas
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📁 Cargar Documento de Marca")
    pdf = st.file_uploader(
        "Selecciona un estudio de mercado, brief o reporte (PDF)",
        type="pdf",
        help="Sube un archivo en formato PDF para iniciar el análisis."
    )

with col2:
    st.subheader("💡 Ejemplos de Consultas")
    st.markdown("""
    * *"¿Cuál es la propuesta de valor principal de la marca?"*
    * *"Identifica el buyer persona descrito en el reporte."*
    * *"¿Cuáles son los canales de comunicación clave recomendados?"*
    * *"Resume las fortalezas y debilidades frente a los competidores."*
    """)

st.divider()

# ==========================================
# LÓGICA DE PROCESAMIENTO RAG
# ==========================================
if pdf is not None and ke:
    try:
        with st.status("Procesando e indexando el documento de marca...", expanded=True) as status:
            st.write("📄 Extrayendo contenido del documento...")
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted

            st.write(f"✅ Texto extraído: **{len(text):,}** caracteres.")

            # Segmentación de texto (Chunking)
            st.write("✂️ Dividiendo el texto en bloques conceptuales...")
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=600,
                chunk_overlap=50,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            st.write(f"✅ Documento estructurado en **{len(chunks)}** bloques estratégicos.")

            # Creación del Vector Store
            st.write("🧠 Generando matriz vectorial e índice semántico...")
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            
            status.update(label="¡Indexación completada con éxito!", state="complete", expanded=False)

        st.success("Documento cargado e indexado. El sistema está listo para responder consultas de mercado.")
        
        # Interfaz de Consulta
        st.subheader("🔍 Consultar al Asistente de Marca")
        
        # Formulario para envío explícito mediante botón
        with st.form(key="marketing_query_form"):
            user_question = st.text_area(
                "Escribe tu pregunta estratégica:",
                placeholder="Ejemplo: Resume la estrategia de posicionamiento y las métricas clave (KPIs) mencionadas...",
                rows=3
            )
            
            submit_button = st.form_submit_button(label="🚀 Analizar y Generar Respuesta")

        # Procesamiento de la pregunta al presionar el botón
        if submit_button:
            if not user_question.strip():
                st.warning("Por favor escribe una consulta antes de procesar.", icon="ℹ️")
            else:
                with st.spinner("Analizando la información estratégica del documento..."):
                    # Búsqueda semántica
                    docs = knowledge_base.similarity_search(user_question, k=4)

                    # Modelo GPT optimizado
                    llm = ChatOpenAI(
                        temperature=0.2, 
                        model_name="gpt-4o-mini"
                    )

                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=user_question)

                    # Presentación visual de la respuesta
                    st.markdown("---")
                    st.subheader("📋 Hallazgos & Respuesta Estratégica")
                    st.info(response)

    except Exception as e:
        st.error("Ocurrió un problema al procesar el archivo o ejecutar la consulta.")
        with st.expander("Ver detalle técnico del error"):
            st.code(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("⚠️ Debes ingresar tu Clave de API de OpenAI en la barra lateral para continuar.")
else:
    st.info("👋 Para comenzar, carga un archivo PDF de mercadeo desde el panel superior.")
