import streamlit as st
import tempfile
import os
import uuid
from data_jornalist import DataJornalist
from editor import Editor
from web_developer import WebDeveloper

st.title("Newspaper Developer AI Agent")

uploaded_file = st.file_uploader("Anexe um arquivo CSV", type="csv")

if uploaded_file:
    if st.button("Analisar Dados"):
        jornalist_agent = DataJornalist()

        #Criando um nome único para evitar problemas
        unique_filename = f"data_{uuid.uuid4().hex}.csv"
        tmp_file_path = os.path.join(tempfile.gettempdir(), unique_filename)

        # Salvando o arquivo com um nome fixo
        with open(tmp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        print(f"✅ Arquivo salvo temporariamente em: {tmp_file_path}")

        analysis = jornalist_agent.run(tmp_file_path)
        st.write("Resultados da Análise")

        # Removendo o arquivo após a análise para não acumular lixo
        os.unlink(tmp_file_path)

        #Incluindo o editor
        editor_agent = Editor()
        newspaper = editor_agent.run(analysis)

        #Incluindo o web developer
        web_developer_agent = WebDeveloper()
        website = web_developer_agent.run(newspaper)

        #Exibindo o resultado na tela
        with st.chat_message("ai"):
            st.write(website)