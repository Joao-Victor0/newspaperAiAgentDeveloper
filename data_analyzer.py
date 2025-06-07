import pandas as pd  # biblioteca de análise de dados
import os
import tempfile
from langchain.tools import BaseTool  # classe Base para ferramenta
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

class DataAnalyzerTool(BaseTool):
    name: str = "data_analyzer"  # nome da ferramenta

    # O que deve ser feito a partir da ferramenta
    description: str = """
        Esta ferramenta analisa dados a partir de um arquivo CSV.  
        - O caminho do arquivo pode conter espaços e caracteres especiais; você deve passá-lo exatamente como recebido, sem modificações.  
        - A ferramenta utiliza estatísticas para identificar tendências e padrões nos dados e retorna um resumo da análise.  
        - Seu papel é interpretar os resultados e destacá-los de forma que contribuam para a construção de uma matéria jornalística.  
    """

    def __init__(self):  # construtor da classe
        super().__init__()  # inicializando o construtor da classe base


    def _run(self, csv_file_path:str) -> str:
        if not os.path.exists(csv_file_path):
            print(f"❌ ERRO: Arquivo não encontrado: {csv_file_path}")
            return "Erro: O arquivo CSV não foi encontrado."

        print(f"🔍 Caminho do arquivo recebido: {csv_file_path}")

        #tratamento da imagem
        try:
            df = pd.read_csv(csv_file_path)  # lendo o arquivo csv
            df_sample = df.head(20) # pegamos apenas as primeiras 20 linhas para evitar limite de tokens
            csv_string = df_sample.to_string(index=False)  #convertendo para string (evita erro de formatação)

        except Exception as err:
            print(f"Erro de leitura na formatação do arquivo: {err}")
            return None

        # descrevendo o que o modelo deve fazer
        instructions = f"""
            Analise os seguintes dados do arquivo CSV:  
            {csv_string}   
            
            Diretrizes:  
            - Resuma os dados de maneira compreensível.  
            - Não crie perguntas adicionais;
            - Evite linguagem técnica excessiva e traduza os insights para um público geral.  
            - Sua função é fornecer informações que possam ser usadas na construção de uma matéria jornalística, mas não redigir a matéria em si.  
            (Em português brasileiro)

            
        """

        # incluindo o modelo
        llm = ChatGroq(model="gemma2-9b-it")  # o modelo de LLM
        message = [HumanMessage(content=instructions)]

        response = llm.invoke(message)  # envia a pergunta e recebe a descrição
        return response.content

    async def _arun(self, csv_file_path: str) -> str:  # função run assíncrona (assinatura de um método)
        raise NotImplementedError("Execução assíncrona não suportada.")