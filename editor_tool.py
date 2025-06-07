from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

class EditorTool(BaseTool):
    name: str = "editor_tool"
    description: str = """
        Esta ferramenta constrói matérias jornalísticas com base na análise de dados feita por um Agente de IA.  
        - A entrada é um texto contendo a análise dos dados.  
        - A saída deve conter:  
          - **Título**  
          - **Subtítulo**  
          - **Texto principal**  
        
        O conteúdo deve manter a imparcialidade e seguir boas práticas jornalísticas.
    """

    def __init__(self):
        super().__init__()


    def _run(self, analyzed_data: str) -> str:
        # descrevendo o que o modelo deve fazer
        instructions = f"""
            Com base no seguinte texto analisado:  
            {analyzed_data}  
            
            Crie uma matéria jornalística para seguindo estas diretrizes:  
            - **Título**: Resuma o tema da matéria de forma objetiva e chamativa.  
            - **Subtítulo**: Explique brevemente o contexto do título, preparando o leitor para o texto.  
            - **Texto principal**: Desenvolva o conteúdo de forma clara, concisa e imparcial, destacando os pontos mais relevantes da análise, de forma resumida.  
            
            🔹 **Observações**:  
            - Use um tom formal e profissional.  
            - O texto deve ser direto e informativo, sem excessos.  
            - Evite opiniões ou especulações; baseie-se apenas nos dados fornecidos.  
            - Estruture o conteúdo de forma fluida para facilitar a leitura.
            
            Traduza o texto analisado para português brasileiro no final.
        """

        # incluindo o modelo
        llm = ChatGroq(model="gemma2-9b-it")
        message = [HumanMessage(content=instructions)]

        response = llm.invoke(message)
        return response.content


    async def _arun(self, text: str) -> str:
        raise NotImplementedError("Execução assíncrona não suportada.")

