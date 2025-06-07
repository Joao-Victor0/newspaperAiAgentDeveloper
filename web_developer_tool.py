from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

class WebDeveloperTool(BaseTool):
    name: str = "web_developer_tool"

    description: str = """
        Usará esta ferramenta para receber o texto fornecido e, desta forma, construir o site
        em HTML, CSS e Javascript em um arquivo HTML, contendo as partes em Javascript e CSS, inserindo a matéria de jornal neste site.
    """

    def __init__(self):
        super().__init__()


    def _run(self, newspaper: str) -> str:
        instructions = f"""
            Você irá construir um site de jornal em HTML, CSS e Javascript, no mesmo arquivo HTML com o CSS e Javascript nele,
            inserindo a matéria do jornal {newspaper}, respeitando a formatação pré-estabelecida de
            título, subtítulo e assunto.
            
            Você deve seguir a proposta de site semelhante a sites como CNN e G1, que são sites de jornalismo.
            Deve investir em um layout entendível para o usuário, focando na matéria que você quer apresentar.
            
            É permitido adicionar elementos visuais como cores e botões interativos com movimento.
        """

        llm = ChatGroq(model="gemma2-9b-it")
        message = [HumanMessage(content=instructions)]

        response = llm.invoke(message)
        return response.content


    async def _arun(self, newspaper: str) -> str:
        raise NotImplementedError("Execução assíncrona não suportada.")