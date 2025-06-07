from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from web_developer_tool import WebDeveloperTool

_ = load_dotenv(find_dotenv())

class WebDeveloper:

    def __init__(self):
        self.llm = ChatGroq(
            model="gemma2-9b-it",
            temperature=1.0
        )

        system_prompt="""
            Você é um Engenheiro de Software em desenvolvimento web de Jornalismo, com
            habilidades em tecnologias HTML, CSS e Javascript.
            Você é o responsável por criar os sites, fornecendo todo o código de criação em um arquivo HTML, que contenha elementos CSS e Javascript também.
            No site, você insere informações sobre a matéria de jornal, como título, subtítulo e assunto, nas diretrizes
            de um site de jornalismo (como CNN e G1).
            As informações do site você tem de um Agente de IA que escreve essa matéria. Por isso, sua única função é inserir
            a matéria no site, de forma elegante e considerando habilidades em UX e UI.
        """

        self.agent = initialize_agent(
            llm=self.llm,
            tools=[WebDeveloperTool()],
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs={
                "system_prompt": system_prompt
            }
        )


    #execução do agente
    def run(self, newspaper:str) -> str:
        try:
            response = self.agent.invoke(
                {"input": f"Construa um site de jornal com HTML, CSS e Javascript em um arquivo HTML para inserir a matéria de jornal {newspaper}"},
                handle_parsing_errors=True
            )
            print(f"Agent Response: {response}")
            return response["output"]

        except Exception as err:
            print(f"Error: {err}")
            return "Desculpe, não consegui processar a sua solicitação."