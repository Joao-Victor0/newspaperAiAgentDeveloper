from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from data_analyzer import DataAnalyzerTool

_ = load_dotenv(find_dotenv())

class DataJornalist:

    def __init__(self) -> None:
        self.llm = ChatGroq(
            model="gemma2-9b-it", #o modelo de LLM
            temperature=0.1 #o grau de criatividade da IA
        )

        system_prompt=""" 
            Você é um assistente de jornalismo de dados especializado na análise de informações em formato CSV, com foco em saúde pública.  
            Sua função é interpretar os dados, identificar tendências relevantes e responder a perguntas específicas de forma clara e objetiva.  
            Seu objetivo final é fornecer insumos para a produção de matérias jornalísticas informativas e acessíveis ao público.  
            Evite análises excessivamente técnicas; foque em informações que sejam úteis e compreensíveis para leitores leigos.  
        """

        self.agent = initialize_agent(
            llm=self.llm,
            tools=[DataAnalyzerTool()],
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, #especializado a fazer uma tarefa fora do CHAT
            verbose=True, #possibilidade de depurar caso algo dê errado
            agent_kwargs={
                'system_prompt':system_prompt
            }
        )


    #execução do Agente
    def run(self, csv_file_path: str) -> str:
        print(f"\n\nCSV_FILE_PATH na execução do Agente em data_jornalist: {csv_file_path}")
        try:
            response = self.agent.invoke({"input": f"Analise os dados do arquivo anexado {csv_file_path}"}, handle_parsing_errors=True)
            print(f"Agent Response: {response}")
            return response["output"]

        except Exception as err:
            print(f'Error: {err}')
            return 'Desculpe, não consegui processar a sua solicitação.'