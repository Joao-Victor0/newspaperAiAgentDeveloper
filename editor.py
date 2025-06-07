from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from editor_tool import EditorTool

_ = load_dotenv(find_dotenv())

class Editor():

    def __init__(self) -> None:
        self.llm = ChatGroq(
            model="gemma2-9b-it",
            temperature="0.3"
        )

        #O que é o modelo
        system_prompt = """
            Você é um editor experiente de um jornal digital.  
            Seu papel é transformar análises de dados em matérias jornalísticas para publicação online.  
            Com base nos insights fornecidos, você deve estruturar a matéria de forma profissional, seguindo os padrões do jornalismo.  
            A matéria deve conter:  
            - **Título** impactante e informativo.  
            - **Subtítulo** que complemente o título e forneça um pouco mais contexto.  
            - **Texto principal** claro, objetivo e informativo e resumido.  
            
            O tom deve ser formal e imparcial, garantindo que a informação seja acessível e compreensível e resumida para o público em geral.  
            O texto deve ser sucinto, abordando os pontos essenciais da análise sem se estender demasiadamente.
        """

        self.agent = initialize_agent(
            llm=self.llm, #modelo utilizado
            tools=[EditorTool()], #ferramenta do modelo
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, #especializado a fazer uma tarefa fora do CHAT
            verbose=True, #possibilidade de depurar caso algo dê errado
            agent_kwargs={
                "system_prompt": system_prompt
            }
        )


    #execução do agente
    def run(self, analyzed_data: str) -> str:
        try:
            response = self.agent.invoke({"input": f"Monte a matéria de jornal usando o texto {analyzed_data}"}, handle_parsing_errors=True)
            print(f"Agent response: {response}")
            return response

        except Exception as err:
            print(f"Error: {err}")
            return "Desculpa, não consegui processar a sua solicitação."