from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from crewai_tools import PDFSearchTool
#from tools.custom_tools import DocumentSearchTool
from agentic_rag.tools.custom_tool import DocumentSearchTool

#Initialize the tool with a specific PDF path for exclusive search within that document
pdf_tool = DocumentSearchTool(pdf='/Users/chennasuryavamshi/Downloads/Data Visualisation Report.pdf')
web_search_tool = SerperDevTool()

@CrewBase
class AgenticRag():
    '''Agentic RAG Crew'''
    
    agents_config = 'config/agents.yaml'
    tasks_conig = 'configs/tasks.yaml'
    
    @agent
    def retreiver_agent(self) -> Agent:
        return Agent(
            config = self.agents_config['retreiver_agent'],
            verbose = True,
            tools = [pdf_tool, web_search_tool]
        )
        
    @agent
    def response_synthesizer_agent(self) -> Agent:
        return Agent(
            config = self.agents_config['response_synthesizer_agent'],
            verbose = True
        )
    
    @task
    def retreival_task(self) -> Task:
        return Task(
            config=self.tasks_config['retreival_task']
        )
    
    @task
    def response_task(self) -> Task:
        return Task(
            config = self.tasks_config['response_task']
        )
    
    @crew
    def crew(self) -> Crew:
        '''Creates the Agentic RAG Crew'''
        
        return Crew(
            agents = self.agents, #Automatically created by the @agent decorator
            tasks = self.tasks,  #Automatically created by the @task decorator
            process = Process.sequential,
            Verbose = True,
            # process = Process.heirarchial in case we wanna use verbose
        )
    
        
    