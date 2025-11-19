from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import initialize_agent, Tool, AgentType
from src.config import Config

class AgentManager:
    """
    Initializes the LLM and orchestrates the tools (Data Analyst vs. Retrieval).
    """

    def __init__(self, df, retrieval_engine, api_key):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            temperature=0,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
        self.df = df
        self.retrieval_engine = retrieval_engine
        self.agent_executor = self._create_master_agent()

    def _create_pandas_agent(self):
        """Creates the agent responsible for aggregations and math."""
        prefix = """
        You are a high-accuracy Data Analyst for the Haryana Gau Seva Aayog.
You are working with a DataFrame named `df`.

COLUMN DEFINITIONS:
- 'Global_Sr': The unique ID across the whole list.
- 'District': The district name (e.g., Ambala, Bhiwani).
- 'Gaushala_Name': The name of the cow shelter.
- 'Cattle_Count': The number of cows (Integer). 0 often means closed.
- 'Status': 'Active' or 'Closed'.
- 'Phone_Number': The mobile number. If 'Not Available', there is no phone.
- 'Contact_Person': The name of the manager/president.

### QUERY STRATEGIES (USE THESE EXACT PATTERNS):
1. **SEARCHING FOR IDS (e.g., "Find GSA-522 (or) gsa522"):**
   - EXACT CODE: `df[df['Registration_No'].astype(str).str.contains('GSA-522', case=False, na=False)]`

RULES:
1. When asked for "Total Cattle", sum the 'Cattle_Count' column.
2. When asked for specific contact details, check 'Contact_Person' and 'Phone_Number'.
3. If asked about "Closed" shelters, filter by `Status == 'Closed'`.
4. "YNR" or "Yamunanagar" refer to the same district.
5. ALWAYS verify your code before answering.
        """
        return create_pandas_dataframe_agent(
            self.llm,
            self.df,
            prefix=prefix,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True,
            verbose=True
        )

    def _create_master_agent(self):
        """Creates the router agent that selects between tools."""
        pandas_agent = self._create_pandas_agent()

        def run_pandas(q):
            try:
                return pandas_agent.run(q)
            except Exception as e:
                return f"Calculation Error: {e}"

        def run_retrieval(q):
            return self.retrieval_engine.search(q)

        tools = [
            Tool(
                name="Hybrid RAG Search",
                func=run_retrieval,
                description=""""USE THIS FOR LOOKUPS AND SEARCHING.
            Use this to find: specific people (e.g., 'Ashok Kumar'), phone numbers, 
            specific gaushala details, IDs (e.g., 'GSA-xxx'), locations (e.g., 'near power house'), 
            or checking if a specific entity exists.""""
            ),
            Tool(
                name="Data Analyst",
                func=run_pandas,
                description="""USE THIS ONLY FOR MATH, COUNTING, AND AGGREGATION.
            Examples: 'Total cattle count', 'Average cattle per district', 'How many shelters are closed'.
            Do NOT use this for finding phone numbers or specific people."""
            )
        ]

        return initialize_agent(
            tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=4,
            verbose=True
        )

    def query(self, user_input: str) -> str:
        return self.agent_executor.run(user_input)