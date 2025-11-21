import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate # Optional, used for structure if needed
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.config import Config

from langchain_google_genai import ChatGoogleGenerativeAI

class AgentManager:
    def __init__(self, df: pd.DataFrame, retrieval_engine, api_key: str):
        self.df = df
        self.retrieval_engine = retrieval_engine
        self.api_key = api_key
        # Initialize the graph once
        self.graph = self._build_graph()

    def _get_dataframe_context(self) -> str:
        """
        Generates strictly constrained context from the DataFrame.
        """
        districts = sorted(self.df['District'].unique().tolist())
        statuses = self.df['Status'].unique().tolist()
        
        return f"""
        DATA DICTIONARY (Strictly adhere to these values):
        1. COLUMN: 'District'
           - VALID VALUES: {districts}
           - MAPPING: "YNR" -> "Yamunanagar".
           
        2. COLUMN: 'Status'
           - VALID VALUES: {statuses}
           - RULE: 0 in 'Cattle_Count' usually implies 'Closed'.

        3. COLUMN: 'Cattle_Count'
           - Type: Integer.
           - ACTION: Sum this column for 'Total' queries.

        4. COLUMN: 'Contact_Person' & 'Phone_Number'
           - Use these for contact queries.
        """
        
    def _get_few_shot_examples(self):
        """
        Teaches the LLM how to write 100% accurate Pandas code.
        Notice how we handle case insensitivity (.str.lower()) and casting.
        """
        return """
        EXAMPLES OF CORRECT THOUGHT PROCESS:

        User: "How many cattle are in Ambala?"
        Thought: The user wants a count. This is a structured query. I will use python_analyst_tool.
        Code: result = df[df['District'].str.lower() == 'ambala']['Cattle_Count'].sum()

        User: "Give me the contact details for the shelter managed by Rajesh Kumar."
        Thought: "Managed by" implies a contact person. I will try Pandas first using string search.
        Code:
        result = df[
            df['Contact_Person'].str.lower().str.contains('rajesh', na=False)
        ][['Gaushala_Name', 'Contact_Person', 'Phone_Number']].to_markdown()
        
        User: "Who is the contact of GSA-102?"
        Thought: This looks like a specific entity lookup. I will try Pandas first.
        Code: result = df[df['Registration_No'] == 'GSA-102'][['Contact_Person', 'Phone_Number']]
        
        User: "What are the rules for registration?"
        Thought: This is a general knowledge question not in the columns. I will use search_knowledge_base.
        
        User: "District wise total cattle count."
        Code:
        result = df.groupby('District')['Cattle_Count'].sum().reset_index().to_markdown()
        
        """
        
    def _build_graph(self):
        """
        Constructs a LangGraph StateGraph (The modern "Agent").
        """
        
        # --- ADVANCED SYSTEM PROMPT ---
        system_prompt = f"""
        ROLE: Lead Data Scientist for Haryana Gau Seva Aayog.
        TASK: Retrieve 100% accurate data using Python/Pandas. 
        You are NOT a chatty assistant. You are a PRECISION ENGINE.

        ACCESS:
        1. You have a Pandas DataFrame loaded as 'df'.
        2. You have a Semantic Search engine.

        {self._get_dataframe_context()}

        STRICT ROUTING PROTOCOL (FOLLOW OR DIE):
        1. IF query asks for: Counts, Sums, Lists of names, Phone numbers, Filtering by District/Status/Village -> YOU MUST USE 'python_analyst_tool'.
        - REASON: Pandas is 100% accurate. Vector search is fuzzy.
        
        2. IF query asks for: General rules, Addresses not in columns, "Near landmark" -> USE 'search_knowledge_base'.

        CODING GUIDELINES for 'python_analyst_tool':
        1. HANDLING MISSING DATA (CRITICAL):
           - Empty Phone Numbers, Names, or Villages are stored as actual NaN (Python None).
           - TO FILTER FOR PRESENCE: Use `.notna()`.
             * Correct: df[df['Phone_Number'].notna()]
             * Wrong: df[df['Phone_Number'] != 'Not Available']
           - TO FILTER FOR ABSENCE: Use `.isna()`.
             * Correct: df[df['Contact_Person'].isna()]
        
        2. STRING MATCHING & SAFETY:
           - ALL text comparisons must be case-insensitive.
           - ALWAYS handle NaNs inside string methods to prevent runtime errors.
           - Standard Pattern: .str.lower().str.contains('value', na=False)
             * Correct: df[df['District'].str.lower() == 'ambala']
             * Correct: df[df['Contact_Person'].str.lower().str.contains('rajesh', na=False)]
        
        3. OUTPUT FORMATTING (PREVENT CRASHES):
           - FINAL RESULT: You MUST assign the final output to the variable 'result'.
           - DATAFRAMES: Convert to Markdown string immediately.
             * Code: result = df[...].to_markdown()
           - LISTS/SERIES: Never return a raw list. Join it into a string.
             * Code: result = ", ".join(df['District'].unique().tolist())
             * Reason: Raw lists cause serialization errors.
        
        4. DATA TYPE AWARENESS:
           - 'Cattle_Count': INTEGER. Safe for .sum(), .mean(), .max().
           - 'Phone_Number': STRING. Do not treat as integer.
           - 'Registration_No': STRING.
           
        5. REGISTRATION NUMBER LOOKUP (EXACT NUMERIC MATCH):
           - Scenario: User asks for "1", "001", "GSA-14", or "312".
           - The DB stores "GSA-001", "GSA-014", "GSA-4312".
           - PROBLEM: .str.contains('1') matches '4312' (WRONG).
           - SOLUTION: You MUST extract the integer value from the dataframe column and compare exactly.
           
           - CODE PATTERN (Copy This Logic):
             # Example: User asks for "312"
             target_id = 312 
             # 1. Remove non-digits (\D) from column
             # 2. Convert to numeric (coerce errors to NaN)
             # 3. Compare EQUALITY (==)
             result = df[
                 pd.to_numeric(df['Registration_No'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce') == target_id
             ][['Gaushala_Name', 'Registration_No', 'Status']].to_markdown()

        {self._get_few_shot_examples()}
        """
        
        # --- 1. TOOL DEFINITIONS ---
        
        @tool
        def python_analyst_tool(python_code: str) -> str:
            """
            EXECUTES PYTHON PANDAS CODE. 
            Use for: Counting, Summing, Filtering, Math.
            INPUT: Valid Python code string.
            
            RULES:
            - The dataframe is pre-loaded as variable 'df'.
            - Assign final result to variable 'result'.
            - Do NOT use print().
            """
            
            # 1. Setup debugging visibility
            print(f"\n🐍 GENERATED CODE:\n{python_code}\n") 
            
            # 1. Initialize locals with 'df'
            local_vars = {"df": self.df, "result": None}
            
            try:
                # 2. Sanitize formatting (remove ```python blocks)
                cleaned_code = python_code.strip()
                if "```python" in cleaned_code:
                    cleaned_code = cleaned_code.split("```python")[1].split("```")[0].strip()
                elif "```" in cleaned_code:
                    cleaned_code = cleaned_code.split("```")[1].split("```")[0].strip()
                        
                # 3. CRITICAL: Add imports inside execution scope if needed
                exec("import pandas as pd", {}, local_vars)
                
                # 4. Execute
                exec(cleaned_code, {}, local_vars)
                
                raw_result = local_vars.get("result", "No result variable set.")
                
                # ---------------------------------------------------------
                # CRITICAL FIX FOR "LIST NO ATTRIBUTE SPLIT"
                # ---------------------------------------------------------
                import pandas as pd
                
                if isinstance(raw_result, pd.DataFrame):
                    # Convert DataFrames to Markdown for beautiful LLM interpretation
                    return raw_result.to_markdown()
                elif isinstance(raw_result, pd.Series):
                    return raw_result.to_markdown()
                elif isinstance(raw_result, list):
                    # Join lists to string
                    return ", ".join(map(str, raw_result))
                else:
                    # Force string for numbers/booleans/etc
                    return str(raw_result)
                
            except Exception as e:
                error_msg = f"PYTHON EXECUTION ERROR: {str(e)}"
                print(f"❌ {error_msg}")
                return error_msg
            
        @tool
        def search_knowledge_base(query: str) -> str:
            """
            ONLY USE THIS FOR UNSTRUCTURED TEXT SEARCH.
            SEARCHES TEXTUAL DATA/VECTOR STORE.
            Use for: Names, phone numbers, landmarks.
            NOT for counting or math.
            
            DO NOT USE THIS FOR:
            - "Who is the contact for X?" (Use Pandas)
            - "Total cattle in Y" (Use Pandas)
            - "List all shelters in Z" (Use Pandas)
            """
            return self.retrieval_engine.search(query)

        tools = [python_analyst_tool, search_knowledge_base]

        # --- 2. MODEL SETUP ---
        
        llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            google_api_key=self.api_key,
            temperature=0,
        )
        
        # Bind tools to the model (The new standard way)
        llm_with_tools = llm.bind_tools(tools)

        # --- 3. DEFINE NODES ---

        # The System Message with the Data Dictionary
        sys_msg = SystemMessage(content=system_prompt)

        def reasoner_node(state: MessagesState):
            """
            The core agent node. It takes history, prepends system logic, and calls LLM.
            """
            # We construct the messages list: System Instruction + Conversation History
            messages = [sys_msg] + state["messages"]
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        # --- 4. BUILD GRAPH ---
        
        builder = StateGraph(MessagesState)

        # Add Nodes
        builder.add_node("agent", reasoner_node)
        builder.add_node("tools", ToolNode(tools)) # LangGraph's prebuilt execution node

        # Add Edges
        builder.add_edge(START, "agent")
        
        # Conditional Edge: Checks if LLM wants to call a tool
        builder.add_conditional_edges(
            "agent",
            tools_condition, # Prebuilt logic: if tool_calls in msg -> "tools", else -> END
        )
        
        # Edge: After tools run, go back to agent to interpret results
        builder.add_edge("tools", "agent")

        # Compile with simple memory (for thread persistence during the session)
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)

    def query(self, user_input: str, thread_id: str = "default_session") -> str:
        """
        Invokes the graph. 
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Stream or Invoke
            # Using invoke for a synchronous response
            final_state = self.graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            
            # Extract the final response from the AI
            return final_state["messages"][-1].content
            
        except Exception as e:
            return f"Agent Execution Failed: {str(e)}"