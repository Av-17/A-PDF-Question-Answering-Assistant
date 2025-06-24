# importing all library
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI # chatmodel, make sure you have google api key in your .env file
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st
import tempfile
import os

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001") #gemini embedding
pdf = st.file_uploader("Enter a file")
if pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf.read())
        path = tmp_file.name # give the name of file to PDFloader

        mypdf = PyMuPDFLoader(path)
        loader = mypdf.load()
        text_Splitter = RecursiveCharacterTextSplitter(chunk_size = 512,chunk_overlap = 50)#divided into 512 chunks
        docs = text_Splitter.split_documents(loader)


        db = FAISS.from_documents(docs, embedding_function)# store into vectore database to retrive a content
        retriever = db.as_retriever(search_type="similarity", search_kwargs = {"k": 4})#k=4 means 4 chuncks will return with similarity

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash") #chat model
        # prompting to get answer according the structure define
        template = """Answer the question based on the following context and the Chathistory. Especially take the latest question into consideration:

        Chathistory: {history}

        Context: {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        rag_chain = prompt | llm # created  a rag chain

        class AgentState(TypedDict): #this is graph state where is define all keys which is required to gain structure output
            messages: List[BaseMessage]# all messages will store in this list 
            documents: List[Document] # all chunks will store in this list
            on_topic: str # store yes or no is question related to pdf context
            rephrased_question: str # refined user question store in this 
            proceed_to_generate: bool #store ture or false to proceed to generate answer
            rephrase_count: int # it store no., that how many time question got rephrases 
            question: HumanMessage # user message


        class GradeQuestion(BaseModel):# pydentic model to get structure output 
            score: str = Field(
                description="Question is about the specified topics? If yes -> 'Yes' if not -> 'No'"
            )

        # this function refine user question
        def question_rewriter(state: AgentState):
            # print(f"Entering question_rewriter with following state: {state}")

            # Reset state variables except for 'question' and 'messages'
            state["documents"] = []
            state["on_topic"] = ""
            state["rephrased_question"] = ""
            state["proceed_to_generate"] = False
            state["rephrase_count"] = 0

            if "messages" not in state or state["messages"] is None:
                state["messages"] = []

            if state["question"] not in state["messages"]:
                state["messages"].append(state["question"])

            if len(state["messages"]) > 1:
                conversation = state["messages"][:-1]
                current_question = state["question"].content
                messages = [
                    SystemMessage(
                        content="You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval."
                    )
                ]
                messages.extend(conversation)
                messages.append(HumanMessage(content=current_question))
                rephrase_prompt = ChatPromptTemplate.from_messages(messages)
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
                prompt = rephrase_prompt.format()
                response = llm.invoke(prompt)
                better_question = response.content.strip()
                print(f"question_rewriter: Rephrased question: {better_question}")
                state["rephrased_question"] = better_question
            else:
                state["rephrased_question"] = state["question"].content
            return state
        # this function classify wheather the user question is related to pdf context or not
        def question_classifier(state: AgentState):
            # print("Entering question_classifier")
            system_message = SystemMessage(
                content="""
            You are a helpful assistant. Your job is to classify whether the user's question is related to the provided PDF content.

            If the user's question is clearly related to the content of the PDF, respond with only 'Yes'.
            If the question is not related to the PDF at all, respond with only 'No'.
            """
            )

            doc = retriever.invoke(state["rephrased_question"])
            human_message = HumanMessage(
                    content=f"User question: {state['rephrased_question']}\n\nRetrieved document:\n{[d.page_content for d in doc]}"
                )
            grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            structured_llm = llm.with_structured_output(GradeQuestion)
            grader_llm = grade_prompt | structured_llm
            result = grader_llm.invoke({})
            state["on_topic"] = result.score.strip()
            print(f"question_classifier: on_topic = {state['on_topic']}")
            return state
        # this function will router the to retrieve node if question is related to pdf or return to off topic node
        def on_topic_router(state: AgentState):
            # print("Entering on_topic_router")
            on_topic = state.get("on_topic", "").strip().lower()
            if on_topic == "yes":
                print("Routing to retrieve")
                return "retrieve"
            else:
                print("Routing to off_topic_response")
                return "off_topic_response"

        # this funtion or node will get all the chunks from vectore database and store into the State
        def retrieve(state: AgentState):
            # print("Entering retrieve")
            documents = retriever.invoke(state["rephrased_question"])
            print(f"retrieve: Retrieved {len(documents)} documents")
            state["documents"] = documents
            return state


        class GradeDocument(BaseModel):#pydentic model to get output in yes or no
            score: str = Field(
                description="Document is relevant to the question? If yes -> 'Yes' if not -> 'No'"
            )
        # this node decide that the question is related to given chunks or not
        def retrieval_grader(state: AgentState):
            # print("Entering retrieval_grader")
            system_message = SystemMessage(
                content="""You are a grader assessing the relevance of a retrieved document to a user question.
                        Only answer with 'Yes' or 'No'.

                        If the document contains information relevant to the user's question, respond with 'Yes'.
                        Otherwise, respond with 'No'."""
            )

            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            structured_llm = llm.with_structured_output(GradeDocument)

            relevant_docs = []
            for doc in state["documents"]:
                human_message = HumanMessage(
                    content=f"User question: {state['rephrased_question']}\n\nRetrieved document:\n{doc.page_content}"
                )
                result = structured_llm.invoke([system_message, human_message])
                print(
                    f"Grading document: {doc.page_content[:30]}... Result: {result.score.strip()}"
                )
                if result.score.strip().lower() == "yes":
                    relevant_docs.append(doc)

            state["documents"] = relevant_docs
            state["proceed_to_generate"] = len(relevant_docs) > 0
            print(f"retrieval_grader: proceed_to_generate = {state['proceed_to_generate']}")
            return state

        # node which decide to generate answer or not
        def proceed_router(state: AgentState):
            # print("Entering proceed_router")
            rephrase_count = state.get("rephrase_count", 0)
            if state.get("proceed_to_generate", False):
                print("Routing to generate_answer")
                return "generate_answer"
            elif rephrase_count >= 2:
                print("Maximum rephrase attempts reached. Cannot find relevant documents.")
                return "cannot_answer"
            else:
                print("Routing to refine_question")
                return "refine_question"
            
        def refine_question(state: AgentState):# this node again refine the question to get better chunks
            # print("Entering refine_question")
            rephrase_count = state.get("rephrase_count", 0)
            if rephrase_count >= 2:
                print("Maximum rephrase attempts reached")
                return state
            question_to_refine = state["rephrased_question"]
            system_message = SystemMessage(
                content="""You are a helpful assistant that slightly refines the user's question to improve retrieval results.
                        Provide a slightly adjusted version of the question."""
            )
            human_message = HumanMessage(
                content=f"Original question: {question_to_refine}\n\nProvide a slightly refined question."
            )
            refine_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            prompt = refine_prompt.format()
            response = llm.invoke(prompt)
            refined_question = response.content.strip()
            print(f"refine_question: Refined question: {refined_question}")
            state["rephrased_question"] = refined_question
            state["rephrase_count"] = rephrase_count + 1
            return state

        def generate_answer(state: AgentState):# this node will generate answer to the refine user question
            # print("Entering generate_answer")
            if "messages" not in state or state["messages"] is None:
                raise ValueError("State must include 'messages' before generating an answer.")

            history = state["messages"]
            documents = state["documents"]
            rephrased_question = state["rephrased_question"]

            response = rag_chain.invoke(
                {"history": history, "context": documents, "question": rephrased_question}
            )

            generation = response.content.strip()

            state["messages"].append(AIMessage(content=generation))
            print(f"generate_answer: Generated response: {generation}")
            return state

        def cannot_answer(state: AgentState):# if not get chunks related to userr question then is node will execute
            # print("Entering cannot_answer")
            if "messages" not in state or state["messages"] is None:
                state["messages"] = []
            state["messages"].append(
                AIMessage(
                    content="I'm sorry, but I cannot find the information you're looking for."
                )
            )
            return state


        def off_topic_response(state: AgentState):# if the question is off topic then this node will run
            # print("Entering off_topic_response")
            if "messages" not in state or state["messages"] is None:
                state["messages"] = []
            state["messages"].append(AIMessage(content="I'm sorry! I cannot answer this question!"))
            return state



        checkpointer = MemorySaver()
        #structure of graph
        workflow = StateGraph(AgentState)
        # added all nodes
        workflow.add_node("question_rewriter", question_rewriter)
        workflow.add_node("question_classifier", question_classifier)
        workflow.add_node("off_topic_response", off_topic_response)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("retrieval_grader", retrieval_grader)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("refine_question", refine_question)
        workflow.add_node("cannot_answer", cannot_answer)

        workflow.add_edge("question_rewriter", "question_classifier")
        workflow.add_conditional_edges(
            "question_classifier",
            on_topic_router,
            {
                "retrieve": "retrieve",
                "off_topic_response": "off_topic_response",
            },
        )
        workflow.add_edge("retrieve", "retrieval_grader")
        workflow.add_conditional_edges(
            "retrieval_grader",
            proceed_router,
            {
                "generate_answer": "generate_answer",
                "refine_question": "refine_question",
                "cannot_answer": "cannot_answer",
            },
        )
        workflow.add_edge("refine_question", "retrieve")
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("cannot_answer", END)
        workflow.add_edge("off_topic_response", END)
        workflow.set_entry_point("question_rewriter")
        graph = workflow.compile(checkpointer=checkpointer)

        query = st.text_input("Ask a question about the document")
        button = st.button("Send")
        if query or button:
            with st.spinner("Generating answer..."):
                input_data = {"question": HumanMessage(content=query)}
                response = graph.invoke(input=input_data, config={"configurable": {"thread_id": 3}})
                # print(f'AI : {response["messages"][-1].content}')
                st.success("✅ Answer:\n"+response["messages"][-1].content)
                with open("Q&A.txt", "a", encoding="utf-8") as file:
                    file.write("Question: " + input_data["question"].content + "\n")
                    file.write("Answer: " + response["messages"][-1].content + "\n\n")
    
