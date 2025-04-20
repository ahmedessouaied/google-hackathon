# app.py
import streamlit as st
from audiorecorder import audiorecorder
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import uuid
import time
from typing import List, Tuple
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain_nomic import NomicEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

# Import voice and image functions
from patient_voice import record_audio, transcribe_with_groq
from brain_doctor import encode_image, analyze_image_with_query
from doctor_voice import text_to_speech_with_elevenlabs

# Load environment variables
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    st.error("ElevenLabs API key not found in environment variables")
    st.stop()
# Initialize connections with caching
@st.cache_resource
def init_neo4j():
    return Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )

@st.cache_resource
def init_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
        max_retries=2
    )

@st.cache_resource
def init_embeddings():
    return NomicEmbeddings(model="nomic-embed-text-v1.5")

@st.cache_resource
def init_vector_index(_embeddings):
    return Neo4jVector.from_existing_graph(
        embedding=_embeddings,
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

# Initialize components
try:
    graph = init_neo4j()
    llm = init_llm()
    embeddings = init_embeddings()
    vector_index = init_vector_index(embeddings)
except Exception as e:
    st.error(f"Initialization error: {str(e)}")
    st.stop()

# Entity extraction chain
class Entities(BaseModel):
    terms: List[str] = Field(..., description="Relevant medical terms from the text")

entity_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract medical terms related to Major Depressive Disorder from text."),
    ("human", "Extract terms from: {question}")
])

entity_chain = entity_prompt | llm.with_structured_output(Entities)

# Full text query generator
def generate_full_text_query(input: str) -> str:
    cleaned = ' '.join([word for word in input.split() if word])
    return ' AND '.join([f"{word}~2" for word in cleaned.split()])

# Structured retriever
def structured_retriever(question: str) -> str:
    try:
        result = []
        entities = entity_chain.invoke({"question": question})
        
        for entity in entities.terms:
            time.sleep(1)  # Rate limiting
            response = graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node, score
                CALL {
                    MATCH (node)-[r]->(neighbor)
                    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    UNION
                    MATCH (node)<-[r]-(neighbor)
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN output LIMIT 50""",
                {"query": generate_full_text_query(entity)}
            )
            result.extend([el.get('output', '') for el in response if el.get('output')])
        
        return "\n".join(filter(None, result))
    except Exception as e:
        st.error(f"Retrieval error: {str(e)}")
        return ""

# NEW RETRIEVER FUNCTION
def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ".join(unstructured_data)}
    """
    return final_data

# Chat history formatting
def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

# Condense question prompt
condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

# Search query chain
_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x: x["question"]),
)

# Final answer template
template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Complete chain
chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

# Create temp directory
os.makedirs("temp", exist_ok=True)

# Configure tabs
tab1, tab2 = st.tabs(["Mental Health Assistant", "Dermatology Analysis"])

with tab1:
    # Original mental health chat interface
    st.title("Mental Health Knowledge Assistant �")
    st.markdown("Ask about Major Depressive Disorder and related concepts")

    if "messages_tab1" not in st.session_state:
        st.session_state.messages_tab1 = []

    for message in st.session_state.messages_tab1:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask your question...", key="tab1_input"):
        st.session_state.messages_tab1.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing knowledge graph..."):
                response = chain.invoke({
                    "question": question,
                    "chat_history": [
                        (m["content"], st.session_state.messages_tab1[i+1]["content"])
                        for i, m in enumerate(st.session_state.messages_tab1[:-1])
                        if i % 2 == 0
                    ]
                })
            st.markdown(response)
        
        st.session_state.messages_tab1.append({"role": "assistant", "content": response})


with tab2:
    st.title("Skin Analysis Assistant 🧴")
    st.markdown("Analyze skin conditions using images and voice queries")
    
    if "messages_tab2" not in st.session_state:
        st.session_state.messages_tab2 = []
    
    # Image upload and voice recording
    col1, col2 = st.columns(2)
    with col1:
        uploaded_image = st.file_uploader("Upload skin image", type=["jpg", "jpeg", "png"], key="tab2_image")
    
    with col2:
        st.write("Record your question:")
        audio = audiorecorder("🎤 Start recording", "⏹ Stop recording", key="recorder")
        
        if len(audio) > 0:
            # Create temp directory if it doesn't exist
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Generate unique filename
            audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}_recording.wav")
            
            try:
                # Export audio with explicit file closing
                with open(audio_path, 'wb') as f:
                    audio.export(f, format="wav")
                
                # Verify file was created
                if not os.path.exists(audio_path):
                    raise Exception("Failed to save audio recording")

                # Transcribe audio
                with st.spinner("Transcribing..."):
                    try:
                        # Explicitly open and close the file for transcription
                        with open(audio_path, 'rb') as audio_file:
                            transcribed_text = transcribe_with_groq(
                                stt_model="whisper-large-v3",
                                audio_filepath=audio_path,
                                GROQ_API_KEY=os.getenv("GROQ_API_KEY")
                            )
                        st.session_state.messages_tab2.append({"role": "user", "content": transcribed_text})
                    except Exception as e:
                        st.error(f"Transcription error: {str(e)}")
                        
            except Exception as e:
                st.error(f"Audio processing error: {str(e)}")
            finally:
                # Add delay and retry mechanism for file deletion
                import time
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                            break
                    except PermissionError:
                        if attempt == max_attempts - 1:
                            st.warning("Could not clean up temporary audio file")
                        time.sleep(0.5)  # Wait before retrying

    # Display messages
    for message in st.session_state.messages_tab2:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Process analysis and add TTS response
    if st.session_state.messages_tab2 and st.session_state.messages_tab2[-1]["role"] == "user":
        last_message = st.session_state.messages_tab2[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing skin condition..."):
                try:
                    if uploaded_image:
                        # Save and process image
                        img_path = f"temp/{uuid.uuid4()}.jpg"
                        with open(img_path, "wb") as f:
                            f.write(uploaded_image.getbuffer())
                        
                        encoded_img = encode_image(img_path)
                        response = analyze_image_with_query(
                            query=last_message,
                            model="meta-llama/llama-4-scout-17b-16e-instruct",
                            encoded_image=encoded_img
                        )
                        os.remove(img_path)
                        
                        st.markdown(response)
                        
                        # Generate and play TTS response
                        try:
                            tts_path = f"temp/{uuid.uuid4()}_response.mp3"
                            text_to_speech_with_elevenlabs(response, tts_path)
                            
                            # Display audio player
                            st.audio(tts_path)
                            
                            # Clean up audio file after playing
                            if os.path.exists(tts_path):
                                os.remove(tts_path)
                        except Exception as e:
                            st.warning(f"Couldn't generate voice response: {str(e)}")
                        
                        st.session_state.messages_tab2.append({"role": "assistant", "content": response})
                    else:
                        st.warning("Please upload an image to analyze")
                        
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")