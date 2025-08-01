import streamlit as st
import json
import asyncio

# Import the main converter class from your engine file
try:
    from engine import Unstructured2Structured
except ImportError:
    st.error("Fatal Error: Could not find the `engine.py` file. Please make sure it's in the same directory as this app.")
    st.stop()

st.set_page_config(
    page_title="Unstructured to Structured",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


with st.sidebar:
    st.header("Configuration")
    st.write("This app uses Groq as its LLM Provider. You'll need a free Groq API key.")
    
    # Input for the Groq API Key
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get your free API key from https://console.groq.com/keys")
    model = st.selectbox("Model used for generation.",("moonshotai/kimi-k2-instruct","llama-3.3-70b-versatile","deepseek-r1-distill-llama-70b","meta-llama/llama-4-maverick-17b-128e-instruct"))
    st.markdown("---")
    st.header("About")
    st.info(
        "This is the \"Two\" approach as described in the Metaforms assessment submission. "
        "It demonstrates a system that converts unstructured text to a structured JSON format, "
        "choosing between 2 strategies based on schema complexity."
    )


# --- Main Application UI ---
st.title("🤖 ")
st.markdown("Provide your unstructured text and a target JSON schema, then click 'Convert' to get the structured response")

# Create two columns for text and schema input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Unstructured Text")

    text_input = st.text_area("Paste your text document or any raw text here.", height=300, placeholder="")

with col2:
    st.subheader("Target JSON Schema")
    schema_input = st.text_area("Paste your target JSON schema here.", height=300, placeholder="")

# The conversion button
convert_button = st.button("Convert to JSON", type="primary", use_container_width=True)


# --- Backend Logic & Result Display ---
if convert_button:
    # --- Input Validation ---
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    elif not text_input.strip():
        st.warning("Please provide some unstructured text to convert.")
    elif not schema_input.strip():
        st.warning("Please provide a JSON schema.")
    else:
        try:
            # Validate that the schema input is valid JSON
            schema_data = json.loads(schema_input)

            # --- Run the Conversion ---
            with st.spinner("Analyzing schema, running RAG, and calling the LLM... Please wait."):
                try:
                    # Instantiate our powerful engine
                    converter = Unstructured2Structured(groq_api_key=groq_api_key, model=model)
                    
                    # Run the asynchronous convert method
                    result = asyncio.run(converter.convert(text=text_input, schema=schema_data))
                    
                    st.success("Conversion complete!")
                    
                    # --- Display Results ---
                    st.subheader("Final Extracted JSON")
                    st.json(result.data)

                    st.subheader("Processing Insights")
                    stats_col, flags_col = st.columns(2)

                    with stats_col:
                        st.metric(label="Strategy Used", value=str(result.strategy_used.value).replace('_', ' ').title())
                        st.metric(label="Processing Time", value=f"{result.processing_stats['processing_time_seconds']} s")
                        st.metric(label="Schema Complexity Score", value=f"{result.processing_stats['complexity_score']:.2f}")

                except Exception as e:
                    st.error(f"An error occurred during the conversion process: {e}")

        except json.JSONDecodeError:
            st.error("The provided JSON schema is not valid. Please check its format.")