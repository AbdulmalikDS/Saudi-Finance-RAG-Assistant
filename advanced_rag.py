import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
import re
from chromadb.config import Settings
import chromadb

# Database initialization
DB_PATH = os.path.join(os.path.expanduser("~"), "OneDrive", "سطح المكتب", "Projects", "Datathon", "advanced rag", "chroma_db")

Groq_api_key = "GROQ_API_KEY"
openai_api_key = "OPENAI_API_KEY"
# Initialize embeddings and connect to database
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings,
    collection_name="langchain"
)

# Initialize Groq
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    temperature=0.4,
)

# Define prompt template
template = """أنت محلل مالي خبير متخصص في تحليل التقارير المالية للشركات السعودية، مع خبرة خاصة في شركة Elm (علم).
# ... (rest of your template) ...
"""

prompt = ChatPromptTemplate.from_template(template)

# Set up retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Create RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def extract_metrics(text):
    """Extract financial metrics from text"""
    metrics = {}
    
    # Extract SAR values
    sar_pattern = r'(?:SAR|SR)\s*([\d,]+(?:\.\d+)?)|(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:SAR|SR)'
    sar_matches = re.finditer(sar_pattern, text)
    for match in sar_matches:
        value = match.group(1) or match.group(2)
        value = float(value.replace(',', ''))
        metrics[f"{value:,.0f} SAR"] = value

    # Extract percentages
    pct_matches = re.finditer(r'(-?\d+(?:\.\d+)?)\s*%', text)
    for match in pct_matches:
        value = float(match.group(1))
        metrics[f"{value}%"] = value

    return metrics

def create_visualization(metrics, chart_type="bar"):
    """Create visualization based on extracted metrics"""
    if not metrics:
        return None
    
    # Custom color palette
    colors = ['#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', 
             '#222A2A', '#B68100', '#750D86', '#EB663B', '#511CFB']
    
    # Create figure based on chart type
    if chart_type == "bar":
        fig = go.Figure(data=[
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                text=[f"{v:,.2f}" for v in metrics.values()],
                textposition='auto',
                marker=dict(
                    color=colors[:len(metrics)],
                    opacity=0.8,
                    line=dict(color='#000000', width=1.5)
                )
            )
        ])
    # ... (rest of the visualization code) ...
    
    return fig

def process_query(question, chart_type="bar"):
    """Process the query and return response and chart if applicable"""
    try:
        response = rag_chain.invoke(question)
        metrics = extract_metrics(response)
        
        if metrics:
            fig = create_visualization(metrics, chart_type)
            return response, fig
        
        return response, None
        
    except Exception as e:
        return f"Error: {str(e)}", None

def main():
    # Create Gradio interface
    demo = gr.Interface(
        fn=process_query,
        inputs=[
            gr.Textbox(label="Your Question", 
                      placeholder="Ask questions about Saudi companies, market trends, and financial metrics..."),
            gr.Radio(
                choices=["bar", "line", "pie"],
                label="Chart Type",
                value="bar"
            )
        ],
        outputs=[
            gr.Textbox(label="Analysis Result"),
            gr.Plot(label="Data Visualization")
        ],
        title="Saudi Stock Market Analysis Assistant",
        description="Ask questions about Saudi companies, market trends, and financial metrics.",
        examples=[
            ["ما هي النتائج المالية لشركة علم؟", "line"],
            ["كم بلغت إيرادات علم في آخر تقرير سنوي؟", "bar"],
            ["ما هي نسب النمو في القطاعات المختلفة؟", "pie"]
        ]
    )
    
    demo.launch(share=False)

if __name__ == "__main__":
    main() 