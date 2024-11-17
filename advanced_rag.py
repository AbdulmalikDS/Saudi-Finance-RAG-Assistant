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
DB_PATH = os.path.join(os.path.expanduser("~"), "PRIAVTE", "PRIVATE", "Projects", "Datathon", "advanced rag", "chroma_db")

Groq_api_key = "PRIVATE"
openai_api_key = "PRIVATE"
# Initialize embeddings and connect to database
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings,
    collection_name="langchain"
)

# Verify connection
try:
    doc_count = vectorstore._collection.count()
    print(f"Successfully connected to existing database")
    print(f"Number of documents in collection: {doc_count}")
except Exception as e:
    print(f"Error connecting to database: {str(e)}")

# Initialize Groq
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    temperature=0.4,
)
# Improved Arabic-aware prompt
template = """أنت محلل مالي خبير متخصص في تحليل التقارير المالية للشركات السعودية، مع خبرة خاصة في شركة Elm (علم).

قواعد اللغة:
- إذا كان السؤال باللغة العربية، يجب أن يكون الرد باللغة العربية
- إذا كان السؤال باللغة الإنجليزية، يجب أن يكون الرد باللغة الإنجليزية

السياق المتوفر من الوثائق:
{context}

السؤال: {question}

يجب هيكلة إجابتك كما يلي:

1. الملخص التنفيذي:
   - النقاط الرئيسية بشكل موجز ودقيق
   - الاستنتاجات الأساسية

2. التفاصيل المالية (إن وجدت):
   - الأرقام المالية الرئيسية
   - مؤشرات الأداء الرئيسية (KPIs)
   - معدلات النمو والتغير
   - المقارنات مع الفترات السابقة

3. التحليل المفصل:
   - شرح العوامل المؤثرة
   - تحليل الاتجاهات
   - المخاطر والفرص
   - التوقعات المستقبلية

4. معلومات إضافية (إن وجدت):
   - معلومات القطاع
   - المقارنات مع المنافسين
   - التطورات الاستراتيجية

معايير الجودة:
- استخدام الأرقام والنسب بدقة عالية
- ذكر المصدر عند الإشارة إلى معلومات محددة
- تنسيق العملة: "1,000,000 ريال سعودي"
- تنسيق النسب المئوية: "٢٥٪" بالعربية أو "25%" بالإنجليزية
- استخدام "Elm" أو "علم" حسب لغة السؤال
- تقديم سياق كافٍ للأرقام والإحصائيات
- الإشارة إلى تواريخ البيانات المالية

ملاحظات إضافية:
- تجنب التخمين - استخدم فقط المعلومات المتوفرة في السياق
- قدم تحليلاً متوازناً يشمل الإيجابيات والتحديات
- اذكر أي قيود أو افتراضات في التحليل
"""

prompt = ChatPromptTemplate.from_template(template)



retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4  # Number of documents to retrieve
    }
)


# Create the chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Test function
def test_rag_query(query):
    """Test a query with the RAG system"""
    try:
        print(f"\nالسؤال: {query}")
        print("-" * 50)
        response = rag_chain.invoke(query)
        print(response)
        print("-" * 50)
    except Exception as e:
        print(f"خطأ: {str(e)}")
"""
# Test queries
arabic_test_queries = [
    "ما هي أبرز النتائج المالية لشركة علم في آخر تقرير؟",
    "ما هي استراتيجية النمو المستقبلية للشركة؟",
    "كم بلغت إيرادات الشركة وما هو معدل النمو؟",
    "ما هي أهم القطاعات التي تعمل فيها الشركة؟",
    "ما هي أبرز المنتجات والخدمات التي تقدمها علم؟"
]

# Test each query
for query in arabic_test_queries:
    test_rag_query(query)
"""


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
        
    if chart_type == "bar":
        fig = go.Figure(data=[
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                text=[f"{v:,.2f}" for v in metrics.values()],
                textposition='auto',
                marker_color='rgb(26, 118, 255)'
            )
        ])
    elif chart_type == "line":  # Explicit line chart handling
        fig = go.Figure(data=[
            go.Scatter(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                mode='lines+markers+text',
                text=[f"{v:,.2f}" for v in metrics.values()],
                textposition='top center',
                line=dict(color='rgb(26, 118, 255)', width=2),
                marker=dict(size=8)
            )
        ])
    elif chart_type == "pie":
        fig = go.Figure(data=[
            go.Pie(
                labels=list(metrics.keys()),
                values=list(metrics.values()),
                textinfo='label+percent',
                hole=.3
            )
        ])
    
    fig.update_layout(
        title="Financial Metrics Visualization",
        template="plotly_dark",
        height=500,
        margin=dict(t=50, l=50, r=50, b=50)
    )
    
    return fig

def process_query(question, chart_type="bar"):
    """Process the query and return response and chart if applicable"""
    try:
        # Get response from RAG chain
        response = rag_chain.invoke(question)
        
        # Extract metrics from the response
        metrics = extract_metrics(response)
        
        if metrics:
            # Create visualization based on selected chart type
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
        gr.Textbox(label="Your Question", placeholder="Ask questions about Saudi companies, market trends, and financial metrics..."),
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
