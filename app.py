import streamlit as st
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.firecrawl import FirecrawlTools
import os
from dotenv import load_dotenv
import json

# ✅ Move this to the first Streamlit command
st.set_page_config(page_title="Exam Syllabus Fetcher", layout="centered")

load_dotenv()

# Define helper classes and functions
class WebSearchAgent:
    def __init__(self, agent):
        self.agent = agent

    def search(self, exam_name):
        query = f"Fetch the latest Syllabus about the exam {exam_name}"
        response = self.agent.run(message=query)
        return response.content if response else None

class LinkExtractor:
    @staticmethod
    def extract_links(text):
        url_pattern = r"https?://[^\s<>\"']+"
        return re.findall(url_pattern, text)

class CrawlAgent:
    def __init__(self, agent):
        self.agent = agent

    def crawl(self, url, exam_name):
        message = f"Fetch the syllabus for the exam {exam_name} from this site {url}. Remember, don’t include promotions, courses, or external links."
        response = self.agent.run(message=message)
        return response.content if response else None

class ExamSyllabusFetcher:
    def __init__(self, web_agent, crawl_agent, llm):
        self.web_agent = web_agent
        self.crawl_agent = crawl_agent
        self.llm = llm

    def fetch_syllabus(self, exam_name, status_box):
        status_box.write("🔍 Searching for syllabus...")
        search_response = self.web_agent.search(exam_name)
        if not search_response:
            return "No relevant syllabus found."

        links = LinkExtractor.extract_links(search_response)
        syllabus_content = search_response if not links else ""
        if len(links) > 2:
            links = links[:2]

        for link in links:
            status_box.write(f"🌍 Crawling: {link}")
            syllabus_text = self.crawl_agent.crawl(link, exam_name)
            if syllabus_text:
                syllabus_content += f"Syllabus from {link}: {syllabus_text}\n\n"

        status_box.write("📝 Summarizing syllabus...")
        return self.summarize_syllabus(syllabus_content, exam_name)

    def summarize_syllabus(self, syllabus_text, exam_name):
        prompt = ChatPromptTemplate.from_template(
            "You have been given the latest syllabus for the {exam} exam from different websites. "
            "You must analyze this syllabus and return the final structured syllabus. "
            "You can integrate both the provided information and your understanding to provide an accurate syllabus. "
            "Return the output in a well-structured JSON format with the following structure: "
            "{{"
            "  \"exam\": \"{exam}\", "
            "  \"subjects\": ["
            "    {{"
            "      \"subject\": \"<subject_name>\", "
            "      \"topics\": ["
            "        {{"
            "          \"topic\": \"<topic_name>\", "
            "          \"subtopics\": [\"<subtopic_1>\", \"<subtopic_2>\", ...]"
            "        }}, ..."
            "      ]"
            "    }}, ..."
            "  ]"
            "}} "
        )

        parser = StrOutputParser()
        formatted_prompt = prompt.format(exam=exam_name, syllabus=syllabus_text)
        llm_response = self.llm.invoke(formatted_prompt)
        return parser.parse(llm_response)

# Initialize agents
Crawl = Agent(name="Crawl Agent", tools=[FirecrawlTools(scrape=False, crawl=True)], show_tool_calls=True)
WebSearch = Agent(name="Web Search Agent", tools=[DuckDuckGoTools()], show_tool_calls=True)
llm = ChatOpenAI(model_name="o3-mini", temperature=0.7)
fetcher = ExamSyllabusFetcher(WebSearchAgent(WebSearch), CrawlAgent(Crawl), llm)

# Streamlit UI
st.title("📘 Exam Syllabus Fetcher")
st.markdown("Enter the name of the exam, and the AI will fetch and summarize the latest syllabus for you.")

# ✅ Sidebar for API Key and Model Selection
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("🔑 OpenAI API Key", type="password")
    model_choice = st.selectbox("🤖 Choose OpenAI Model", ["o3-mini", "gpt-3.5-turbo", "gpt-4"])
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

exam_name = st.text_input("✏️ Exam Name:", placeholder="e.g., JEE, GATE, UPSC")

if st.button("🚀 Fetch Syllabus", use_container_width=True):
    if exam_name:
        status_box = st.empty()
        with st.spinner("Processing..."):
            syllabus_summary = fetcher.fetch_syllabus(exam_name, status_box)
            syllabus_summary = json.loads(syllabus_summary)  # ✅ Fix applied
        status_box.empty()
        st.success("✅ Syllabus Retrieved!")
        syllabus_str = json.dumps(syllabus_summary, indent=4)
        st.code(syllabus_str, language="json")
        st.download_button(
            label="📥 Download Syllabus JSON",
            data=syllabus_str,
            file_name="syllabus.json",
            mime="application/json"
        )
    else:
        st.warning("⚠️ Please enter an exam name.")
