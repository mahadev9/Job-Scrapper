from typing import Dict, List

from llm_models import LLM_MODELS, LLMModels

from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """
You are an expert job matching assistant. Your task is to analyze job listings against a candidate's resume and determine the best matches.

Analyze the provided resume carefully, focusing on:
- Technical skills and programming languages
- Years of experience and seniority level
- Domain expertise and industry knowledge
- Education and certifications
- Project experience and achievements

For each job listing, evaluate:
1. **Skills Match**: How well the candidate's technical skills align with job requirements
2. **Experience Level**: Whether the candidate's seniority matches the role
3. **Domain Fit**: Relevance of past work experience to the job's industry/domain
4. **Growth Potential**: Opportunities for the candidate to learn and grow

Scoring guidelines:
- 0.9-1.0: Excellent match - candidate meets or exceeds all key requirements
- 0.7-0.89: Strong match - candidate meets most requirements with minor gaps
- 0.5-0.69: Moderate match - candidate has relevant skills but notable gaps
- 0.3-0.49: Weak match - significant skill/experience gaps
- 0.0-0.29: Poor match - minimal alignment with requirements

Only include jobs with a match_score >= 0.5. Provide clear, specific reasons for each match.
"""

USER_PROMPT = """
Based on the resume provided, analyze the following job listings and return only the jobs that are a good match.

Candidate Resume Summary:
{resume_summary}

Job Listings:
{jobs}

Return a structured list of matched jobs with their match scores and reasons.
"""

RESUME_DETAILS = {
    "link": "https://www.maitri.pro/Mahadev%20Mahesh%20Maitri%20Resume.pdf",
    "summary": (
        "Results-driven AI/ML Engineer with 4+ years of experience specializing "
        "in intelligent document processing, RAG systems, and NLP applications. "
        "Expertise in developing production-ready solutions using Python, FastAPI, "
        "LangChain, and transformer models, achieving 92% extraction accuracy and "
        "87% precision in complex risk assessment tasks. Proven track record in "
        "fine-tuning LLMs, building agentic workflows with LangGraph, and implementing "
        "semantic search systems that improve response relevance by 20-25%. "
        "Strong foundation in deep learning frameworks (PyTorch, TensorFlow) "
        "with demonstrated ability to deliver measurable business impact through "
        "AI-driven automation and cost optimization."
    ),
}


class ApplyOptions(BaseModel):
    apply_url: str = Field(description="URL to apply for the job")
    apply_title: str = Field(description="Title of the application link")


class MatchedJob(BaseModel):
    job_title: str = Field(description="The title of the job")
    job_description: str = Field(description="The description of the job")
    job_location: str = Field(description="The location of the job")
    company_name: str = Field(description="The name of the company")
    match_score: float = Field(
        ge=0.0, le=1.0, description="Score representing the match quality"
    )
    match_reason: str = Field(description="Reason for the match")
    apply_options: List[ApplyOptions] = Field(default_factory=list)


class MatchedJobsResponse(BaseModel):
    matched_jobs: List[MatchedJob] = Field(default_factory=list)


def create_llm_client(model: LLMModels) -> BaseChatModel:
    args = {
        "model": LLM_MODELS[model],
        "max_retries": 3,
        "temperature": 1.0,
    }
    if model == LLMModels.OPENAI:
        args["use_responses_api"] = True
    return init_chat_model(**args)


def gemini_msg_content(jobs) -> List[Dict[str, str]]:
    return [
        {
            "type": "image_url",
            "image_url": RESUME_DETAILS["link"],
        },
        {
            "type": "text",
            "text": USER_PROMPT.format(
                jobs=jobs, resume_summary=RESUME_DETAILS["summary"]
            ),
        },
    ]


def openai_msg_content(jobs) -> List[Dict[str, str]]:
    return [
        {
            "type": "input_file",
            "file_url": RESUME_DETAILS["link"],
        },
        {
            "type": "input_text",
            "text": USER_PROMPT.format(
                jobs=jobs, resume_summary=RESUME_DETAILS["summary"]
            ),
        },
    ]


def filter_jobs_with_llm(model, jobs) -> MatchedJobsResponse:
    client = create_llm_client(model).with_structured_output(MatchedJobsResponse)

    content = []
    if model == LLMModels.GEMINI:
        content = gemini_msg_content(jobs)
    elif model == LLMModels.OPENAI:
        content = openai_msg_content(jobs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]

    return client.invoke(messages)
