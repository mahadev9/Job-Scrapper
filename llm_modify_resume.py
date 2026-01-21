from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from llm_models import LLMModels, create_llm_client
from resume import RESUME_DETAILS

SYSTEM_PROMPT = """
You are an expert resume optimization assistant specializing in tailoring resume bullet points to job descriptions.

Your task is to refine existing resume bullet points to better align with a target job description while:
- Maintaining complete accuracy and truthfulness to the original experience
- Preserving all core achievements and responsibilities
- Following Google's XYZ format: "Accomplished [X] as measured by [Y] by doing [Z]"
- Emphasizing relevant skills and technologies mentioned in the job description
- Using strong action verbs and quantifiable metrics

Key Guidelines:
1. **Stay Truthful**: Never fabricate experiences, technologies, or metrics
2. **Subtle Optimization**: Reword and reframe, don't invent
3. **Keyword Alignment**: Incorporate relevant keywords from the job description naturally
4. **XYZ Format**: Structure each bullet as: Action + Result + Method
5. **Quantify Impact**: Emphasize measurable outcomes (%, numbers, scale)
6. **Relevance**: Prioritize bullets that align with the job requirements

Do NOT:
- Add technologies or skills not in the original resume
- Inflate numbers or achievements
- Change the fundamental nature of the work performed
- Make drastic alterations that misrepresent the candidate's experience
"""

USER_PROMPT = """
Job Description:
{job_description}

Original Resume Bullet Points:
{resume_details_points}

Task: Refine these resume bullet points to better align with the job description while maintaining complete accuracy. Follow Google's XYZ format: "Accomplished [X] as measured by [Y] by doing [Z]".

For each bullet point:
1. Identify relevant keywords from the job description
2. Reframe the accomplishment to highlight alignment
3. Ensure quantifiable metrics are present
4. Use strong, relevant action verbs
5. Keep the core truth of the original experience intact

Return ONLY the refined bullet points, maintaining the same number of points as provided.
"""


def gemini_msg_content(
    job_description: str, resume_details_points: str
) -> List[Dict[str, str]]:
    return [
        {
            "type": "image_url",
            "image_url": RESUME_DETAILS["link"],
        },
        {
            "type": "text",
            "text": USER_PROMPT.format(
                job_description=job_description,
                resume_details_points=resume_details_points,
            ),
        },
    ]


def openai_msg_content(
    job_description: str, resume_details_points: str
) -> List[Dict[str, str]]:
    return [
        {
            "type": "input_file",
            "file_url": RESUME_DETAILS["link"],
        },
        {
            "type": "input_text",
            "text": USER_PROMPT.format(
                job_description=job_description,
                resume_details_points=resume_details_points,
            ),
        },
    ]


def fetch_updated_resume_details(
    model: LLMModels, job_description: str, resume_details_points: str
) -> str:
    client = create_llm_client(model)

    content = []
    if model == LLMModels.GEMINI:
        content = gemini_msg_content(job_description, resume_details_points)
    elif model == LLMModels.OPENAI:
        content = openai_msg_content(job_description, resume_details_points)

    contents = client.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]
    ).content

    response_text = ""
    for item in contents:
        if item["type"] == "text":
            response_text += item["text"] + "\n"

    return response_text.strip()
