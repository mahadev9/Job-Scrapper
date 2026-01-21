from enum import Enum

from langchain.chat_models import BaseChatModel, init_chat_model


class LLMModels(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


LLM_MODELS = {
    LLMModels.OPENAI: "openai:gpt-5-mini",
    LLMModels.GEMINI: "google_genai:gemini-3-flash-preview",
}


def create_llm_client(model: LLMModels) -> BaseChatModel:
    args = {
        "model": LLM_MODELS[model],
        "max_retries": 3,
        "temperature": 1.0,
    }
    if model == LLMModels.OPENAI:
        args["use_responses_api"] = True
    return init_chat_model(**args)
