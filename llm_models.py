from enum import Enum


class LLMModels(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


LLM_MODELS = {
    LLMModels.OPENAI: "openai:gpt-5-mini",
    LLMModels.GEMINI: "google_genai:gemini-3-flash-preview",
}
