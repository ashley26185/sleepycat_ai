from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.providers.huggingface import HuggingFaceProvider
from pydantic_ai import Agent  # Import UserMessage directly from pydantic_ai
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    HF_TOKEN: str

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()

def main():
    print("Hello from pydantic-huggingface!")
    provider = HuggingFaceProvider(api_key=settings.HF_TOKEN)
    model = HuggingFaceModel(model_name="Qwen/QwQ-32B", provider=provider)
    agent = Agent(model,
                  instructions='Be concise, reply with one sentence.')

    # Use the synchronous API with a plain string prompt (latest pydantic-ai expects this for HuggingFace)
    result_sync = agent.run_sync("Is QuantumScape a good stock to invest in based on current progress by the company and why?")
    print(result_sync)

if __name__ == "__main__":
    main()


