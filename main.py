from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.providers.huggingface import HuggingFaceProvider
from pydantic_ai import Agent  # Import UserMessage directly from pydantic_ai
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging
import logging.config
import yaml

def setup_logging(config_path="logging_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)

class Settings(BaseSettings):
    HF_TOKEN: str
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()

def main():
    setup_logging()
    # Get a logger object
    logger = logging.getLogger("pydnatic_huggingface")
    print("Hello from pydantic-huggingface!")
    provider = HuggingFaceProvider(api_key=settings.HF_TOKEN)
    model = HuggingFaceModel(model_name="Qwen/QwQ-32B", provider=provider)
    agent = Agent(model,
                  instructions='Respond in a paragraph of not more than 5 sentences.')

    # Use the synchronous API with a plain string prompt (latest pydantic-ai expects this for HuggingFace)
    result_sync = agent.run_sync("What is the weather like in Phuket in December?")
    print(result_sync)
    logger.info(f'Synchronous result: {result_sync}')

if __name__ == "__main__":
    main()


