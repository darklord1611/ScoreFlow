# executor_direct.py
from openai import OpenAI
import asyncio

class DirectOpenAIExecutor:
    """Executor trực tiếp gọi GPT-4o-mini qua HTTP, không qua MetaGPT."""
    def __init__(self,
                 base_url="https://api.yescale.io/v1",
                 api_key="sk-xxx",
                 model="gpt-4o-mini",
                 temperature=0.0):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120)
        self.model = model
        self.temperature = temperature

    async def aask(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        def _query():
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return resp.choices[0].message.content
        return await loop.run_in_executor(None, _query)
