import asyncio
import yaml
from graph_401_1265 import Workflow
from metagpt.configs.models_config import ModelsConfig

async def main():
    # Load config2.yaml để lấy thông tin executor
    with open("config/config2.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Tạo config chuẩn cho Workflow
    llm_config = ModelsConfig.default().get(config["llm"]["model"])

    # Định nghĩa một bài toán mẫu
    problem = "If a rectangle has a width of 3 cm and a length of 8 cm, what is its area?"

    # Tạo instance Workflow và chạy
    wf = Workflow(config=llm_config, problem=problem)
    result = await wf()
    print("\n===== Workflow Output =====\n")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
