import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PATIENT_ID = "1d604da9-9a81-4ba9-80c2-de3375d59b40"
ALLERGEN = "penicillin"
CONDITION = "Chronic sinusitis"

env = os.environ.copy()
env.update({
    "MCP_TRANSPORT": "stdio",
    "MONGO_URI": "mongodb://localhost:27017",
    "CHROMA_HOST": "localhost",
})

params = StdioServerParameters(
    command=sys.executable,          # the venv's python, not whatever is on PATH
    args=["-m", "src.server"],
    env=env,
)


async def main():
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("TOOLS")
            for t in tools.tools:
                first = (t.description or "").strip().split("\n")[0]
                print(f"  {t.name}: {first or '*** NO DESCRIPTION ***'}")

            print("\nALLERGY CHECK (expect a conflict)")
            r = await session.call_tool(
                "check_allergies",
                {"patient_id": PATIENT_ID, "proposed_drugs": [ALLERGEN, "metformin"]},
            )
            print(r.content[0].text)

            print("\nDRUG SEARCH")
            r = await session.call_tool(
                "query_drug_database", {"disease": CONDITION, "top_k": 3}
            )
            print(r.content[0].text[:800])


asyncio.run(main())