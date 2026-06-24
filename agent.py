import os
import json
import asyncio

from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# ---------------------------------------------------
# Load environment variables
# ---------------------------------------------------
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY is not set.\n"
        "Run:\n"
        'export GOOGLE_API_KEY="YOUR_API_KEY"'
    )

# ---------------------------------------------------
# Load orders
# ---------------------------------------------------
with open("orders.json", "r") as f:
    ORDERS = json.load(f)


# ---------------------------------------------------
# Tool 1
# ---------------------------------------------------
def lookup_order(order_id: str):
    """
    Returns an order from orders.json.
    """
    order = ORDERS.get(order_id)

    if order is None:
        return {"error": f"Order '{order_id}' not found."}

    return order


# ---------------------------------------------------
# Tool 2
# ---------------------------------------------------
def calculate(expression: str):
    """
    Evaluate simple arithmetic.
    Example:
        1200 * 2
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------
# Agent
# ---------------------------------------------------
agent = Agent(
    name="orders_assistant",
    model="gemini-3.1-flash-lite",
    description="Helpful order assistant.",
    instruction="""
You are a helpful orders assistant.

Always use lookup_order() whenever order information is needed.

Always use calculate() whenever arithmetic is needed.

If an order cannot be found, clearly say so.

For warranty questions:

- Warranty expires after purchase_date + warranty_months.
- Assume today's date is 2026-06-24.
- Explain whether the product is still under warranty.
""",
    tools=[lookup_order, calculate],
)


# ---------------------------------------------------
# IDs
# ---------------------------------------------------
APP_NAME = "orders_app"
USER_ID = "user1"
SESSION_ID = "session1"


async def main():

    # Create session
    session_service = InMemorySessionService()

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    query = """
        Tell me about order A9999.
        Is it still under warranty?
    """

    content = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )

    print("\n========== AGENT TRACE ==========\n")

    for event in runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    ):

        if event.content is None:
            continue

        for part in event.content.parts:

            if getattr(part, "function_call", None):

                fc = part.function_call

                print("\n🔧 Tool Call")
                print(f"   {fc.name}({dict(fc.args)})")

            elif getattr(part, "function_response", None):

                fr = part.function_response

                print("\n✅ Tool Response")
                print(fr.response)

            elif getattr(part, "text", None):

                print("\n🤖 Final Answer\n")
                print(part.text)

    print("\n========== END ==========")


if __name__ == "__main__":
    asyncio.run(main())