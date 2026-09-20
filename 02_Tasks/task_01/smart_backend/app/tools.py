# app/tools.py

import json
import logging

logger = logging.getLogger(__name__)

# ============================================================
# TOOL 1: CALCULATOR
# ============================================================

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform basic arithmetic calculations.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform."
                },
                "a": {"type": "number", "description": "The first number."},
                "b": {"type": "number", "description": "The second number."}
            },
            "required": ["operation", "a", "b"]
        }
    }
}

def calculate(operation: str, a: float, b: float) -> float | str:
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            return "Error: Cannot divide by zero."
        return a / b
    return f"Error: Unknown operation '{operation}'."

# ============================================================
# TOOL 2: WEATHER
# ============================================================

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city."
                }
            },
            "required": ["city"]
        }
    }
}

MOCK_WEATHER = {
    "london": "15°C, rainy",
    "tokyo": "28°C, sunny",
    "new york": "22°C, cloudy",
}

def get_weather(city: str) -> str:
    return MOCK_WEATHER.get(city.lower(), f"{city}: 20°C, Clear skies")

# ============================================================
# TOOL 3: WEB SEARCH
# ============================================================

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information about a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                }
            },
            "required": ["query"]
        }
    }
}

def web_search(query: str) -> list[str]:
    return [
        f"Result 1 for '{query}': Introduction and overview of the topic.",
        f"Result 2 for '{query}': Latest information and useful resources.",
        f"Result 3 for '{query}': Tutorials, examples, and practical guides."
    ]

# ============================================================
# REGISTER ALL TOOLS
# ============================================================

TOOLS = [CALCULATOR_TOOL, WEATHER_TOOL, WEB_SEARCH_TOOL]

TOOL_FUNCTIONS = {
    "calculator": calculate,
    "get_weather": get_weather,
    "web_search": web_search
}

# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(tool_name: str, arguments: str):
    """
    Execute the requested tool using the arguments returned by the LLM.
    """
    function = TOOL_FUNCTIONS.get(tool_name)

    if function is None:
        logger.warning("Tool called but not found | tool_name=%s", tool_name)
        return f"Error: Unknown tool '{tool_name}'."

    logger.info("Tool called | tool_name=%s | arguments=%s", tool_name, arguments)

    try:
        args = json.loads(arguments)
        result = function(**args)

        # Preview: first 100 chars of the result to avoid bloating logs
        result_preview = str(result)[:100]
        logger.debug("Tool result | tool_name=%s | result_preview=%s", tool_name, result_preview)

        return result

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.error(
            "Tool execution failed | tool_name=%s | error_type=%s | detail=%s",
            tool_name,
            type(e).__name__,
            str(e)
        )
        return f"Error executing tool: {str(e)}"
