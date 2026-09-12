# =============================================================================
# FILE: 01_setup_and_basics.py
# TOPIC: Setting up OpenAI SDK + Your first LLM calls
#
# BEFORE RUNNING:
#   pip install openai tiktoken python-dotenv
#
#   Create a .env file in this folder:
#   OPENAI_API_KEY=sk-your-key-here
#
# Get your API key at: https://platform.openai.com/api-keys
# =============================================================================

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # loads OPENAI_API_KEY from .env file

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------------------------------------------------------
# EXAMPLE 1: The simplest possible LLM call
# Think of this like calling any other API you've built in FastAPI.
# You send a request, you get a response. That's it.
# -----------------------------------------------------------------------------


def simple_call():
    """The most basic LLM call possible."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheaper model, great for learning
        messages=[{"role": "user", "content": "What is Python in one sentence?"}],
    )

    # The response is a structured object, not just a string
    # Navigate it like you would a JSON API response
    answer = response.choices[0].message.content
    print("Answer:", answer)
    print()

    # Token usage — always check this, it determines your cost
    print(f"Input tokens:  {response.usage.prompt_tokens}")
    print(f"Output tokens: {response.usage.completion_tokens}")
    print(f"Total tokens:  {response.usage.total_tokens}")


# -----------------------------------------------------------------------------
# EXAMPLE 2: System prompt + User message
# The system prompt is like middleware in your Express/FastAPI app —
# it runs before every request and shapes the model's behavior.
# -----------------------------------------------------------------------------


def with_system_prompt():
    """Using system prompt to give the model a persona and constraints."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior Python developer with 10 years of experience. "
                    "You explain concepts concisely, use code examples, "
                    "and always mention potential pitfalls."
                ),
            },
            {"role": "user", "content": "Explain Python decorators."},
        ],
        temperature=0.7,
        max_tokens=300,  # limit response length to save tokens
    )

    print(response.choices[0].message.content)


# -----------------------------------------------------------------------------
# EXAMPLE 3: Multi-turn conversation
# Key insight: LLMs are STATELESS. They remember nothing between API calls.
# You must manually pass the full conversation history every single time.
# This is exactly like passing JWT tokens — you carry the state yourself.
# -----------------------------------------------------------------------------


def multi_turn_conversation():
    """
    Simulating a multi-turn chat.
    Notice how we build up the messages list manually.
    This is how ALL chatbots work under the hood.
    """

    # Start with just the system prompt
    conversation_history = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep answers brief.",
        }
    ]

    # Simulate 3 turns of conversation
    user_messages = [
        "My name is Jipra. What's your name?",
        "What did I just tell you my name was?",  # tests if context is maintained
        "What programming language should I learn for AI engineering?",
    ]

    for user_msg in user_messages:
        print(f"\nUser: {user_msg}")

        # Add the new user message to history
        conversation_history.append({"role": "user", "content": user_msg})

        # Send the ENTIRE history every time — this is the key pattern
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=conversation_history, temperature=0.7
        )

        assistant_reply = response.choices[0].message.content
        print(f"Assistant: {assistant_reply}")

        # Add the assistant's reply to history for the next turn
        conversation_history.append({"role": "assistant", "content": assistant_reply})

    print(f"\nFinal conversation had {len(conversation_history)} messages")
    print(f"That's a lot of tokens being sent repeatedly!")
    # This is why context window management matters in production


# -----------------------------------------------------------------------------
# EXAMPLE 4: Temperature experiment
# Run this and see the difference yourself — reading about it is not enough.
# -----------------------------------------------------------------------------


def temperature_experiment():
    """
    Ask the same question with different temperatures.
    Shows the real impact of this parameter.
    """

    prompt = "Give me a one-sentence creative description of the color blue."
    temperatures = [0.0, 0.7, 1.5]

    print("=== Temperature Experiment ===")
    print(f"Prompt: '{prompt}'\n")

    for temp in temperatures:
        # Run it twice at each temperature to show consistency/randomness
        responses = []
        for _ in range(2):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=60,
            )
            responses.append(response.choices[0].message.content.strip())

        print(f"Temperature = {temp}:")
        print(f"  Run 1: {responses[0]}")
        print(f"  Run 2: {responses[1]}")
        print(f"  Identical: {responses[0] == responses[1]}")
        print()

    # At temp=0.0, Run 1 and Run 2 should be identical (deterministic)
    # At temp=1.5, they will be noticeably different


# -----------------------------------------------------------------------------
# Run all examples
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("EXAMPLE 1: Simple Call")
    print("=" * 60)
    simple_call()

    print("\n" + "=" * 60)
    print("EXAMPLE 2: With System Prompt")
    print("=" * 60)
    with_system_prompt()

    print("\n" + "=" * 60)
    print("EXAMPLE 3: Multi-turn Conversation")
    print("=" * 60)
    multi_turn_conversation()

    print("\n" + "=" * 60)
    print("EXAMPLE 4: Temperature Experiment")
    print("=" * 60)
    temperature_experiment()
