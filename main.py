from __future__ import annotations

from pathlib import Path
import argparse
import sys

from langchain_core.messages import HumanMessage
import functools

from dotenv import load_dotenv
from agent.REAgent import REAgent




try:
    import gradio as gr
except Exception:  # pragma: no cover - gradio optional at runtime
    gr = None  # type: ignore



DEFAULT_MODEL = "gpt-4o-mini"
##todo: make link thread_id with user session
config = {"configurable": {"thread_id": "3"}}

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the REA chatbot.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI chat model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--prompt",
        help="Prompt to send. If omitted, you'll be interactively prompted.",
    )
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Launch a Gradio chat UI instead of CLI single-turn mode.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for Gradio server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio server (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio public sharing URL (use with caution)",
    )
    return parser.parse_args(argv)

def launch_gradio(agent: REAgent, host: str, port: int, share: bool) -> int:
    chat_with_agent = functools.partial(agent.chat, config=config)

    gr.ChatInterface(chat_with_agent,
                     type="messages",
                     ).launch()
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI/Gradio entry point."""
    load_dotenv(override=True)

    args = parse_args(argv)

    agent = REAgent(model=args.model)

    if getattr(args, "gradio", False):
        return launch_gradio(agent, host=args.host, port=args.port, share=args.share)

    # CLI single-turn mode
    try:
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(errors="replace")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


