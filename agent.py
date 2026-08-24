# tools.py

import os

import pandas as pd
from dotenv import load_dotenv

import utils
import laptop_SP_search_tools
import aisuite



load_dotenv()



def _truncate_tool_results(
    messages: list[dict],
    max_chars: int = 2000
) -> list[dict]:
    """
    Cap every tool-role message at max_chars before sending to Groq.
    Tool results (DB rows as JSON) can be 8–15k chars; sending them
    in full eats the token budget and causes malformed tool-call syntax.
    """
    truncated = []
    for msg in messages:
        if msg["role"] == "tool" and len(msg.get("content", "")) > max_chars:
            truncated.append({
                **msg,
                "content": msg["content"][:max_chars] + "\n...[truncated]"
            })
        else:
            truncated.append(msg)
    return truncated


def _simplify_question(question: str) -> str:
    """
    Shorten a long question for retry after a tool_use_failed error.
    Groq's llama-3.3-70b-versatile produces the legacy XML function-call
    syntax on complex prompts; a shorter rewrite reliably avoids this.
    """
    if len(question) <= 120:
        return question
    return (
        "Suggest laptops/smartphones based on this query: "
        + question[:200]
    )


class _FallbackResponse:
    """
    Minimal mock of a Groq/aisuite response object returned when all
    retry attempts fail. Carries a plain-text error message as the
    assistant content with no tool_calls, so the agent loop treats it
    as a final (degraded) answer and exits cleanly instead of crashing.
    """
    class _Choice:
        class _Message:
            def __init__(self, content: str):
                self.content    = content
                self.tool_calls = None

            def model_dump(self):
                return {"content": self.content, "tool_calls": None}

        def __init__(self, content: str):
            self.message = _FallbackResponse._Choice._Message(content)

    def __init__(self, error_message: str):
        self.choices = [_FallbackResponse._Choice(error_message)]


def _call_groq_with_retry(
    client,
    model: str,
    messages: list[dict],
    tools: list[dict],
    original_question: str,
    max_retries: int = 2
):
    """
    Call the LLM with automatic retry on tool_use_failed (400) errors.

    Attempt 0 : normal call.
    Attempt 1  : retry identical messages (transient errors often resolve).
    Attempt 2  : replace last user message with a simplified question.

    On last-attempt failure the function NO LONGER re-raises.
    Instead it returns a _FallbackResponse whose content is the error
    message. The agent loop sees no tool_calls and treats it as a
    degraded final answer, so subsequent rounds and tool calls in the
    outer agent are unaffected.

    Non-tool-fail errors (rate limits, auth, network) are still
    re-raised immediately on every attempt because retrying them with
    a simplified question would not help.
    """
    last_user_idx = max(
        i for i, m in enumerate(messages) if m["role"] == "user"
    )

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
        except Exception as e:
            err = str(e)
            is_tool_fail = (
                "tool_use_failed" in err
                or "Failed to call a function" in err
            )
            is_last_attempt = attempt == max_retries

            # Non-tool-fail errors: re-raise immediately — retrying won't help
            if not is_tool_fail:
                raise

            last_error = e

            if is_last_attempt:
                # All retries exhausted — return graceful fallback so the
                # agent loop can continue rather than crashing
                fallback_msg = (
                    f"[tool_use_failed after {max_retries + 1} attempts — "
                    f"could not generate a valid tool call. "
                    f"Error: {err}]"
                )
                print(f"  [_call_groq_with_retry] All retries exhausted. "
                      f"Returning fallback response.")
                return _FallbackResponse(fallback_msg)

            if attempt == 0:
                print(f"  [Retry {attempt + 1}] tool_use_failed — retrying same messages...")
            else:
                simplified = _simplify_question(original_question)
                print(f"  [Retry {attempt + 1}] tool_use_failed — retrying with simplified question...")
                messages = [
                    *messages[:last_user_idx],
                    {"role": "user", "content": simplified},
                ]


def LS_research_agent(
    user_question: str,
    model: str ="groq:openai/gpt-oss-120b",
    system_prompt: str | None = None,
    MAX_TOOL_CALLS: int = 2,
    verbose: bool = True,
    show_tool_results: bool =False,
    show_thinking: bool =False,
    show: bool= False
) -> str:
    """
    Agentic loop that recommends laptops/smartphones by querying the Milvus Collections through laptop_SP_search_tools.

    Uses laptop_SP_search_tools.get_available_tools() for its internal tool schemas
    and laptop_SP_search_tools.handle_tool_call() / laptop_SP_search_tools.create_tool_response_message()
    for execution — no tool logic is duplicated here.

    Parameters
    ----------
    client00         : LLM client (aisuite or groq SDK instance).
    model            : Model string e.g. 'groq:openai/gpt-oss-120b'.
    user_question    : The user's legal question in plain language.
    system_prompt    : Optional override; defaults to the Bangladesh
                       Labour Act expert prompt below.
    MAX_TOOL_CALLS  : Max number of times model can call tools per turn.
    verbose          : If True, logs each tool call via utils.

    Returns
    -------
    final_answer : str
    """
    utils.log_agent_title_html("Laptops/Smartphone Recommendor Agent", "🕵🏼‍♂️")

    if system_prompt is None:
        system_prompt = f"""
            You are an Expert Broker Agent who suggests products like laptops and smartphones from the available database based on the customer queries.
            You have access to a Milvus-based hybrid vector database containing 1949 devices information (different specs fields, price, description, reviews; (all stored as device metadata)),
            under these unique categories (all stored as device metadata): Laptop, Smartphone, Feature Phone, Mobile Phone, Premium Ultrabook, Gaming Laptop.

            You have access to the following tools to retrieve those devices following user queries: 
            - hybrid_search (primary; recommended for category wise filtered results)
            - dense_search (supports context based search only; not recommended for category wise filtered results)
            - hybrid_search_rrf (supports category wise filtered results)
            
                       
            Your Task:
            1. Write a proper query (Use your expert device knowledge and expert broker brain to write specs-based queries!) to discover exactly 5-7 laptops/smartphones that you think will match with the customers specifications and expectations.
            2. Now call one (that you find appropriate) of the supplied tools with your written query to search the database. 
            3. Now present those (atleast 5/7) found devices in your output message with proper mapped explanations to why they are appropriate for their purpose (You should include review references[if available] to justify your recommendation).
            NB: avoid presenting devices that have bad reviews. 
            
            However you Must follow the Rules:
            - No answer will be from direct generation, you must and you will use the search tools first to retrive matching devices.
            - You can call tools only once/twice, and you have only one turn (this turn) per conversation, so use your queries effeciently.
            - if the number of tool calls has reached 2, terminate further tool calls and give the final recommendations based on what you have retrieved.
            - top_k for all tools must be < 10.
            
            You must follow this Output Format for your Recommendations message:
            1) Device1 name
               Device1 specs (all)
               URL (if available)
               explanation why it is a good choice
            2) Device2 name
               Device2 specs (all)
               URL (if available)
               explanation why it is a good choice
            3) Device3 name
               Device3 specs (all)
               URL (if available)
               explanation why it is a good choice
                                .
                                . 
                                .  
                                (list atleast 5/7 devices if you can)
        """


# =========================
# Environment & Client
# =========================

    if model=="groq:openai/gpt-oss-120b":
        client = aisuite.Client()
    elif model=="gemini-3.5-flash":
        client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    elif model=="nvidia/nemotron-3-super-120b-a12b" or model=="openai/gpt-oss-120b":
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1/"
        )
    else:
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1/"
        )        
        
        
        
        
    # Fetch tool schemas from laptop_SP_search_tools — single source of truth
    tools = laptop_SP_search_tools.get_available_tools()

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_question},
    ]
    

    #nothing now   # mirrors system prompt cap
    tool_call_count = 0

    for round_num in range(8):

        safe_messages = _truncate_tool_results(messages, max_chars=2000)

        response = _call_groq_with_retry(
            client=client,
            model=model,
            messages=safe_messages,
            tools=tools,
            original_question=user_question,
        )

        assistant_message = response.choices[0].message

        # Strip SDK-injected fields (e.g. reasoning_content) that Groq
        # rejects when they appear back in subsequent requests
        if hasattr(assistant_message, "model_dump"):
            raw = assistant_message.model_dump()
        else:
            raw = dict(assistant_message)

        clean_assistant_msg = {"role": "assistant", "content": raw.get("content")}
        if raw.get("tool_calls"):
            clean_assistant_msg["tool_calls"] = raw["tool_calls"]
            
        
        
        if tool_call_count >= MAX_TOOL_CALLS and raw.get("tool_calls"):
            messages.append({
                "role":    "system",
                "content": f"""
                        As the max tool call limit is reached, you can not call tools anymore!
                        you have to give the recommendations (following the output format) based on what you have retrieved.
                        """
            })
            continue
        
        

        # No tool calls → LLM produced the final answer
        if not raw.get("tool_calls"):

            # Force at least one tool call before accepting an answer
            if tool_call_count == 0:
                messages.append({
                    "role":    "system",
                    "content": "You have to call a tool before giving any answer."
                })
                continue
            
            
            final_answer = raw.get("content", "")
            if tool_call_count >= MAX_TOOL_CALLS and (not final_answer):
                messages.append({
                    "role":    "system",
                    "content": "You must give recommendations following the Output Format!"
                })
                continue

            
            messages.append(clean_assistant_msg)
            if show:
                utils.log_final_summary_html(final_answer)
            return final_answer

        # Tool call round — execute each call via laptop_SP_search_tools dispatcher
        messages.append(clean_assistant_msg)

        for tool_call in assistant_message.tool_calls:
            if verbose:
                if show_thinking:                
                    utils.log_tool_call_html(
                        tool_call.function.name,
                        tool_call.function.arguments
                    )

            tool_result_raw = laptop_SP_search_tools.handle_tool_call(tool_call)

            # NULL-result single retry
            if tool_result_raw is None:
                print(
                    f"  [NULL RESULT] Tool returned None — retrying once: "
                    f"{tool_call.function.name}"
                )
                tool_result_raw = laptop_SP_search_tools.handle_tool_call(tool_call)
                if tool_result_raw is None:
                    print("  [NULL RESULT] Retry also returned None — using empty result.")
                    tool_result_raw = {"error": "Tool returned no result after retry."}
                    
            if show_tool_results:
                utils.log_tool_result_html(tool_result_raw)

            messages.append(
                laptop_SP_search_tools.create_tool_response_message(tool_call, tool_result_raw)
            )

            tool_call_count += 1

            if tool_call_count >= MAX_TOOL_CALLS:
                messages.append({
                    "role":    "system",
                    "content": f"""
                        As the max tool call limit is reached,
                        you have to terminate further tool calls and give the recommendations based on what you have retrieved.
                        """
                })
                break

        # If tool call limit was reached mid-round, loop immediately so
        # the model can produce its final answer from retrieved contexts
        if tool_call_count >= MAX_TOOL_CALLS:
            continue

    return "[Max tool rounds reached without a final answer. Try narrowing your question.]"
