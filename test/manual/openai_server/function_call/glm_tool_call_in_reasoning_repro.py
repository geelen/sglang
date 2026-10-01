"""Reproduce GLM tool calls whose names absorb reasoning text.

When a GLM model writes the literal ``<tool_call>`` inside its reasoning, the
``glm45`` reasoning parser's streaming path ends the reasoning block there and
the ``glm47`` tool parser turns the rest of the reasoning, ``</think>`` and any
visible text into the "function name" of the next real tool call. Non-streaming
requests split at ``</think>`` and are not affected.

Subcommands (run next to a server started with ``--reasoning-parser glm45
--tool-call-parser glm47``; ``replay`` needs the checkpoint tokenizer):

  run        send one matrix cell of chat requests and append JSONL rows
  replay     feed the exact output token ids of non-streaming rows through the
             installed streaming and one-shot parsers, as serving_chat does
  summarize  count malformed tool names per cell

The request bodies are ordinary OpenAI chat requests (messages, tools,
reasoning_effort, max_tokens); ``model`` and ``stream`` are overridden.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

STOP_TOKENS = ("<|user|>", "<|observation|>", "<|endoftext|>")


def classify_names(names, declared):
    """``malformed``: a non-empty name the request never declared.

    ``phantom_call``: a call that streamed arguments (``{}``) but no name. That
    happens when the tool parser saw ``<tool_call>`` but never a complete name,
    and the end-of-stream flush sent the placeholder's arguments anyway.
    """
    bad = [n for n in names if n and n not in declared]
    return {
        "bad_names": bad,
        "malformed": bool(bad),
        "phantom_call": any(not n for n in names),
        "name_has_think_end": any("</think>" in n for n in names),
        "name_has_tool_call": any("<tool_call>" in n for n in names),
        "name_has_whitespace": any(any(c.isspace() for c in n) for n in names),
    }


def classify_leak(result):
    """Reasoning markup that a correct parse never puts outside reasoning.

    ``defect`` is the parser defect: an undeclared tool name, or reasoning
    delivered as content (``reasoning_leak``: the content starts at a
    ``<tool_call>`` the reasoning mentioned and carries the block's
    ``</think>``). Two model behaviours are counted separately because no
    parser change affects them: ``runaway`` (after a correctly split answer the
    model wrote another turn, with another ``</think>``; without tools the
    ``<|assistant|>`` separator is a skipped special token) and
    ``args_have_think_end`` (``</think>`` inside the arguments of a call that a
    constrained decoder forced after the model emitted ``<tool_call>`` in its
    visible text).
    """
    calls = result.get("tool_calls", [])
    content = result.get("content", "")
    leak = content.lstrip().startswith("<tool_call>") and "</think>" in content
    flags = {
        "reasoning_leak": leak,
        "runaway": (not leak and "</think>" in content)
        or "<|assistant|>" in content
        or any("<|assistant|>" in c["arguments"] for c in calls),
        "args_have_think_end": any("</think>" in c["arguments"] for c in calls),
    }
    flags["defect"] = bool(result.get("malformed") or leak)
    return flags


def build_body(template, model, stream, tools, sampling, seed, max_tokens):
    body = {k: v for k, v in template.items() if k not in ("model", "stream")}
    body["model"] = model
    body["stream"] = stream
    if tools == "none":
        body.pop("tools", None)
        body.pop("tool_choice", None)
    if max_tokens:
        body["max_tokens"] = max_tokens
    if sampling == "greedy":
        body["temperature"] = 0
    body["seed"] = seed
    if stream:
        body["stream_options"] = {"include_usage": True}
    else:
        body["return_token_ids"] = True
    return body


def post(base_url, body, timeout):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def read_stream(response, started):
    """Accumulate an SSE chat stream exactly as a client would."""
    reasoning, content, chunks = [], [], []
    calls, finish, usage = {}, None, None
    timing, last = {}, [None]

    def mark(kind):
        now = time.time()
        timing.setdefault(kind, round(now - started, 3))
        if last[0] is not None:
            timing["max_gap_s"] = max(
                timing.get("max_gap_s", 0), round(now - last[0], 3)
            )
        last[0] = now

    for raw in response:
        line = raw.decode("utf-8").rstrip("\n")
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :]
        if data == "[DONE]":
            break
        event = json.loads(data)
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                mark("first_reasoning_s")
                reasoning.append(delta["reasoning_content"])
                timing["max_reasoning_chunk_chars"] = max(
                    timing.get("max_reasoning_chunk_chars", 0),
                    len(delta["reasoning_content"]),
                )
                chunks.append(["reasoning", delta["reasoning_content"]])
            if delta.get("content"):
                mark("first_answer_s")
                content.append(delta["content"])
                chunks.append(["content", delta["content"]])
            for call in delta.get("tool_calls") or []:
                mark("first_answer_s")
                slot = calls.setdefault(
                    call.get("index", 0), {"name": "", "arguments": ""}
                )
                function = call.get("function") or {}
                if function.get("name"):
                    slot["name"] += function["name"]
                    chunks.append(["tool_name", function["name"]])
                if function.get("arguments"):
                    slot["arguments"] += function["arguments"]
                    chunks.append(["tool_args", function["arguments"]])
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return {
        "reasoning_content": "".join(reasoning),
        "content": "".join(content),
        "tool_calls": [calls[i] for i in sorted(calls)],
        "finish_reason": finish,
        "usage": usage,
        "chunks": chunks,
        **timing,
    }


def read_json(response):
    payload = json.loads(response.read())
    choice = payload["choices"][0]
    message = choice["message"]
    return {
        "reasoning_content": message.get("reasoning_content") or "",
        "content": message.get("content") or "",
        "tool_calls": [
            {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
            for c in message.get("tool_calls") or []
        ],
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage"),
        "output_ids": choice.get("response_token_ids") or choice.get("token_ids"),
        "prompt_tail_ids": (choice.get("prompt_token_ids") or [])[-4:],
    }


def one_request(args, template, declared, index):
    seed = args.seed_base + (0 if args.sampling == "greedy" else index)
    body = build_body(
        template,
        args.model,
        args.stream,
        args.tools,
        args.sampling,
        seed,
        args.max_tokens,
    )
    started = time.time()
    row = {
        "label": args.label,
        "prompt": args.prompt_name,
        "stream": args.stream,
        "tools": args.tools,
        "concurrency": args.concurrency,
        "sampling": args.sampling,
        "seed": seed,
        "index": index,
        "declared": sorted(declared) if args.tools == "declared" else [],
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
    }
    try:
        with post(args.base_url, body, args.timeout) as response:
            row["status"] = response.status
            row.update(
                read_stream(response, started) if args.stream else read_json(response)
            )
    except urllib.error.HTTPError as error:
        row["status"] = error.code
        row["error"] = error.read().decode("utf-8", "replace")[:2000]
    except Exception as error:  # transport failures are recorded, not retried
        row["status"] = None
        row["error"] = repr(error)[:2000]
    row["elapsed_s"] = round(time.time() - started, 3)
    names = [c["name"] for c in row.get("tool_calls", [])]
    row.update(classify_names(names, declared))
    row.update(classify_leak(row))
    return row


def cmd_run(args):
    template = json.load(open(args.prompt))
    declared = {t["function"]["name"] for t in template.get("tools", [])}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(
            pool.map(lambda i: one_request(args, template, declared, i), range(args.n))
        )
    with open(args.out, "a") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    bad = sum(r["malformed"] for r in rows)
    defect = sum(r["defect"] for r in rows)
    errors = sum(r.get("status") != 200 for r in rows)
    print(
        f"{args.label} {args.prompt_name} stream={args.stream} tools={args.tools} "
        f"{args.sampling} c={args.concurrency}: malformed {bad}/{len(rows)}, "
        f"defect {defect}/{len(rows)}, errors {errors}"
    )


def incremental_deltas(tokenizer, ids, skip_special_tokens):
    """Token-by-token text deltas, holding back incomplete UTF-8 like the server."""
    deltas, emitted = [], ""
    for i in range(1, len(ids) + 1):
        text = tokenizer.decode(ids[:i], skip_special_tokens=skip_special_tokens)
        if text.endswith("�"):
            continue
        deltas.append(text[len(emitted) :])
        emitted = text
    return deltas


def replay_row(row, tokenizer, tools, force_reasoning):
    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    tools = tools if row["tools"] == "declared" else []
    tool_objs = [Tool(**t) for t in tools]
    ids = list(row["output_ids"])
    # The matched stop token is returned in the ids but never in the text.
    stops = {tokenizer.convert_tokens_to_ids(t) for t in STOP_TOKENS}
    if ids and ids[-1] in stops:
        ids = ids[:-1]
    # serving_chat keeps special tokens only for tool-enabled requests.
    skip = not tools
    raw = tokenizer.decode(ids, skip_special_tokens=skip)
    tool_call_id = tokenizer.convert_tokens_to_ids("<tool_call>")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    end = ids.index(think_end_id) if think_end_id in ids else len(ids)
    # Tokens from the first <tool_call> inside reasoning to </think>: what a
    # parser that waits for </think> holds back (special id or plain text).
    first_mention = None
    for i in range(end):
        if ids[i] == tool_call_id or "<tool_call>" in tokenizer.decode(
            ids[max(0, i - 8) : i + 1], skip_special_tokens=False
        ):
            first_mention = i
            break
    if force_reasoning is None:
        # GLM templates end the prompt with <think>, so output starts inside it.
        force_reasoning = not raw.startswith("<think>")

    # Streaming path: reasoning parser first, its normal text into the tool parser.
    reasoning_parser = ReasoningParser("glm45", True, force_reasoning)
    tool_parser = FunctionCallParser(tool_objs, "glm47")
    reasoning, content, calls = [], [], {}
    deltas = incremental_deltas(tokenizer, ids, skip)
    for i, delta in enumerate(deltas):
        last = i == len(deltas) - 1
        r, n = reasoning_parser.parse_stream_chunk(delta)
        if last:
            er, en = reasoning_parser.parse_stream_end()
            r, n = (r or "") + (er or ""), (n or "") + (en or "")
        if r:
            reasoning.append(r)
        if tools:
            normal, items = tool_parser.parse_stream_chunk(n or "")
            if last:
                end_text, end_items = tool_parser.parse_stream_end()
                normal, items = (normal or "") + end_text, list(items) + end_items
        else:
            normal, items = n, []
        if normal:
            content.append(normal)
        for item in items:
            slot = calls.setdefault(item.tool_index, {"name": "", "arguments": ""})
            slot["name"] += item.name or ""
            slot["arguments"] += item.parameters or ""
    flush_error = None
    if tools:
        # serving_chat._check_for_unstreamed_tool_args: after the last chunk it
        # sends whatever arguments the detector recorded but did not stream,
        # under the last recorded index and without a name.
        detector = tool_parser.detector
        records = getattr(detector, "prev_tool_call_arr", None) or []
        streamed = getattr(detector, "streamed_args_for_tool", None) or []
        index = len(records) - 1
        if 0 <= index < len(streamed):
            expected = records[index].get("arguments", {})
            if not isinstance(expected, str):
                try:
                    expected = json.dumps(expected, ensure_ascii=False)
                except TypeError as error:  # e.g. "..." parsed to Ellipsis
                    flush_error = repr(error)
                    expected = streamed[index]
            if (
                expected.startswith(streamed[index])
                and expected[len(streamed[index]) :]
            ):
                slot = calls.setdefault(index, {"name": "", "arguments": ""})
                slot["arguments"] += expected[len(streamed[index]) :]
    streaming = {
        "flush_error": flush_error,
        "reasoning_content": "".join(reasoning),
        "content": "".join(content),
        "tool_calls": [calls[i] for i in sorted(calls)],
    }

    # One-shot path, as _build_chat_response does for stream=false.
    one_reasoning, one_text = ReasoningParser(
        "glm45", False, force_reasoning
    ).parse_non_stream(raw)
    one_text_after, one_calls = one_text, []
    if tools and FunctionCallParser(tool_objs, "glm47").has_tool_call(one_text):
        one_text_after, one_calls = FunctionCallParser(
            tool_objs, "glm47"
        ).parse_non_stream(one_text)
    one_shot = {
        "reasoning_content": one_reasoning or "",
        "content": one_text_after,
        "tool_calls": [{"name": c.name, "arguments": c.parameters} for c in one_calls],
    }
    declared = {t["function"]["name"] for t in tools}
    return {
        "raw_text": raw,
        "n_tokens": len(ids),
        "think_end_token_index": end if end < len(ids) else None,
        "tool_call_special_ids_before_think_end": ids[:end].count(tool_call_id),
        "tool_call_text_before_think_end": tokenizer.decode(
            ids[:end], skip_special_tokens=False
        ).count("<tool_call>"),
        "tool_call_special_ids_after_think_end": ids[end:].count(tool_call_id),
        "held_tokens": None if first_mention is None else end - first_mention,
        "streaming": streaming,
        "streaming_class": classify_names(
            [c["name"] for c in streaming["tool_calls"]], declared
        ),
        "one_shot": one_shot,
        "one_shot_class": classify_names(
            [c["name"] for c in one_shot["tool_calls"]], declared
        ),
    }


def cmd_replay(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    template = json.load(open(args.prompt))
    tools = template.get("tools", [])
    with open(args.out, "a") as out:
        for line in open(args.rows):
            row = json.loads(line)
            if row.get("stream") or not row.get("output_ids"):
                continue
            if args.label and row["label"] != args.label:
                continue
            if row["prompt"] != args.prompt_name:
                continue
            force = None if args.force_reasoning is None else bool(args.force_reasoning)
            result = replay_row(row, tokenizer, tools, force)
            result.update(
                {k: row[k] for k in ("label", "prompt", "seed", "index", "sampling")}
            )
            result["server"] = {
                k: row.get(k)
                for k in ("reasoning_content", "content", "tool_calls", "malformed")
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"{row['label']} seed={row['seed']} idx={row['index']}: "
                f"special-id <tool_call> before </think>="
                f"{result['tool_call_special_ids_before_think_end']}, "
                f"replayed streaming malformed={result['streaming_class']['malformed']}, "
                f"one-shot malformed={result['one_shot_class']['malformed']}, "
                f"server(non-stream) malformed={row['malformed']}"
            )


def cmd_replay_summary(args):
    """Fidelity and outcome counts for ``replay`` output."""
    cells = {}
    for path in args.rows:
        for line in open(path):
            row = json.loads(line)
            server = row["server"]
            one = row["one_shot"]
            stream = row["streaming"]
            stream_flags = classify_leak(
                {**stream, "malformed": row["streaming_class"]["malformed"]}
            )
            cell = cells.setdefault(
                (row["label"], row["prompt"]),
                {
                    k: 0
                    for k in (
                        "n",
                        "faithful",
                        "mention",
                        "text_mention",
                        "stream_any",
                        "stream_defect",
                        "stream_nameless",
                        "one_shot_defect",
                        "reasoning_equal",
                        "calls_equal",
                    )
                },
            )
            cell["n"] += 1
            cell["faithful"] += (
                one["reasoning_content"] == server["reasoning_content"]
                and one["content"] == server["content"]
                and [c["name"] for c in one["tool_calls"]]
                == [c["name"] for c in server["tool_calls"]]
            )
            mention = row["tool_call_special_ids_before_think_end"] > 0
            cell["mention"] += mention
            cell["text_mention"] += (
                row["tool_call_text_before_think_end"]
                > row["tool_call_special_ids_before_think_end"]
            )
            cell["stream_defect"] += stream_flags["defect"]
            nameless = row["streaming_class"].get("phantom_call", False)
            cell["stream_nameless"] += nameless
            cell["stream_any"] += (
                stream_flags["defect"]
                or nameless
                or stream["reasoning_content"] != one["reasoning_content"]
            )
            cell["one_shot_defect"] += row["one_shot_class"]["malformed"]
            cell["reasoning_equal"] += (
                stream["reasoning_content"] == one["reasoning_content"]
            )
            cell["calls_equal"] += [c["name"] for c in stream["tool_calls"]] == [
                c["name"] for c in one["tool_calls"]
            ]
    print(
        "label\tprompt\tn\treplay==server(json)\t<tool_call>-id-in-reasoning"
        "\ttext-only-mention\tSTREAMING-ANY-DEFECT\tstreaming-malformed-or-leak"
        "\tstreaming-nameless\tone-shot-defect\tstream-reasoning==one-shot"
        "\tstream-call-names==one-shot"
    )
    for key in sorted(cells):
        c = cells[key]
        cols = (
            "n",
            "faithful",
            "mention",
            "text_mention",
            "stream_any",
            "stream_defect",
            "stream_nameless",
            "one_shot_defect",
            "reasoning_equal",
            "calls_equal",
        )
        print("\t".join(key) + "".join(f"\t{c[k]}" for k in cols))


def cmd_latency(args):
    """Client-side stream timing per cell: median / p90 / max seconds."""
    cells = {}
    for path in args.rows:
        for line in open(path):
            row = json.loads(line)
            if not row["stream"] or row.get("status") != 200:
                continue
            key = (row["label"], row["prompt"], row["sampling"], row["concurrency"])
            cells.setdefault(key, []).append(row)

    def stats(values):
        values = sorted(v for v in values if v is not None)
        if not values:
            return "-"
        p90 = values[max(0, int(0.9 * len(values)) - 1)]
        return f"{values[len(values) // 2]:.2f}/{p90:.2f}/{values[-1]:.2f}"

    print(
        "label\tprompt\tsampling\tconc\tn\tfirst_reasoning_s\tfirst_answer_s"
        "\telapsed_s\tmax_gap_s\tmax_reasoning_chunk_chars"
    )
    for key in sorted(cells):
        rows = cells[key]
        print(
            "\t".join(map(str, key))
            + f"\t{len(rows)}"
            + "".join(
                "\t" + stats([r.get(k) for r in rows])
                for k in (
                    "first_reasoning_s",
                    "first_answer_s",
                    "elapsed_s",
                    "max_gap_s",
                    "max_reasoning_chunk_chars",
                )
            )
        )


def cmd_summarize(args):
    declared_for = {
        prompt: names.split(",")
        for prompt, names in (item.split("=", 1) for item in args.declared_for)
    }
    cells = {}
    for path in args.rows:
        for line in open(path):
            row = json.loads(line)
            # Re-derive the flags so rows from older harness revisions agree.
            declared = set(
                row.get("declared") or declared_for.get(row["prompt"]) or args.declared
            )
            if row["tools"] != "declared":
                declared = set()
            names = [c["name"] for c in row.get("tool_calls", [])]
            row.update(classify_names(names, declared))
            row.update(classify_leak(row))
            key = (
                row["label"],
                row["prompt"],
                "stream" if row["stream"] else "json",
                row["tools"],
                row["sampling"],
                row["concurrency"],
            )
            cell = cells.setdefault(
                key,
                {
                    k: 0
                    for k in (
                        "n",
                        "malformed",
                        "leak",
                        "defect",
                        "phantom",
                        "runaway",
                        "args",
                        "errors",
                    )
                },
            )
            cell["n"] += 1
            cell["malformed"] += bool(row.get("malformed"))
            cell["leak"] += bool(row.get("reasoning_leak"))
            cell["defect"] += bool(row.get("defect"))
            cell["phantom"] += bool(row.get("phantom_call"))
            cell["runaway"] += bool(row.get("runaway"))
            cell["args"] += bool(row.get("args_have_think_end"))
            cell["errors"] += row.get("status") != 200
    print(
        "label\tprompt\tmode\ttools\tsampling\tconc\tDEFECT\tmalformed_name"
        "\treasoning_leak\tnameless_call\t|model:runaway\targs_think_end\terrors"
    )
    for key in sorted(cells):
        c = cells[key]
        n = c["n"]
        cols = [
            c[k] for k in ("defect", "malformed", "leak", "phantom", "runaway", "args")
        ]
        print(
            "\t".join(map(str, key))
            + "".join(f"\t{v}/{n}" for v in cols)
            + f"\t{c['errors']}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    run.add_argument("--model", default="zai-org/GLM-5.3-Flash")
    run.add_argument("--prompt", required=True)
    run.add_argument("--prompt-name", required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--stream", type=int, choices=[0, 1], required=True)
    run.add_argument("--tools", choices=["declared", "none"], default="declared")
    run.add_argument("--sampling", choices=["greedy", "default"], required=True)
    run.add_argument("--seed-base", type=int, default=1000)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--n", type=int, default=1)
    run.add_argument("--max-tokens", type=int)
    run.add_argument("--timeout", type=float, default=1800)
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)

    replay = sub.add_parser("replay")
    replay.add_argument("--rows", required=True)
    replay.add_argument("--prompt", required=True)
    replay.add_argument("--prompt-name", required=True)
    replay.add_argument("--label")
    replay.add_argument("--tokenizer", required=True)
    replay.add_argument(
        "--force-reasoning", type=int, choices=[0, 1], help="default: auto"
    )
    replay.add_argument("--out", required=True)
    replay.set_defaults(func=cmd_replay)

    replay_summary = sub.add_parser("replay-summary")
    replay_summary.add_argument("rows", nargs="+")
    replay_summary.set_defaults(func=cmd_replay_summary)

    latency = sub.add_parser("latency")
    latency.add_argument("rows", nargs="+")
    latency.set_defaults(func=cmd_latency)

    summarize = sub.add_parser("summarize")
    summarize.add_argument("rows", nargs="+")
    summarize.add_argument(
        "--declared", nargs="+", default=["read", "bash", "edit", "write"]
    )
    summarize.add_argument(
        "--declared-for",
        action="append",
        default=[],
        help="PROMPT=name1,name2 for rows recorded without their declared tools",
    )
    summarize.set_defaults(func=cmd_summarize)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
