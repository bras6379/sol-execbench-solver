#!/usr/bin/env bash
set -euo pipefail

# Purpose: turn Codex CLI --json output into readable review text.

jq --unbuffered -Rr '
  def timestamp:
    now | strflocaltime("%H:%M:%S");

  def stamp:
    tostring
    | gsub("\r"; "")
    | select(length > 0)
    | "[\(timestamp)] \(.)";

  def trunc($n):
    tostring as $s
    | if ($s | length) > $n then ($s[0:$n] + "...") else $s end;

  def value_summary:
    (if type == "string" then
      .
    elif type == "number" or type == "boolean" then
      tostring
    elif type == "null" then
      "null"
    elif type == "array" then
      "[\(length) items]"
    elif type == "object" then
      "{\(keys | length) keys}"
    else
      tostring
    end) | trunc(180);

  def fields($keys):
    [ $keys[] as $key
      | if .[$key]? == null then
          empty
        else
          "\($key)=\(.[$key] | value_summary)"
        end
    ] | join(" ");

  def append_detail($prefix; $detail):
    if $detail == "" then $prefix else "\($prefix) \($detail)" end;

  def strip_shell_wrapper:
    tostring
    | if test("^/bin/(zsh|bash) -lc ") then
        sub("^/bin/(zsh|bash) -lc "; "")
      else
        .
      end
    | sub("^[\"'\'' ]+"; "")
    | sub("[\"'\'' ]+$"; "");

  def item_type:
    (.type // .kind // "unknown");

  def render_tool_call:
    item_type as $t
    | if $t == "command_execution" then
        "[tool] Bash \((.command // "<unknown command>") | strip_shell_wrapper | trunc(800))"
      elif .name? then
        append_detail("[tool] \(.name)"; ((.input // .arguments // {}) | fields(keys_unsorted)))
      else
        append_detail("[tool] \($t)"; (fields(["command", "path", "pattern", "name", "status"])))
      end;

  def render_item_started:
    item_type as $t
    | if ($t | test("tool|call|command|exec")) then
        render_tool_call
      else
        empty
      end;

  def render_item_completed:
    item_type as $t
    | if $t == "agent_message" then
        (.text // .message // empty)
      elif ($t | test("reasoning|thinking")) then
        empty
      elif ($t | test("tool|call|command|exec")) then
        if ((.exit_code? // 0) != 0) then
          "[tool:error] Bash exit=\(.exit_code) \((.command // "<unknown command>") | strip_shell_wrapper | trunc(800))"
        else
          empty
        end
      elif (.text? | type) == "string" then
        .text
      else
        empty
      end;

  def render_event:
    if .type == "thread.started" then
      "[codex] thread=\(.thread_id // "unknown")"
    elif .type == "turn.started" then
      empty
    elif .type == "turn.completed" then
      "[codex] done input_tokens=\(.usage.input_tokens // "?") cached_input_tokens=\(.usage.cached_input_tokens // "?") output_tokens=\(.usage.output_tokens // "?") reasoning_output_tokens=\(.usage.reasoning_output_tokens // "?")"
    elif .type == "item.started" then
      (.item // .) | render_item_started
    elif .type == "item.completed" then
      (.item // .) | render_item_completed
    elif (.type // "" | test("error")) then
      "[codex:error] \(.message // .error // tostring)"
    elif (.message? | type) == "string" then
      .message
    else
      empty
    end;

  . as $line
  | try ($line | fromjson | render_event) catch ("[unparsed] \($line | trunc(1200))")
  | stamp
'
