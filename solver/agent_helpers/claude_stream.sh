#!/usr/bin/env bash
set -euo pipefail

# Purpose: turn Claude Code stream-json output into readable review text.

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

  def render_tool_call:
    .name as $name
    | (.input // {}) as $input
    | if $name == "Read" then
        append_detail("[tool] Read \($input.file_path // "<unknown file>")"; ($input | fields(["offset", "limit"])))
      elif $name == "Grep" then
        append_detail("[tool] Grep"; ($input | fields(["pattern", "path", "include"])))
      elif $name == "Glob" then
        append_detail("[tool] Glob"; ($input | fields(["pattern", "path"])))
      elif $name == "Bash" then
        append_detail("[tool] Bash \(($input.command // "<unknown command>") | trunc(800))"; ($input | fields(["description", "timeout"])))
      elif $name == "LS" then
        append_detail("[tool] LS \($input.path // "<unknown path>")"; ($input | fields(["ignore"])))
      elif $name == "Edit" or $name == "Write" or $name == "MultiEdit" then
        append_detail("[tool] \($name) \($input.file_path // $input.path // "<unknown file>")"; ($input | fields(["replace_all"])))
      else
        append_detail("[tool] \($name)"; ($input | fields(keys_unsorted)))
      end;

  def content_items:
    if (.message.content? | type) == "array" then
      .message.content[]
    elif (.message.content? | type) == "string" then
      { type: "text", text: .message.content }
    else
      empty
    end;

  def render_content:
    if .type == "text" then
      .text
    elif .type == "tool_use" then
      render_tool_call
    elif .type == "tool_result" then
      empty
    elif .type == "thinking" or .type == "redacted_thinking" then
      empty
    elif .type == "image" then
      "[image]"
    else
      empty
    end;

  def render_event:
    if .type == "system" and .subtype == "init" then
      "[claude] session=\(.session_id // "unknown") model=\(.model // "unknown") cwd=\(.cwd // "unknown")"
    elif .type == "assistant" then
      content_items | render_content
    elif .type == "user" then
      content_items | render_content
    elif .type == "result" then
      "[claude] done subtype=\(.subtype // "unknown") turns=\(.num_turns // "?") duration_ms=\(.duration_ms // "?") cost_usd=\(.total_cost_usd // "?")"
    elif .type == "error" then
      "[claude:error] \(.message // .error // tostring)"
    else
      empty
    end;

  . as $line
  | try ($line | fromjson | render_event) catch ("[unparsed] \($line | trunc(1200))")
  | stamp
'
