//! Tool schemas sent to the upstream model on every completion.
//!
//! These are the OpenAI-function-style declarations the LLM sees in
//! `CompletionRequest.tools`; the runtime dispatch happens in
//! [`crate::tool_runner`]. Keeping declarations and execution in
//! separate modules makes it easy to add a tool without touching the
//! runner, and vice versa.
//!
//! Stage 3 ships five: filesystem read / write / edit, directory
//! listing, and `bash`. Image generation, web fetch, MCP-derived
//! tools, etc. are out of scope here.

use serde_json::json;

use crate::provider::ToolSpec;

pub const READ_FILE: &str = "read_file";
pub const WRITE_FILE: &str = "write_file";
pub const EDIT_FILE: &str = "edit_file";
pub const LIST_DIR: &str = "list_dir";
pub const BASH: &str = "bash";

/// Build the static tool list passed to the model on every prompt.
/// Cheap — the JSON Schema fragments are constructed each call but
/// the bodies are small constants. If this ever shows up in a
/// profile we can `OnceLock` the Vec.
pub fn all_tools() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: READ_FILE.to_string(),
            description: "Read the contents of a text file. Returns the file's text.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file."
                    },
                    "line": {
                        "type": "integer",
                        "description": "Optional 1-based line number to start reading from.",
                        "minimum": 1
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional maximum number of lines to read.",
                        "minimum": 1
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: WRITE_FILE.to_string(),
            description: "Write text content to a file, replacing any existing contents. \
                Creates the file (and parent directories) if needed."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file."
                    },
                    "content": {
                        "type": "string",
                        "description": "Full new contents of the file."
                    }
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: EDIT_FILE.to_string(),
            description: "Replace one exact substring in a file with another. \
                Fails if `old_text` does not appear in the file, or appears more than once. \
                Use multiple edit_file calls for multiple edits."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file."
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text fragment to replace. Must be unique within the file."
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text."
                    }
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: LIST_DIR.to_string(),
            description:
                "List the entries of a directory. Returns names and a (f|d|l) kind per entry."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the directory."
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: BASH.to_string(),
            description: "Run a shell command via `sh -c`. \
                Returns combined stdout+stderr and the exit status. \
                The command runs in the session's working directory unless `cwd` is given."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command line, evaluated by `sh -c`."
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Optional absolute path to run the command from."
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        },
    ]
}

/// Try to infer which tool was intended from the shape of an
/// `arguments` object alone. Used by the agent when the model
/// emits a `<tool_call>` whose JSON has the right arguments but a
/// missing or invalid top-level `name` field — a recurring
/// Qwen3.6-27B failure mode.
///
/// Returns `Some(name)` only when the argument keys uniquely match
/// exactly one tool in the catalogue. Ambiguous shapes (`{path}`
/// alone could be either [`READ_FILE`] or [`LIST_DIR`]) return
/// `None` so the caller surfaces a Failed-card and lets the model
/// retry rather than guessing wrong.
///
/// Inference table (key set → tool):
///
/// | Keys                                  | Tool         |
/// |---------------------------------------|--------------|
/// | `{command}` or `{command, cwd}`       | `bash`       |
/// | `{path, content}`                     | `write_file` |
/// | `{path, old_text, new_text}`          | `edit_file`  |
/// | `{path}` / `{path, line}` / `{path, line, limit}` | *ambiguous* — None |
/// | (anything else)                       | None         |
pub fn infer_tool_name(arguments: &serde_json::Value) -> Option<&'static str> {
    let obj = arguments.as_object()?;
    let keys: std::collections::HashSet<&str> = obj.keys().map(|s| s.as_str()).collect();

    // `command` is unique to bash. Allow the optional `cwd` arg
    // alongside but nothing else (any unrecognised keys → bail and
    // let the model retry rather than misroute).
    if keys.contains("command") && keys.iter().all(|k| matches!(*k, "command" | "cwd")) {
        return Some(BASH);
    }
    // `content` is unique to write_file.
    if keys.contains("content") && keys.contains("path") && keys.len() == 2 {
        return Some(WRITE_FILE);
    }
    // `old_text` + `new_text` are unique to edit_file.
    if keys.contains("old_text")
        && keys.contains("new_text")
        && keys.contains("path")
        && keys.len() == 3
    {
        return Some(EDIT_FILE);
    }
    // `{path}` / `{path, line}` / `{path, line, limit}` overlap
    // between read_file (file contents) and list_dir (directory
    // contents). No safe inference — refuse.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_tools_has_five_named_entries() {
        let tools = all_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            vec![READ_FILE, WRITE_FILE, EDIT_FILE, LIST_DIR, BASH]
        );
    }

    #[test]
    fn infer_bash_from_command_only() {
        let args = serde_json::json!({"command": "ls /tmp"});
        assert_eq!(infer_tool_name(&args), Some(BASH));
    }

    #[test]
    fn infer_bash_from_command_and_cwd() {
        let args = serde_json::json!({"command": "ls", "cwd": "/tmp"});
        assert_eq!(infer_tool_name(&args), Some(BASH));
    }

    #[test]
    fn infer_bash_from_mkdir_like_real_failure() {
        // Lifted verbatim from the agent failure that motivated
        // this helper (helexa-acp.log @ 10:03:11).
        let args = serde_json::json!({
            "command": "mkdir -p /home/grenade/git/beat/beat/doc/plan/{01-discovery,02-segmentation,03-description,04-summary,05-output}"
        });
        assert_eq!(infer_tool_name(&args), Some(BASH));
    }

    #[test]
    fn infer_write_file() {
        let args = serde_json::json!({"path": "/tmp/x", "content": "hi"});
        assert_eq!(infer_tool_name(&args), Some(WRITE_FILE));
    }

    #[test]
    fn infer_edit_file() {
        let args = serde_json::json!({
            "path": "/tmp/x", "old_text": "a", "new_text": "b"
        });
        assert_eq!(infer_tool_name(&args), Some(EDIT_FILE));
    }

    #[test]
    fn refuse_ambiguous_path_only() {
        let args = serde_json::json!({"path": "/tmp/x"});
        assert_eq!(infer_tool_name(&args), None);
    }

    #[test]
    fn refuse_ambiguous_path_with_optionals() {
        // read_file accepts these optionals; list_dir doesn't —
        // but Qwen wouldn't reliably emit them either, so we
        // can't use their presence to disambiguate. Refuse.
        let args = serde_json::json!({"path": "/tmp/x", "line": 1, "limit": 50});
        assert_eq!(infer_tool_name(&args), None);
    }

    #[test]
    fn refuse_command_with_extra_unknown_keys() {
        // Defence in depth: an unrecognised key alongside
        // `command` means we don't really know what tool the
        // model wanted; refuse rather than guess.
        let args = serde_json::json!({"command": "ls", "extra": "?"});
        assert_eq!(infer_tool_name(&args), None);
    }

    #[test]
    fn refuse_empty_args() {
        let args = serde_json::json!({});
        assert_eq!(infer_tool_name(&args), None);
    }

    #[test]
    fn refuse_non_object_args() {
        let args = serde_json::json!("not an object");
        assert_eq!(infer_tool_name(&args), None);
    }

    #[test]
    fn every_tool_has_an_object_parameter_schema() {
        for tool in all_tools() {
            let ty = tool.parameters.get("type").and_then(|v| v.as_str());
            assert_eq!(
                ty,
                Some("object"),
                "tool {} parameters.type must be \"object\"",
                tool.name
            );
            assert!(
                tool.parameters.get("properties").is_some(),
                "tool {} missing properties",
                tool.name
            );
            assert!(
                tool.parameters.get("required").is_some(),
                "tool {} missing required list",
                tool.name
            );
        }
    }
}
