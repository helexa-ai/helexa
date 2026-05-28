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
