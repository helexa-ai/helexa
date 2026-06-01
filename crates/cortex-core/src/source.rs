//! Scheme-qualified model identifiers.
//!
//! cortex/neuron historically resolves every model id through hf-hub
//! against `https://huggingface.co`. Helexa is adding an EU-hosted
//! registry (`registry.helexa.ai`) alongside HF — both speak the same
//! HF-compatible wire format, but the bytes, jurisdiction, and trust
//! root differ. Model ids therefore need a scheme:
//!
//!   - `huggingface:Qwen/Qwen3.6-27B`         — HF-hosted bytes
//!   - `helexa:Qwen/Qwen3.6-27B-Uncensored`  — helexa registry bytes
//!   - `helexa:SomeOperator/CustomFinetune`  — operator publishing
//!     under the helexa namespace; same scheme handles all `org/name`
//!     pairs hosted in that registry.
//!
//! Bare `org/name` parses with an empty scheme; the caller (typically
//! a harness) substitutes its configured default scheme so existing
//! configs keep working through the transition.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Parsed `scheme:org/name`. Bare `org/name` produces an empty scheme
/// — call `with_default_scheme` (or check `is_scheme_unset`) to
/// resolve before using.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelSourceId {
    pub scheme: String,
    pub org: String,
    pub name: String,
}

/// Errors from `ModelSourceId::from_str`. Carries the offending input
/// so log lines / API errors can echo what the operator typed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    #[error("empty model id")]
    Empty,
    #[error("model id '{0}' is missing the '/' between org and name")]
    MissingSlash(String),
    #[error("model id '{0}' has an empty scheme before ':'")]
    EmptyScheme(String),
    #[error("model id '{0}' has an empty org")]
    EmptyOrg(String),
    #[error("model id '{0}' has an empty name")]
    EmptyName(String),
    #[error("model id '{0}' has a scheme containing '/' which is reserved for org/name")]
    SchemeContainsSlash(String),
    #[error("model id '{0}' has a name containing ':' which is reserved for the scheme prefix")]
    NameContainsColon(String),
}

impl ModelSourceId {
    /// Construct directly from already-validated parts. Used by tests
    /// and call sites that have the fields separately; the public API
    /// for parsing user input is `FromStr`.
    pub fn new(scheme: impl Into<String>, org: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            scheme: scheme.into(),
            org: org.into(),
            name: name.into(),
        }
    }

    /// True when this id parsed from a bare `org/name` (no scheme
    /// prefix). The harness substitutes its configured default in
    /// `with_default_scheme` before resolving against a registry.
    pub fn is_scheme_unset(&self) -> bool {
        self.scheme.is_empty()
    }

    /// Substitute `default` for an empty scheme. No-op when the scheme
    /// is already set. Returns self by value so it composes neatly:
    /// `id.parse::<ModelSourceId>()?.with_default_scheme("huggingface")`.
    pub fn with_default_scheme(mut self, default: &str) -> Self {
        if self.scheme.is_empty() {
            self.scheme = default.to_string();
        }
        self
    }

    /// The `org/name` half — what an hf-hub `Api::model(...)` call
    /// expects regardless of which scheme/endpoint we're hitting.
    pub fn repo_path(&self) -> String {
        format!("{}/{}", self.org, self.name)
    }
}

impl fmt::Display for ModelSourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.scheme.is_empty() {
            write!(f, "{}/{}", self.org, self.name)
        } else {
            write!(f, "{}:{}/{}", self.scheme, self.org, self.name)
        }
    }
}

impl FromStr for ModelSourceId {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err(ParseError::Empty);
        }
        // Scheme split. Only the *first* colon counts — anything after
        // belongs to org/name (and would be rejected separately because
        // `:` isn't allowed there).
        let (scheme, rest) = match s.split_once(':') {
            Some((scheme, rest)) => {
                if scheme.is_empty() {
                    return Err(ParseError::EmptyScheme(s.to_string()));
                }
                if scheme.contains('/') {
                    return Err(ParseError::SchemeContainsSlash(s.to_string()));
                }
                (scheme.to_string(), rest)
            }
            None => (String::new(), s),
        };
        let (org, name) = rest
            .split_once('/')
            .ok_or_else(|| ParseError::MissingSlash(s.to_string()))?;
        if org.is_empty() {
            return Err(ParseError::EmptyOrg(s.to_string()));
        }
        if name.is_empty() {
            return Err(ParseError::EmptyName(s.to_string()));
        }
        if name.contains(':') {
            return Err(ParseError::NameContainsColon(s.to_string()));
        }
        Ok(Self {
            scheme,
            org: org.to_string(),
            name: name.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qualified() {
        let id: ModelSourceId = "huggingface:Qwen/Qwen3.6-27B".parse().unwrap();
        assert_eq!(id.scheme, "huggingface");
        assert_eq!(id.org, "Qwen");
        assert_eq!(id.name, "Qwen3.6-27B");
        assert_eq!(id.repo_path(), "Qwen/Qwen3.6-27B");
        assert!(!id.is_scheme_unset());
    }

    #[test]
    fn parses_helexa_scheme() {
        let id: ModelSourceId = "helexa:SomeOperator/Qwen3.6-27B-Uncensored"
            .parse()
            .unwrap();
        assert_eq!(id.scheme, "helexa");
        assert_eq!(id.org, "SomeOperator");
        assert_eq!(id.name, "Qwen3.6-27B-Uncensored");
    }

    #[test]
    fn parses_bare_id_with_empty_scheme() {
        let id: ModelSourceId = "Qwen/Qwen3-30B-A3B-Instruct".parse().unwrap();
        assert_eq!(id.scheme, "");
        assert_eq!(id.org, "Qwen");
        assert_eq!(id.name, "Qwen3-30B-A3B-Instruct");
        assert!(id.is_scheme_unset());
    }

    #[test]
    fn substitutes_default_scheme_only_when_unset() {
        let id: ModelSourceId = "Qwen/Q3".parse().unwrap();
        assert_eq!(id.with_default_scheme("huggingface").scheme, "huggingface");

        let id: ModelSourceId = "helexa:Qwen/Q3".parse().unwrap();
        assert_eq!(
            id.with_default_scheme("huggingface").scheme,
            "helexa",
            "default substitution must not override an explicit scheme"
        );
    }

    #[test]
    fn display_roundtrips_qualified_id() {
        let s = "helexa:Helexa/Qwen3.6-27B";
        let id: ModelSourceId = s.parse().unwrap();
        assert_eq!(id.to_string(), s);
    }

    #[test]
    fn display_roundtrips_bare_id() {
        let s = "Qwen/Q3";
        let id: ModelSourceId = s.parse().unwrap();
        assert_eq!(id.to_string(), s);
    }

    #[test]
    fn rejects_empty() {
        assert_eq!("".parse::<ModelSourceId>().unwrap_err(), ParseError::Empty);
    }

    #[test]
    fn rejects_missing_slash() {
        match "Qwen".parse::<ModelSourceId>().unwrap_err() {
            ParseError::MissingSlash(s) => assert_eq!(s, "Qwen"),
            other => panic!("expected MissingSlash, got {other:?}"),
        }
        match "huggingface:Qwen".parse::<ModelSourceId>().unwrap_err() {
            ParseError::MissingSlash(s) => assert_eq!(s, "huggingface:Qwen"),
            other => panic!("expected MissingSlash, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_scheme() {
        match ":Qwen/Q3".parse::<ModelSourceId>().unwrap_err() {
            ParseError::EmptyScheme(s) => assert_eq!(s, ":Qwen/Q3"),
            other => panic!("expected EmptyScheme, got {other:?}"),
        }
    }

    #[test]
    fn rejects_scheme_with_slash() {
        match "hugg/ingface:Q/N".parse::<ModelSourceId>().unwrap_err() {
            ParseError::SchemeContainsSlash(s) => assert_eq!(s, "hugg/ingface:Q/N"),
            other => panic!("expected SchemeContainsSlash, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_org_or_name() {
        match "huggingface:/N".parse::<ModelSourceId>().unwrap_err() {
            ParseError::EmptyOrg(_) => {}
            other => panic!("expected EmptyOrg, got {other:?}"),
        }
        match "huggingface:Q/".parse::<ModelSourceId>().unwrap_err() {
            ParseError::EmptyName(_) => {}
            other => panic!("expected EmptyName, got {other:?}"),
        }
    }

    #[test]
    fn rejects_name_with_colon() {
        match "huggingface:Q/N:weird"
            .parse::<ModelSourceId>()
            .unwrap_err()
        {
            ParseError::NameContainsColon(s) => assert_eq!(s, "huggingface:Q/N:weird"),
            other => panic!("expected NameContainsColon, got {other:?}"),
        }
    }

    #[test]
    fn serde_roundtrips_via_struct() {
        // We serialize as a struct (scheme/org/name fields) so the
        // shape is self-describing in API payloads. Callers that want
        // the compact `scheme:org/name` string use `Display`/`FromStr`.
        let id = ModelSourceId::new("helexa", "Helexa", "Qwen3.6-27B");
        let json = serde_json::to_string(&id).unwrap();
        let back: ModelSourceId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }
}
