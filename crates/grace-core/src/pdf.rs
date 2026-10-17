//! Pure-Rust PDF document extraction and semantic RAG chunking.
//!
//! Ported from `src/grace/tools/dispatcher.py` (`_read_pdf`, `_summarize_pdf`) and
//! `src/grace/rag/retriever.py` (`DocumentChunker`, `LocalRagIndex`).
//!
//! Enforces:
//! - Safe roots check (`is_within_safe_pdf_roots`) requiring confirmation outside safe directories (R26)
//! - 50 MB file size cap (R14)
//! - Max 50 pages and 50,000 characters
//! - Encrypted PDF detection
//! - Overlapping semantic chunking (chunk_size=400, overlap=80)
//! - TF-IDF / term-matching query extraction and summary chunk distribution
//! - Output defanging and wrapping via `wrap_untrusted` (R18)

use crate::planner::wrap_untrusted;
use crate::safety::is_within_safe_pdf_roots;
use crate::sentence_split::split_sentences;
use regex::Regex;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::LazyLock;

pub const MAX_PDF_BYTES: u64 = 50 * 1024 * 1024; // 50MB
pub const MAX_PDF_PAGES: usize = 50;
pub const MAX_PDF_CHARS: usize = 50000;

static SECTION_TAG_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[(Section|Excerpt|Page)\s*\d+\]:?").unwrap()
});

static HEADING_MD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"#{1,6}\s*").unwrap()
});

static BOLD_MD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\*\*([^*]+)\*\*").unwrap()
});

static ITALIC_MD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\*([^*]+)\*").unwrap()
});

static CODE_MD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"`([^`]+)`").unwrap()
});

static BULLET_MD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^\s*[-*•]\s*").unwrap()
});

/// Splits raw document text into overlapping chunks of approximately `chunk_size` chars.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let cleaned = text.trim();
    if cleaned.is_empty() {
        return Vec::new();
    }
    if cleaned.len() <= chunk_size {
        return vec![cleaned.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    let text_len = cleaned.len();

    while start < text_len {
        let mut end = (start + chunk_size).min(text_len);
        if end < text_len {
            let search_start = start + chunk_size / 2;
            if search_start < end {
                if let Some(pos) = cleaned[search_start..end].rfind(' ') {
                    end = search_start + pos;
                }
            }
        }
        let chunk = cleaned[start..end].trim();
        if !chunk.is_empty() {
            chunks.push(chunk.to_string());
        }
        if end >= text_len {
            break;
        }
        start = (start + 1).max(end.saturating_sub(overlap));
    }
    chunks
}

/// Selects `top_k` representative chunks evenly distributed across the document.
pub fn get_summary_chunks(chunks: &[String], top_k: usize) -> Vec<String> {
    if chunks.is_empty() {
        return Vec::new();
    }
    if chunks.len() <= top_k {
        return chunks.to_vec();
    }
    let step = chunks.len() as f64 / top_k as f64;
    (0..top_k)
        .map(|i| chunks[(i as f64 * step) as usize].clone())
        .collect()
}

/// Ranks chunks against a search query using term occurrence overlap.
pub fn query_chunks(chunks: &[String], query: &str, top_k: usize) -> Vec<(usize, String)> {
    if chunks.is_empty() || query.trim().is_empty() {
        return Vec::new();
    }
    let query_terms: Vec<String> = query
        .to_lowercase()
        .split_whitespace()
        .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let mut scored: Vec<(usize, usize, &String)> = chunks
        .iter()
        .enumerate()
        .map(|(idx, chunk)| {
            let lower = chunk.to_lowercase();
            let mut score = 0;
            for term in &query_terms {
                if lower.contains(term) {
                    score += 1;
                }
            }
            (score, idx, chunk)
        })
        .filter(|(score, _, _)| *score > 0)
        .collect();

    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored
        .into_iter()
        .take(top_k)
        .map(|(_, idx, chunk)| (idx, chunk.clone()))
        .collect()
}

/// Cleans markdown and section tags to produce a concise 2-4 sentence summary suitable for speech.
pub fn clean_summary_for_speech(summary_context: &str) -> String {
    let mut cleaned = SECTION_TAG_RE.replace_all(summary_context, "").to_string();
    cleaned = HEADING_MD_RE.replace_all(&cleaned, "").to_string();
    cleaned = BOLD_MD_RE.replace_all(&cleaned, "$1").to_string();
    cleaned = ITALIC_MD_RE.replace_all(&cleaned, "$1").to_string();
    cleaned = CODE_MD_RE.replace_all(&cleaned, "$1").to_string();
    cleaned = BULLET_MD_RE.replace_all(&cleaned, "").to_string();
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    let normalized = words.join(" ");

    let sentences = split_sentences(&normalized);
    if !sentences.is_empty() {
        sentences.iter().take(4).cloned().collect::<Vec<_>>().join(" ")
    } else {
        "The document does not contain readable summary text.".to_string()
    }
}

/// Extracts text from a physical PDF file using pure-Rust `lopdf`.
pub fn extract_pdf_text_from_file(path: &Path) -> Result<String, (String, String)> {
    let doc = lopdf::Document::load(path)
        .map_err(|e| ("Error reading PDF".to_string(), format!("Error reading PDF: {e}")))?;

    if doc.is_encrypted() {
        return Err((
            "PDF is password-protected or encrypted".to_string(),
            "That PDF is password-protected, so I can't read it.".to_string(),
        ));
    }

    let mut text = String::new();
    let pages = doc.get_pages();
    for (page_num, _) in pages.into_iter().take(MAX_PDF_PAGES) {
        if let Ok(page_text) = doc.extract_text(&[page_num]) {
            if !page_text.trim().is_empty() {
                text.push_str(&page_text);
                text.push_str("\n\n");
            }
        }
        if text.len() >= MAX_PDF_CHARS {
            text.truncate(MAX_PDF_CHARS);
            break;
        }
    }

    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err((
            "No extractable text in PDF".to_string(),
            "The PDF doesn't contain extractable text.".to_string(),
        ));
    }

    Ok(trimmed.to_string())
}

/// Reads a local PDF document, extracts relevant excerpts, and returns Grace's standard tool result envelope.
pub fn read_pdf_document(path_str: &str, query: Option<&str>, confirmed: bool) -> Value {
    if path_str.trim().is_empty() {
        return json!({"status": "error", "error": "Missing 'path' parameter"});
    }

    if !is_within_safe_pdf_roots(path_str) && !confirmed {
        let filename = Path::new(path_str).file_name().and_then(|n| n.to_str()).unwrap_or(path_str);
        return json!({
            "status": "error",
            "error": format!("Refusing to read '{path_str}' without confirmation: path is outside Documents, Desktop, and Downloads."),
            "text": format!("Reading {filename} from outside your normal documents folders requires confirmation.")
        });
    }

    let path = Path::new(path_str);
    if !path.is_file() {
        return json!({
            "status": "error",
            "error": format!("File not found: {path_str}"),
            "text": "I couldn't find that PDF."
        });
    }

    match std::fs::metadata(path) {
        Ok(meta) if meta.len() > MAX_PDF_BYTES => {
            return json!({
                "status": "error",
                "error": format!("PDF file size ({} bytes) exceeds the 50MB limit.", meta.len()),
                "text": "That PDF is too large for me to process."
            });
        }
        Err(e) => {
            return json!({
                "status": "error",
                "error": format!("Error accessing PDF: {e}"),
                "text": format!("Error accessing PDF: {e}")
            });
        }
        _ => {}
    }

    let text = match extract_pdf_text_from_file(path) {
        Ok(t) => t,
        Err((err, user_msg)) => {
            return json!({
                "status": "error",
                "error": err,
                "text": user_msg
            });
        }
    };

    let chunks = chunk_text(&text, 400, 80);
    let rag_text = if let Some(q) = query.filter(|q| !q.trim().is_empty()) {
        let matches = query_chunks(&chunks, q, 3);
        if !matches.is_empty() {
            matches
                .into_iter()
                .map(|(idx, c)| format!("[Excerpt {}]: {c}", idx + 1))
                .collect::<Vec<_>>()
                .join("\n\n")
        } else {
            get_summary_chunks(&chunks, 3)
                .into_iter()
                .enumerate()
                .map(|(i, c)| format!("[Excerpt {}]: {c}", i + 1))
                .collect::<Vec<_>>()
                .join("\n\n")
        }
    } else {
        get_summary_chunks(&chunks, 3)
            .into_iter()
            .enumerate()
            .map(|(i, c)| format!("[Excerpt {}]: {c}", i + 1))
            .collect::<Vec<_>>()
            .join("\n\n")
    };

    let result_text = if !rag_text.is_empty() {
        rag_text
    } else {
        text.chars().take(2000).collect()
    };

    json!({
        "status": "ok",
        "text": wrap_untrusted(&result_text),
        "action": "read_pdf"
    })
}

/// Summarizes a local PDF document across sections and returns Grace's standard tool result envelope.
pub fn summarize_pdf_document(path_str: &str, confirmed: bool) -> Value {
    if path_str.trim().is_empty() {
        return json!({"status": "error", "error": "Missing 'path' parameter"});
    }

    if !is_within_safe_pdf_roots(path_str) && !confirmed {
        let filename = Path::new(path_str).file_name().and_then(|n| n.to_str()).unwrap_or(path_str);
        return json!({
            "status": "error",
            "error": format!("Refusing to summarize '{path_str}' without confirmation: path is outside Documents, Desktop, and Downloads."),
            "text": format!("Summarizing {filename} from outside your normal documents folders requires confirmation.")
        });
    }

    let path = Path::new(path_str);
    if !path.is_file() {
        return json!({
            "status": "error",
            "error": format!("File not found: {path_str}"),
            "text": "I couldn't find that PDF."
        });
    }

    match std::fs::metadata(path) {
        Ok(meta) if meta.len() > MAX_PDF_BYTES => {
            return json!({
                "status": "error",
                "error": format!("PDF file size ({} bytes) exceeds the 50MB limit.", meta.len()),
                "text": "That PDF is too large for me to process."
            });
        }
        Err(e) => {
            return json!({
                "status": "error",
                "error": format!("Error accessing PDF: {e}"),
                "text": format!("Error accessing PDF: {e}")
            });
        }
        _ => {}
    }

    let text = match extract_pdf_text_from_file(path) {
        Ok(t) => t,
        Err((err, user_msg)) => {
            return json!({
                "status": "error",
                "error": err,
                "text": user_msg
            });
        }
    };

    let chunks = chunk_text(&text, 400, 80);
    let summary_chunks = get_summary_chunks(&chunks, 4);
    let summary_context = summary_chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| format!("[Section {}]: {c}", i + 1))
        .collect::<Vec<_>>()
        .join("\n\n");

    let speakable = clean_summary_for_speech(&summary_context);

    json!({
        "status": "ok",
        "text": wrap_untrusted(&summary_context),
        "speakable_summary": speakable,
        "action": "summarize_pdf"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_text_splits_into_overlapping_chunks() {
        let sample = "The quick brown fox jumps over the lazy dog. ".repeat(20);
        let chunks = chunk_text(&sample, 200, 40);
        assert!(chunks.len() >= 3);
        assert!(chunks[0].len() <= 200);
    }

    #[test]
    fn get_summary_chunks_distributes_across_list() {
        let chunks: Vec<String> = (0..10).map(|i| format!("Chunk {i}")).collect();
        let summary = get_summary_chunks(&chunks, 3);
        assert_eq!(summary.len(), 3);
        assert_eq!(summary[0], "Chunk 0");
        assert_eq!(summary[1], "Chunk 3");
        assert_eq!(summary[2], "Chunk 6");
    }

    #[test]
    fn query_chunks_ranks_matching_terms() {
        let chunks = vec![
            "Apples and oranges are fruit.".to_string(),
            "Financial earnings report for third quarter.".to_string(),
            "Bananas and pineapples in the market.".to_string(),
        ];
        let matches = query_chunks(&chunks, "quarter earnings", 2);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].0, 1);
        assert!(matches[0].1.contains("earnings"));
    }

    #[test]
    fn clean_summary_for_speech_removes_section_tags_and_markdown() {
        let raw = "[Section 1]: ## Overview\n\nThis is a **crucial** update. Everything is fine.\n\n[Section 2]: More text here.";
        let speakable = clean_summary_for_speech(raw);
        assert!(!speakable.contains("[Section"));
        assert!(!speakable.contains("##"));
        assert!(!speakable.contains("**"));
        assert!(speakable.contains("crucial update"));
    }

    #[test]
    fn read_pdf_refuses_unsafe_path_without_confirmation() {
        let res = read_pdf_document("C:/Windows/System32/drivers/etc/malicious.pdf", None, false);
        assert_eq!(res["status"], "error");
        assert!(res["error"].as_str().unwrap().contains("outside Documents"));
    }

    #[test]
    fn read_pdf_refuses_missing_path() {
        let res = read_pdf_document("", None, true);
        assert_eq!(res["status"], "error");
        assert_eq!(res["error"], "Missing 'path' parameter");
    }
}
