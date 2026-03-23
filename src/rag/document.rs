//! Generic document RAG — ingest, chunk, embed, store, retrieve.
//!
//! Supports markdown, plain text, HTML (via nanohtml2text), and PDF (with `rag-pdf` feature).
//! Stores chunks in the existing memory backend via `MemoryCategory::Custom("rag_document")`.

use crate::config::RagConfig;
use crate::memory::chunker;
use crate::memory::embeddings::{create_embedding_provider, EmbeddingProvider};
use crate::memory::{Memory, MemoryCategory};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// A document stored in the RAG index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub source: String, // file path or URL
    pub content_type: DocumentType,
    pub chunk_count: usize,
    pub created_at: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DocumentType {
    Markdown,
    Text,
    Html,
    Pdf,
}

impl std::fmt::Display for DocumentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Markdown => write!(f, "markdown"),
            Self::Text => write!(f, "text"),
            Self::Html => write!(f, "html"),
            Self::Pdf => write!(f, "pdf"),
        }
    }
}

/// A retrieved chunk with relevance score.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RetrievedChunk {
    pub document_id: String,
    pub document_title: String,
    pub chunk_index: usize,
    pub content: String,
    pub score: f64,
}

/// Request to ingest a document.
#[derive(Debug, serde::Deserialize)]
pub struct IngestRequest {
    pub title: String,
    pub content: Option<String>,
    pub url: Option<String>,
    pub file_path: Option<String>,
    pub content_type: Option<DocumentType>,
}

/// Request to query the RAG system.
#[derive(Debug, serde::Deserialize)]
pub struct QueryRequest {
    pub query: String,
    pub max_chunks: Option<usize>,
    pub min_relevance: Option<f64>,
}

/// Generic document RAG index.
pub struct DocumentRag {
    pub config: RagConfig,
    memory: Arc<dyn Memory>,
    embedding: Arc<dyn EmbeddingProvider>,
    documents: RwLock<HashMap<String, Document>>,
}

impl DocumentRag {
    /// Create a new DocumentRag instance.
    pub fn new(
        config: RagConfig,
        memory: Arc<dyn Memory>,
        embedding: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        Self {
            config,
            memory,
            embedding,
            documents: RwLock::new(HashMap::new()),
        }
    }

    /// Create embedding provider from memory search config.
    pub fn create_embedding(
        provider: &str,
        api_key: Option<&str>,
        model: &str,
        dims: usize,
    ) -> Arc<dyn EmbeddingProvider> {
        Arc::from(create_embedding_provider(provider, api_key, model, dims))
    }

    /// Ingest a document into the RAG index.
    pub async fn ingest(&self, req: &IngestRequest) -> Result<Document> {
        // Resolve content
        let (raw_content, doc_type) = self.resolve_content(req).await?;

        // Chunk the content and extract owned data.
        // Done in a separate sync function to ensure Rc<str> is dropped before any .await.
        let chunk_data = Self::chunk_and_own(&raw_content, self.config.chunk_max_tokens)?;

        // Create document record
        let doc_id = Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let document = Document {
            id: doc_id.clone(),
            title: req.title.clone(),
            source: req
                .file_path
                .clone()
                .or_else(|| req.url.clone())
                .unwrap_or_else(|| "inline".into()),
            content_type: doc_type.clone(),
            chunk_count: chunk_data.len(),
            created_at: now.clone(),
        };

        // Store each chunk in memory backend
        for (i, (_idx, content, heading)) in chunk_data.iter().enumerate() {
            let key = format!("rag:{}:{}", doc_id, i);
            let metadata = serde_json::json!({
                "document_id": doc_id,
                "document_title": document.title,
                "chunk_index": i,
                "content_type": doc_type.to_string(),
                "source": document.source,
                "heading": heading,
            });
            let stored_content = format!(
                "{}\n\n---\n\n{}",
                serde_json::to_string(&metadata)?,
                content
            );

            self.memory
                .store(&key, &stored_content, MemoryCategory::Custom("rag_document".into()), None)
                .await
                .context(format!("Failed to store chunk {i}"))?;
        }

        // Track document
        {
            let mut docs = self.documents.write().await;
            docs.insert(doc_id.clone(), document.clone());
        }

        Ok(document)
    }

    /// Query the RAG index for relevant chunks.
    pub async fn query(&self, query: &str, max_chunks: usize, min_relevance: f64) -> Result<Vec<RetrievedChunk>> {
        // Use the memory backend's recall which does hybrid search internally
        let entries = self
            .memory
            .recall(query, max_chunks * 2, None, None, None)
            .await
            .context("Failed to query memory backend")?;

        let mut results = Vec::new();

        for entry in entries {
            // Filter by category
            if !matches!(entry.category, MemoryCategory::Custom(ref c) if c == "rag_document") {
                continue;
            }

            let score = entry.score.unwrap_or(0.0);
            if score < min_relevance {
                continue;
            }

            // Parse metadata from content
            let (metadata, content) = Self::parse_chunk_content(&entry.content);

            let chunk = RetrievedChunk {
                document_id: metadata.get("document_id").cloned().unwrap_or_default(),
                document_title: metadata.get("document_title").cloned().unwrap_or_default(),
                chunk_index: metadata
                    .get("chunk_index")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(0),
                content,
                score,
            };

            results.push(chunk);
            if results.len() >= max_chunks {
                break;
            }
        }

        Ok(results)
    }

    /// Delete a document and all its chunks.
    pub async fn delete_document(&self, document_id: &str) -> Result<bool> {
        let removed = {
            let mut docs = self.documents.write().await;
            docs.remove(document_id).is_some()
        };

        if removed {
            // List all rag_document memories and delete those belonging to this document
            let entries = self
                .memory
                .list(Some(&MemoryCategory::Custom("rag_document".into())), None)
                .await?;

            for entry in entries {
                if entry.key.contains(&format!("rag:{}:", document_id)) {
                    self.memory.forget(&entry.key).await?;
                }
            }
        }

        Ok(removed)
    }

    /// List all indexed documents.
    pub async fn list_documents(&self) -> Vec<Document> {
        let docs = self.documents.read().await;
        docs.values().cloned().collect()
    }

    /// Build a context string from retrieved chunks for prompt injection.
    pub fn build_context(chunks: &[RetrievedChunk]) -> String {
        if chunks.is_empty() {
            return String::new();
        }

        let mut ctx = String::from("[RAG Context]\n");
        for chunk in chunks {
            ctx.push_str(&format!(
                "--- Document: {} (chunk {}, score: {:.2}) ---\n{}\n\n",
                chunk.document_title,
                chunk.chunk_index,
                chunk.score,
                chunk.content
            ));
        }
        ctx.push_str("[/RAG Context]\n");
        ctx
    }

    // --- Private helpers ---

    /// Chunk markdown and convert to owned data (no Rc<str>).
    fn chunk_and_own(text: &str, max_tokens: usize) -> Result<Vec<(usize, String, Option<String>)>> {
        let chunks = chunker::chunk_markdown(text, max_tokens);
        if chunks.is_empty() {
            anyhow::bail!("Document produced no chunks after splitting");
        }
        Ok(chunks
            .into_iter()
            .map(|c| {
                (
                    c.index,
                    c.content,
                    c.heading.map(|h| h.to_string()),
                )
            })
            .collect())
    }

    async fn resolve_content(&self, req: &IngestRequest) -> Result<(String, DocumentType)> {
        // Inline content takes priority
        if let Some(ref content) = req.content {
            let doc_type = req
                .content_type
                .as_ref()
                .cloned()
                .unwrap_or(DocumentType::Markdown);
            return Ok((content.clone(), doc_type));
        }

        // URL fetch
        if let Some(ref url) = req.url {
            let html = reqwest::get(url)
                .await
                .context(format!("Failed to fetch URL: {url}"))?
                .text()
                .await
                .context("Failed to read URL response body")?;
            let text = nanohtml2text::html2text(&html);
            let doc_type = req
                .content_type
                .as_ref()
                .cloned()
                .unwrap_or(DocumentType::Html);
            return Ok((text, doc_type));
        }

        // File path
        if let Some(ref path) = req.file_path {
            let path = Path::new(path);
            let raw = tokio::fs::read_to_string(path)
                .await
                .context(format!("Failed to read file: {}", path.display()))?;

            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            let doc_type = match ext.as_str() {
                "md" | "markdown" => DocumentType::Markdown,
                "html" | "htm" => {
                    let text = nanohtml2text::html2text(&raw);
                    return Ok((text, req.content_type.clone().unwrap_or(DocumentType::Html)));
                }
                "pdf" => {
                    #[cfg(feature = "rag-pdf")]
                    {
                        let text = pdf_extract::extract_text_from_path(path)
                            .context("Failed to extract PDF text")?;
                        return Ok((text, DocumentType::Pdf));
                    }
                    #[cfg(not(feature = "rag-pdf"))]
                    {
                        anyhow::bail!("PDF support requires the `rag-pdf` feature flag");
                    }
                }
                "txt" | _ => DocumentType::Text,
            };

            Ok((raw, doc_type))
        } else {
            anyhow::bail!("IngestRequest must specify content, url, or file_path");
        }
    }

    /// Parse stored chunk content back into (metadata, actual_content).
    fn parse_chunk_content(stored: &str) -> (HashMap<String, String>, String) {
        // Content format: "{json_metadata}\n\n---\n\n{actual_content}"
        if let Some(sep_pos) = stored.find("\n\n---\n\n") {
            let meta_str = &stored[..sep_pos];
            let content = &stored[sep_pos + 7..]; // skip "\n\n---\n\n"
            if let Ok(meta) = serde_json::from_str::<HashMap<String, serde_json::Value>>(meta_str) {
                let map: HashMap<String, String> = meta
                    .into_iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k, s.to_string())))
                    .collect();
                return (map, content.to_string());
            }
        }
        (HashMap::new(), stored.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_chunk_content_valid() {
        let stored = r#"{"document_id":"abc","document_title":"Test","chunk_index":0}

---

This is the actual content."#;
        let (meta, content) = DocumentRag::parse_chunk_content(stored);
        assert_eq!(meta.get("document_id").unwrap(), "abc");
        assert_eq!(meta.get("document_title").unwrap(), "Test");
        assert_eq!(content, "This is the actual content.");
    }

    #[test]
    fn test_parse_chunk_content_no_separator() {
        let stored = "Just plain content with no metadata";
        let (meta, content) = DocumentRag::parse_chunk_content(stored);
        assert!(meta.is_empty());
        assert_eq!(content, stored);
    }

    #[test]
    fn test_build_context_empty() {
        assert!(DocumentRag::build_context(&[]).is_empty());
    }

    #[test]
    fn test_build_context_with_chunks() {
        let chunks = vec![RetrievedChunk {
            document_id: "doc1".into(),
            document_title: "Test Doc".into(),
            chunk_index: 0,
            content: "Hello world".into(),
            score: 0.85,
        }];
        let ctx = DocumentRag::build_context(&chunks);
        assert!(ctx.contains("[RAG Context]"));
        assert!(ctx.contains("Test Doc"));
        assert!(ctx.contains("Hello world"));
        assert!(ctx.contains("[/RAG Context]"));
    }
}
