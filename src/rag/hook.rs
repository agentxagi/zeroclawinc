//! RAG hook — injects document context into prompts via `before_prompt_build`.

use crate::hooks::{HookHandler, HookResult};
use crate::rag::document::DocumentRag;
use std::sync::Arc;

/// Hook that injects RAG context into the prompt before it's sent to the LLM.
pub struct DocumentRagHook {
    rag: Arc<DocumentRag>,
}

impl DocumentRagHook {
    pub fn new(rag: Arc<DocumentRag>) -> Self {
        Self { rag }
    }
}

#[async_trait::async_trait]
impl HookHandler for DocumentRagHook {
    fn name(&self) -> &str {
        "document-rag-hook"
    }

    fn priority(&self) -> i32 {
        10 // Run early so RAG context is available for other hooks
    }

    async fn before_prompt_build(&self, prompt: String) -> HookResult<String> {
        // Use the prompt itself as the RAG query
        let max_chunks = self.rag.config.max_chunks;
        let min_relevance = self.rag.config.min_relevance;

        match self.rag.query(&prompt, max_chunks, min_relevance).await {
            Ok(chunks) if !chunks.is_empty() => {
                let context = DocumentRag::build_context(&chunks);
                let augmented = format!("{}\n\n{}", context, prompt);
                tracing::debug!(
                    "RAG hook: injected {} chunks ({} bytes) into prompt",
                    chunks.len(),
                    context.len()
                );
                HookResult::Continue(augmented)
            }
            Ok(_) => {
                tracing::debug!("RAG hook: no relevant chunks found");
                HookResult::Continue(prompt)
            }
            Err(e) => {
                tracing::warn!("RAG hook: query failed: {e}");
                // Don't block the prompt on RAG failure
                HookResult::Continue(prompt)
            }
        }
    }
}
