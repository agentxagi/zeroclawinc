//! RAG API endpoints for document management and retrieval.

use super::api::{require_auth, AppState};
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json, Response},
};
use serde_json::json;

/// GET /api/rag/documents — list all indexed documents.
pub async fn handle_rag_documents_list(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let rag = match &state.rag {
        Some(r) => r,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "RAG is not enabled. Set [rag] enabled = true in config."})),
            )
                .into_response();
        }
    };

    let documents = rag.list_documents().await;
    Json(json!({"documents": documents})).into_response()
}

/// POST /api/rag/ingest — ingest a document.
pub async fn handle_rag_ingest(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let rag = match &state.rag {
        Some(r) => r,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "RAG is not enabled. Set [rag] enabled = true in config."})),
            )
                .into_response();
        }
    };

    let req = match serde_json::from_value::<crate::rag::document::IngestRequest>(body) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("Invalid request: {e}")})),
            )
                .into_response();
        }
    };

    match rag.ingest(&req).await {
        Ok(doc) => (StatusCode::CREATED, Json(json!({"document": doc}))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Ingestion failed: {e}")})),
        )
            .into_response(),
    }
}

/// POST /api/rag/query — query the RAG system.
pub async fn handle_rag_query(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let rag = match &state.rag {
        Some(r) => r,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "RAG is not enabled. Set [rag] enabled = true in config."})),
            )
                .into_response();
        }
    };

    let req = match serde_json::from_value::<crate::rag::document::QueryRequest>(body) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("Invalid request: {e}")})),
            )
                .into_response();
        }
    };

    let max_chunks = req.max_chunks.unwrap_or(rag.config.max_chunks);
    let min_relevance = req.min_relevance.unwrap_or(rag.config.min_relevance);

    match rag.query(&req.query, max_chunks, min_relevance).await {
        Ok(chunks) => Json(json!({"chunks": chunks, "count": chunks.len()})).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Query failed: {e}")})),
        )
            .into_response(),
    }
}

/// DELETE /api/rag/documents/:id — delete a document and its chunks.
pub async fn handle_rag_document_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let rag = match &state.rag {
        Some(r) => r,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "RAG is not enabled. Set [rag] enabled = true in config."})),
            )
                .into_response();
        }
    };

    match rag.delete_document(&id).await {
        Ok(true) => Json(json!({"deleted": true, "id": id})).into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("Document {id} not found")})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Delete failed: {e}")})),
        )
            .into_response(),
    }
}
