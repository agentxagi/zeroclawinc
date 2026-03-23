//! /api/callback — receive and store task results from external systems.

use super::api::{require_auth, AppState};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json},
};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
pub struct CallbackRequest {
    pub task_id: String,
    pub status: String,
    pub result: Option<serde_json::Value>,
    pub metadata: Option<serde_json::Value>,
}

/// POST /api/callback — receive a task result callback.
///
/// Stores the callback in memory for retrieval and optionally forwards
/// to a configured webhook URL (for ZeroInc integration).
pub async fn handle_api_callback(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<CallbackRequest>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let now = chrono::Utc::now().to_rfc3339();
    let callback_key = format!("callback:{}", body.task_id);

    // Store in memory for retrieval
    let content = serde_json::to_string_pretty(&json!({
        "task_id": body.task_id,
        "status": body.status,
        "result": body.result,
        "metadata": body.metadata,
        "received_at": now,
    }))
    .unwrap_or_default();

    match state
        .mem
        .store(
            &callback_key,
            &content,
            crate::memory::MemoryCategory::Custom("callback".into()),
            None,
        )
        .await
    {
        Ok(()) => Json(json!({
            "stored": true,
            "task_id": body.task_id,
            "status": body.status,
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("Failed to store callback: {e}")})),
        )
            .into_response(),
    }
}

/// GET /api/callback/:task_id — retrieve a stored callback result.
pub async fn handle_api_callback_get(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Path(task_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let callback_key = format!("callback:{}", task_id);

    match state.mem.get(&callback_key).await {
        Ok(Some(entry)) => {
            // Parse the stored JSON content
            let parsed: serde_json::Value =
                serde_json::from_str(&entry.content).unwrap_or(json!({"raw": entry.content}));
            Json(json!({
                "found": true,
                "data": parsed,
            }))
            .into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("No callback found for task_id '{task_id}'")})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}
