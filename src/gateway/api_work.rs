//! /api/worker — async task execution via the agent loop.

use super::api::{require_auth, AppState};
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Deserialize)]
pub struct WorkerRequest {
    pub task_id: Option<String>,
    pub prompt: String,
    pub session_id: Option<String>,
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkStatus {
    pub work_id: String,
    pub task_id: Option<String>,
    pub status: String, // "pending", "running", "completed", "failed", "timeout"
    pub result: Option<String>,
    pub error: Option<String>,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
}

/// In-memory work status store.
pub type WorkStore = Arc<RwLock<HashMap<String, WorkStatus>>>;

/// POST /api/worker — submit a task for async execution.
pub async fn handle_api_worker(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<WorkerRequest>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let work_id = Uuid::new_v4().to_string();
    let timeout_secs = body.timeout_secs.unwrap_or(600);
    let task_id = body.task_id.clone();
    let prompt = body.prompt.clone();
    let session_id = body.session_id.clone();

    // Create initial status
    let status = WorkStatus {
        work_id: work_id.clone(),
        task_id: task_id.clone(),
        status: "pending".into(),
        result: None,
        error: None,
        started_at: None,
        completed_at: None,
    };

    // Store initial status
    {
        let mut store = state.work_store.write().await;
        store.insert(work_id.clone(), status);
    }

    // Spawn async work task
    let state_clone = state.clone();
    let wid = work_id.clone();
    tokio::spawn(async move {
        // Mark as running
        {
            let mut store = state_clone.work_store.write().await;
            if let Some(s) = store.get_mut(&wid) {
                s.status = "running".into();
                s.started_at = Some(chrono::Utc::now().to_rfc3339());
            }
        }

        // Run with timeout
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            super::run_gateway_chat_with_tools(&state_clone, &prompt, session_id.as_deref()),
        )
        .await;

        // Update status
        let mut store = state_clone.work_store.write().await;
        if let Some(s) = store.get_mut(&wid) {
            match result {
                Ok(Ok(response)) => {
                    s.status = "completed".into();
                    s.result = Some(response);
                    s.completed_at = Some(chrono::Utc::now().to_rfc3339());
                }
                Ok(Err(e)) => {
                    s.status = "failed".into();
                    s.error = Some(e.to_string());
                    s.completed_at = Some(chrono::Utc::now().to_rfc3339());
                }
                Err(_) => {
                    s.status = "timeout".into();
                    s.error = Some(format!("Task timed out after {timeout_secs}s"));
                    s.completed_at = Some(chrono::Utc::now().to_rfc3339());
                }
            }
        }
    });

    Json(json!({
        "work_id": work_id,
        "task_id": task_id,
        "status": "pending",
    }))
    .into_response()
}

/// GET /api/worker/:id/status — poll work status.
pub async fn handle_api_worker_status(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(work_id): Path<String>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let store = state.work_store.read().await;
    match store.get(&work_id) {
        Some(status) => Json(status).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("Work '{work_id}' not found")})),
        )
            .into_response(),
    }
}

/// GET /api/worker — list all work items.
pub async fn handle_api_worker_list(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let store = state.work_store.read().await;
    let items: Vec<&WorkStatus> = store.values().collect();
    Json(json!({"work_items": items, "count": items.len()})).into_response()
}
