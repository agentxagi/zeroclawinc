//! GET /api/worker/{id}/events — SSE stream for real-time worker task progress.

use super::api::{require_auth, AppState};
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
};
use std::convert::Infallible;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use async_stream::stream;

/// GET /api/worker/{id}/events — SSE event stream for a specific worker task.
pub async fn handle_worker_events(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(work_id): Path<String>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    // Check task exists
    {
        let store = state.work_store.read().await;
        if !store.contains_key(&work_id) {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": format!("Work '{work_id}' not found")})),
            )
                .into_response();
        }
    }

    let rx = match state.worker_event_tx.get(&work_id) {
        Some(tx) => tx.subscribe(),
        None => {
            // Task exists but no active event channel — it may have already
            // completed. Return a single final-status event and close.
            let store = state.work_store.read().await;
            let status = store
                .get(&work_id)
                .map(|s| serde_json::json!({
                    "type": "status",
                    "work_id": work_id,
                    "status": s.status,
                    "result": s.result,
                    "error": s.error,
                }))
                .unwrap_or(serde_json::json!({"type": "error", "message": "task not found"}));

            let stream = stream! {
                yield Ok::<_, Infallible>(Event::default().data(status.to_string()));
            };
            return Sse::new(stream).into_response();
        }
    };

    let stream =
        BroadcastStream::new(rx).filter_map(
            |result: Result<
                serde_json::Value,
                tokio_stream::wrappers::errors::BroadcastStreamRecvError,
            >| {
                match result {
                    Ok(value) => Some(Ok::<_, Infallible>(
                        Event::default().data(value.to_string()),
                    )),
                    Err(_) => None,
                }
            },
        );

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}
