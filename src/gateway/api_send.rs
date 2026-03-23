//! /api/send — outbound message dispatch via channels.

use super::api::{require_auth, AppState};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json},
};
use crate::channels::{Channel, SendMessage};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
pub struct SendRequest {
    pub channel: String,
    pub target: String,
    pub message: String,
    pub subject: Option<String>,
}

/// POST /api/send — dispatch an outbound message to a channel.
pub async fn handle_api_send(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<SendRequest>,
) -> impl IntoResponse {
    if let Err(e) = require_auth(&state, &headers) {
        return e.into_response();
    }

    let channel_name = body.channel.to_lowercase();

    let msg = if let Some(subject) = &body.subject {
        SendMessage::with_subject(&body.message, &body.target, subject)
    } else {
        SendMessage::new(&body.message, &body.target)
    };

    let send_result: Result<&str, String> = match channel_name.as_str() {
        "whatsapp" => match &state.whatsapp {
            Some(ch) => ch.send(&msg).await.map(|_| "whatsapp").map_err(|e| e.to_string()),
            None => Err("WhatsApp channel not configured".into()),
        },
        "linq" => match &state.linq {
            Some(ch) => ch.send(&msg).await.map(|_| "linq").map_err(|e| e.to_string()),
            None => Err("Linq channel not configured".into()),
        },
        "nextcloud_talk" | "nextcloud-talk" => match &state.nextcloud_talk {
            Some(ch) => ch.send(&msg).await.map(|_| "nextcloud_talk").map_err(|e| e.to_string()),
            None => Err("Nextcloud Talk channel not configured".into()),
        },
        "wati" => match &state.wati {
            Some(ch) => ch.send(&msg).await.map(|_| "wati").map_err(|e| e.to_string()),
            None => Err("WATI channel not configured".into()),
        },
        _ => Err(format!(
            "Unknown channel '{}'. Available: whatsapp, linq, nextcloud_talk, wati",
            channel_name
        )),
    };

    match send_result {
        Ok(ch) => Json(json!({
            "sent": true,
            "channel": ch,
            "target": body.target,
        }))
        .into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": e})),
        )
            .into_response(),
    }
}
