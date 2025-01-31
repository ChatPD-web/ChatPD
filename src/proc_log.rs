use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// 假设有一个结构体来表示日志条目
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LogEntry {
    pub timestamp: String,
    pub paper_id: String,
    pub status: String,
    pub message: String,
}

impl LogEntry {
    // 将日志条目序列化为JSON字符串
    pub fn to_json(&self) -> Value {
        json!({
            "timestamp": self.timestamp,
            "paper_id": self.paper_id,
            "status": self.status,
            "message": self.message,
        })
    }
}
