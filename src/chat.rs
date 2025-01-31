use reqwest::Client;
use serde_json::Value;
use once_cell::sync::Lazy;
use std::{error::Error, fs, io, time::Duration};
// use tiktoken_rs::async_openai::get_chat_completion_max_tokens;
// use async_openai::types::{ChatCompletionRequestMessage, Role};
use tiktoken_rs::{num_tokens_from_messages, ChatCompletionRequestMessage};
use crate::proc_log::LogEntry;


use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ChatInput {
    Prompt(String),
    Messages(Vec<Message>),
    CompletionRequest(CompletionRequest),
}

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    api_key: String,
    api_base: String,
    pub model: String,
    token_model: String,
}

impl Config {
    pub fn from_file(config_path: &str) -> Self {
        let config_text = fs::read_to_string(config_path)
            .expect("Failed to read config file");
        let config: Config = serde_json::from_str(&config_text)
            .expect("Failed to parse config JSON");
        config
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatRecord {
    pub arxiv_id: String,
    pub messages: Vec<Message>,
    pub log: LogEntry,
    pub model: String,
    pub result: Option<String>,
}


static HTTP_CLIENT: Lazy<Client> = Lazy::new(Client::new);


// 不使用这个函数
pub async fn chat_with_gpt(config: &Config, input: ChatInput) -> Result<Value, Box<dyn Error + Send>> {
    // 构造请求体
    let request_body = if config.model.contains("gemini") {
        match input {
            ChatInput::Prompt(prompt) => {
                let message = serde_json::json!({
                    "role": "user",
                    "parts": [{"text": prompt}],
                });
                serde_json::json!({
                    "contents": [message],
                })
            },
            ChatInput::Messages(messages) => {
                let formatted_messages: Vec<Value> = messages.into_iter().map(|msg| {
                    serde_json::json!({
                        "role": msg.role,
                        "parts": [{"text": msg.content}],
                    })
                }).collect();
                serde_json::json!({
                    "contents": formatted_messages,
                })
            },
            ChatInput::CompletionRequest(completion_request) => {
                let formatted_messages: Vec<Value> = completion_request.messages.into_iter().map(|msg| {
                    serde_json::json!({
                        "role": msg.role,
                        "parts": [{"text": msg.content}],
                    })
                }).collect();
                serde_json::json!({
                    "contents": formatted_messages,
                })
            },
        }
    } else {
        match input {
            ChatInput::Prompt(prompt) => {
                let message = Message {
                    role: "user".to_string(),
                    content: prompt,
                };
                serde_json::json!({
                    "model": config.model,
                    "messages": [message],
                    "temperature": 0.7,
                })
            },
            ChatInput::Messages(messages) => {
                serde_json::json!({
                    "model": config.model,
                    "messages": messages,
                    "temperature": 0.7,
                })
            },
            ChatInput::CompletionRequest(completion_request) => serde_json::json!(completion_request),
        }
    };

    println!("{:?}", request_body);

    // 发起同步HTTP POST请求
    let api_url = if config.model.contains("gemini") {
        format!("{}/models/{}:GenerateContent?key={}", config.api_base, config.model, config.api_key)
    } else {
        format!("{}/chat/completions", config.api_base)
    };
    println!("{}", api_url);

    let response = if config.model.contains("gemini") {
        HTTP_CLIENT
        .post(&api_url)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await
        .map_err(|e| Box::new(e) as Box<dyn Error + Send>)? // 使用 map_err 来转换错误
    } else {
        HTTP_CLIENT
        .post(&api_url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", config.api_key))
        .json(&request_body)
        .send()
        .await
        .map_err(|e| Box::new(e) as Box<dyn Error + Send>)? // 使用 map_err 来转换错误
    };

    if response.status().is_success() {
        let response_text = response.text().await.map_err(|e| Box::new(e) as Box<dyn Error + Send>)?; // 再次使用 map_err 来转换错误
        let response_json: Value = serde_json::from_str(&response_text).map_err(|e| Box::new(e) as Box<dyn Error + Send>)?; // 再次使用 map_err 来转换错误
        Ok(response_json)
    } else {
        Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Request failed with status: {}", response.status()),
        )) as Box<dyn Error + Send>) // 确保错误类型匹配
    }
}

pub fn chat_with_llm_sync(config: &Config, input: ChatInput) -> Result<Value, Box<dyn Error + Send>> {
    // 构造请求体
    let request_body = if config.model.contains("gemini") {
        // Gemini和其他模型的请求体格式不同
        // 除了一般格式上的要求，Gemini是没有system和assistant的role
        match input {
            ChatInput::Prompt(prompt) => {
                let message = serde_json::json!({
                    "role": "user",
                    "parts": [{"text": prompt}],
                });
                serde_json::json!({
                    "contents": [message],
                })
            },
            ChatInput::Messages(messages) => {
                let formatted_messages: Vec<Value> = messages.into_iter()
                .filter(|msg| msg.role == "user")
                .map(|msg| {
                    serde_json::json!({
                        "role": msg.role,
                        "parts": [{"text": msg.content}],
                    })
                }).collect();
                serde_json::json!({
                    "contents": formatted_messages,
                })
            },
            ChatInput::CompletionRequest(completion_request) => {
                let formatted_messages: Vec<Value> = completion_request.messages.into_iter()
                .filter(|msg| msg.role == "user")
                .map(|msg| {
                    serde_json::json!({
                        "role": msg.role,
                        "parts": [{"text": msg.content}],
                    })
                }).collect();
                serde_json::json!({
                    "contents": formatted_messages,
                })
            },
        }
    } else {
        match input {
            ChatInput::Prompt(prompt) => {
                let message = Message {
                    role: "user".to_string(),
                    content: prompt,
                };
                serde_json::json!({
                    "model": config.model,
                    "messages": [message],
                    "temperature": 0.7,
                })
            },
            ChatInput::Messages(messages) => {
                serde_json::json!({
                    "model": config.model,
                    "messages": messages,
                    "temperature": 0.7,
                })
            },
            ChatInput::CompletionRequest(completion_request) => serde_json::json!(completion_request),
        }
    };

     
    // println!("{:?}", request_body);

    // 发起同步HTTP POST请求
    let api_url = if config.model.contains("gemini") {
        format!("{}/models/{}:generateContent?key={}", config.api_base, config.model, config.api_key)
    } else {
        format!("{}/chat/completions", config.api_base)
    };

    // println!("{}", api_url);

    // 设置超时时间
    let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120)) // 设置超时时间为120秒
            .build()
            .map_err(|e| Box::new(e) as Box<dyn Error + Send>)?;
    // let response = client.clone().post(&api_url)
    //     .header("Content-Type", "application/json")
    //     .header("Authorization", format!("Bearer {}", config.api_key))
    //     .json(&request_body)
    //     .send()
    //     .map_err(|e| Box::new(e) as Box<dyn Error + Send>)?;
    let response = if config.model.contains("gemini") {
        client.clone().post(&api_url)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .map_err(|e| Box::new(e) as Box<dyn Error + Send>)?
    } else {
        client.clone().post(&api_url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", config.api_key))
        .json(&request_body)
        .send()
        .map_err(|e| Box::new(e) as Box<dyn Error + Send>)?
    };

    // 检查响应状态，并解析响应体
    if response.status().is_success() {
        let response_text = response.text().map_err(|e| Box::new(e) as Box<dyn Error + Send>)?;
        let response_json: Value = serde_json::from_str(&response_text).map_err(|e| Box::new(e) as Box<dyn Error + Send>)?;
        Ok(response_json)
    } else {
        Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Request failed with status: {}", response.status()),
        )) as Box<dyn Error + Send>)
    }
}

pub trait GetLLMOutput {
    fn get_llm_output(&self, config: &Config) -> Result<String, Box<dyn Error + Send>>;
}

impl GetLLMOutput for Value {
    fn get_llm_output(&self, config: &Config) -> Result<String, Box<dyn Error + Send>> {
        // 首先确保这个变量要么包含candidates要么包含choices
        if !self["candidates"].is_array() && !self["choices"].is_array() {
            return Err(Box::new(io::Error::new(io::ErrorKind::Other, "Failed to get response text")) as Box<dyn Error + Send>);
        }

        if config.model.contains("gemini") {
            Ok(self["candidates"][0]["content"]["parts"][0]["text"]
                .as_str()
                .ok_or_else(|| Box::new(io::Error::new(io::ErrorKind::Other, "Failed to get response text")) as Box<dyn Error + Send>)?
                .to_string())
        } else {
            Ok(self["choices"][0]["message"]["content"]
                .as_str()
                .ok_or_else(|| Box::new(io::Error::new(io::ErrorKind::Other, "Failed to get response text")) as Box<dyn Error + Send>)?
                .to_string())
        }
    }
}



impl Config {
    pub fn new(config_path: &str) -> Self {
        let config_text = fs::read_to_string(config_path)
            .expect("Failed to read config file");
        serde_json::from_str(&config_text)
            .expect("Failed to parse config JSON")
    }
}


pub trait Tokenizable {
    fn num_tokens(&self, config: &Config) -> usize;
    fn content_with_max_tokens(&self, config: &Config, max_tokens: &usize) -> String;
}

impl Tokenizable for Message {
    fn num_tokens(&self, config: &Config) -> usize {
        let messages = vec![
            ChatCompletionRequestMessage{
                content: Some(self.content.clone()),
                role: self.role.clone(),
                name: None,
                function_call: None,
            }
        ];
        num_tokens_from_messages(&config.token_model, &messages).unwrap()
    }

    fn content_with_max_tokens(&self, config: &Config, max_tokens: &usize) -> String {
        let mut left = 0;
        let mut right = self.content.len();
        let mut result = 0;

        while left <= right {
            let mid = left + (right - left) / 2;
            let test_content = &self.content[..mid];
            let test_message = Message {
                content: test_content.to_string(),
                role: self.role.clone(),
            };

            let tokens = test_message.num_tokens(config);

            if tokens <= *max_tokens {
                result = mid; // 更新满足条件的最长 content 长度
                left = mid + 1; // 尝试寻找更长的满足条件的 content
            } else {
                right = mid - 1; // 缩短 content 长度以减少 token 数量
            }
        }

        self.content[..result].to_string()
    }
}

impl Tokenizable for String {
    fn num_tokens(&self, config: &Config) -> usize {
        let messages = vec![
            ChatCompletionRequestMessage {
                content: Some(self.clone()),
                role: "user".to_string(),
                name: None,
                function_call: None,
            }
        ];
        num_tokens_from_messages(&config.token_model, &messages).unwrap()
    }

    fn content_with_max_tokens(&self, config: &Config, max_tokens: &usize) -> String {
        let char_indices: Vec<_> = self.char_indices().map(|(idx, _)| idx).collect();
        let mut left = 0;
        let mut right = char_indices.len();
        let mut result_idx = 0;

        while left <= right {
            let mid = left + (right - left) / 2;
            let byte_pos = *char_indices.get(mid).unwrap_or(&self.len());

            let test_content = &self[..byte_pos];
            let tokens = test_content.num_tokens(&config);

            // 允许一定的误差减少二分查找的次数
            if tokens > *max_tokens - 10 && tokens <= *max_tokens + 30 {
                result_idx = mid;
                break;
            }

            if tokens <= *max_tokens {
                result_idx = mid;
                left = mid + 1;
            } else {
                right = mid.checked_sub(1).unwrap_or(0);
            }
        }

        // 使用char_indices来确定安全的字符串边界
        let byte_pos = *char_indices.get(result_idx).unwrap_or(&self.len());
        self[..byte_pos].to_string()
    }
}

impl Tokenizable for str {
    fn num_tokens(&self, config: &Config) -> usize {
        let messages = vec![
            ChatCompletionRequestMessage {
                content: Some(self.to_string()),
                role: "user".to_string(),
                name: None,
                function_call: None,
            }
        ];
        num_tokens_from_messages(&config.token_model, &messages).unwrap()
    }

    fn content_with_max_tokens(&self, config: &Config, max_tokens: &usize) -> String {
        let mut left = 0;
        let mut right = self.len();
        let mut result = 0;

        while left <= right {
            let mid = left + (right - left) / 2;
            let test_content = &self[..mid];

            let tokens = test_content.num_tokens(config);

            if tokens <= *max_tokens {
                result = mid; // 更新满足条件的最长 content 长度
                left = mid + 1; // 尝试寻找更长的满足条件的 content
            } else {
                right = mid - 1; // 缩短 content 长度以减少 token 数量
            }
        }

        self[..result].to_string()
    }
}

impl Tokenizable for CompletionRequest {
    fn num_tokens(&self, config: &Config) -> usize {
        self.messages.iter().map(|message| message.num_tokens(config)).sum()
    }

    fn content_with_max_tokens(&self, config: &Config, max_tokens: &usize) -> String {
        // 本质这个函数不应该被用到
        unimplemented!()
    }
}

impl CompletionRequest {
    pub fn new(config: &Config, messages: Vec<Message>) -> Self {
        Self {
            model: config.model.to_string(),
            messages,
            temperature: 0.7,
        }
    }
}

impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        if role != "user" && role != "assistant" && role != "system" {
            panic!("Invalid role: {}", role);
        }
        Self {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}
