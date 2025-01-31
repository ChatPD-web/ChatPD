mod arxiv_paper;
mod chat;
mod proc_log;

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use std::error::Error;

use chrono::Utc;

use futures::stream::{self, StreamExt};

use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use tokio::runtime::Runtime;

use crate::arxiv_paper::*;
use crate::chat::{chat_with_llm_sync, ChatInput, CompletionRequest, Config, GetLLMOutput, ChatRecord};
use crate::proc_log::LogEntry;



fn read_strings_from_file(file_path: &str) -> io::Result<Vec<String>> {
    // 打开文件
    let file = File::open(file_path)?;
    let reader = io::BufReader::new(file);

    // 使用 iterator 遍历每一行，读取文件内容到 Vec<String> 中
    let mut strings = Vec::new();
    for line in reader.lines() {
        let line = line?; // 获取每行内容
        strings.push(line); // 将每行内容作为 String 加入 Vec
    }

    Ok(strings)
}

async fn run_parallel_tasks(
    config: Arc<Config>,
    papers: Vec<Paper>,
    input_token_sz: usize,
) -> Result<Vec<Result<PaperRequest, String>>, Box<dyn Error + Send + Sync>> {
    // 获取当前 Tokio 运行时的 Handle
    let handle = tokio::runtime::Handle::current();

    // 使用 Rayon 并行处理
    let results: Vec<Result<PaperRequest, String>> = papers
        .into_par_iter()
        .map(|paper| {
            handle.block_on(async {
                paper.get_paper_request(&config, &input_token_sz).await
            })
            .map_err(|e| e.to_string()) // 将错误转换为 String 类型
        })
        .collect();

    Ok(results)
}

#[derive(Debug, Clone, Copy)]
enum Mode {
    Debug,
    GetPaper,
    GetRequest,
    GetResponse,
    ExtractJsonResponse,
}


#[tokio::main]
async fn main() {
    let update_month = "2024-12";
    let input_token_sz = 1500;
    // let mode: Mode = Mode::GetPaper;
    // let mode: Mode = Mode::GetRequest;
    // let mode: Mode = Mode::GetResponse;
    let mode: Mode = Mode::ExtractJsonResponse;
    let config = Config::from_file("mini4o_openai_api.json");

    // 将配置包装在 Arc 中以便在多个任务间共享
    let config = Arc::new(config);
    let arxiv_ids_path = format!("data/update_data/{}_arxiv_id.txt", update_month);
    let paper_file_path = format!("data/update_data/{}_arxiv_papers.ndjson", update_month);
    let papers_request_path = format!(
        "data/update_data/{}_arxiv_papers_request.ndjson",
        update_month
    );
    let llm_response_path = format!(
        "data/update_data/{}_arxiv_papers_request_response.ndjson",
        update_month
    );
    let pure_response_path = format!(
        "data/update_data/{}_arxiv_papers_request_pure_response.ndjson",
        update_month
    );
    let has_get_request_path = format!("data/update_data/has_get_request.txt");

    let arxiv_metadata_path = "/home/ubuntu/data/arxiv-2024.json";
    // let arxiv_metadata: Vec<ArxivObject> = serde_json::from_str(&arxiv_metadata).unwrap();

    let max_retries = 3;
    let wait_time = 10;

    let target_prefix = format!(
        "{}{}.",
        &update_month[2..4], // 提取年份后两位，例如 "24"
        &update_month[5..7]  // 提取月份，例如 "01"
    );

    let mut papers: Vec<Paper> = match fs::read_to_string(&paper_file_path) {
        Ok(file_content) => {
            file_content
                .lines()
                .enumerate() // 获取行号
                .filter(|(_, line)| !line.trim().is_empty()) // 跳过空行
                .map(|(line_number, line)| {
                    // 解析每一行，如果出错则返回错误
                    serde_json::from_str::<Paper>(line).map_err(|e| {
                        eprintln!("Error parsing line {}: {}: {}", line_number + 1, line, e);
                        e
                    })
                })
                .filter_map(Result::ok) // 只保留成功解析的行
                .collect()
        }
        Err(e) => {
            // 如果文件不存在，打印错误信息并返回空数组
            eprintln!("Error reading file {}: {}", paper_file_path, e);
            Vec::new()
        }
    };


    let mut paper_requests: Vec<PaperRequest> = fs::read_to_string(&papers_request_path)
        .map(|content| {
            if content.trim().is_empty() {
                eprintln!("File is empty");
                vec![] // 返回一个空的 Vec
            } else {
                content
                    .lines()
                    .enumerate() // 获取行号
                    .map(|(line_number, line)| {
                        serde_json::from_str::<PaperRequest>(line).map_err(|e| {
                            eprintln!("Error while processing file: {}", paper_file_path);
                            eprintln!("Error parsing line {}: {}: {}", line_number + 1, line, e);
                            e
                        })
                    })
                    .filter_map(Result::ok) // 只保留成功解析的行
                    .collect()
            }
        })
        .unwrap_or_else(|err| {
            eprintln!("Error reading file: {}", err);
            vec![] // 如果文件读取失败，返回空 Vec
        });


    // 记录开始时间
    let start = std::time::Instant::now();
    // 设置并发数量
    const CONCURRENT_REQUESTS: usize = 64;

    // 终止程序


    match mode {
        Mode::Debug => {
            todo!()
        }
        Mode::GetPaper => {
             // 载入已有的arxiv_metadata
            let mut arxiv_metadatas: Vec<ArxivObject> = match fs::read_to_string(&arxiv_metadata_path) {
                Ok(file_content) => {
                    file_content
                        .lines()
                        .enumerate() // 获取行号
                        .map(|(line_number, line)| {
                            // 解析每一行，如果出错则返回错误
                            serde_json::from_str::<ArxivObject>(line).map_err(|e| {
                                eprintln!("Error while processing file: {}", arxiv_metadata_path);
                                eprintln!("Error parsing line {}: {}: {}", line_number + 1, line, e);
                                e
                            })
                        })
                        .filter_map(Result::ok) // 只保留成功解析的行
                        .collect()
                }
                Err(e) => {
                    // 如果文件不存在，打印错误信息并返回空数组
                    eprintln!("Error reading file {}: {}", arxiv_metadata_path, e);
                    Vec::new()
                }
            };
            // 只考虑catogories中包括cs.AI的arxiv
            arxiv_metadatas.retain(|arxiv| arxiv.categories.contains(&"cs.AI".to_string()) && arxiv.id.starts_with(&target_prefix));
            println!("Processing {} papers", arxiv_metadatas.len());
            
            
            let mut arxiv_ids: Vec<String> = arxiv_metadatas
                .iter()
                .map(|arxiv| arxiv.id.clone())
                .collect();
            // 获取已经处理过的arxiv_id
            let processed_arxiv_ids: Vec<String> =
                papers.iter().map(|paper| paper.id.clone()).collect();
            println!(
                "Processed number of papers: {} has get paper",
                processed_arxiv_ids.len()
            );
            // 过滤掉已经处理过的arxiv_id
            arxiv_ids.retain(|arxiv_id| !processed_arxiv_ids.contains(arxiv_id));
            let results = stream::iter(arxiv_ids)
                .map(|arxiv_id| async move {
                    let mut arxiv_object = arxiv_paper::ArxivObject::new();
                    arxiv_object.id = arxiv_id.clone();

                    match arxiv_object.parse_paper_html().await {
                        Ok(paper) => {
                            println!("Successfully parsed paper {}", arxiv_id);
                            Ok(paper)
                        }
                        Err(e) => {
                            eprintln!("Failed to parse paper {}: {}", arxiv_id, e);
                            Err(e)
                        }
                    }
                })
                .buffer_unordered(CONCURRENT_REQUESTS)
                .collect::<Vec<_>>()
                .await;
            // 处理结果
            let (successful, failed): (Vec<_>, Vec<_>) =
                results.into_iter().partition(Result::is_ok);
            let mut results: Vec<Paper> = successful.into_iter().map(Result::unwrap).collect();

           
            let arxiv_metadata_map: HashMap<String, ArxivObject> = arxiv_metadatas
                .clone()
                .into_iter()
                .map(|arxiv| (arxiv.id.clone(), arxiv))
                .collect();

            for result in &mut results {
                if let Some(arxiv_info) = arxiv_metadata_map.get(&result.id) {
                    result.arxiv_info = Some(arxiv_info.clone());
                }
            }

            let arxiv_objects_ndjson = results
                .into_iter()
                .map(|paper| serde_json::to_string(&paper).unwrap())
                .collect::<Vec<_>>()
                .join("\n");
            // 保存到文件
            // 使用 OpenOptions 以追加模式打开文件
            let mut file = OpenOptions::new()
                .create(true) // 如果文件不存在，则创建
                .append(true) // 如果文件存在，则追加内容
                .open(&paper_file_path)
                .unwrap();
            // 写入内容
            writeln!(file, "{}", arxiv_objects_ndjson).unwrap();
            println!("Successfully saved papers to {}", paper_file_path);
        }
        Mode::GetRequest => {

            // 先读入已经处理过的arxiv_id
            let has_get_request_arxiv_ids =
                match read_strings_from_file(has_get_request_path.as_str()) {
                    Ok(ids) => ids, // 文件读取成功，返回内容
                    Err(e) => {
                        eprintln!("Error reading file: {}", e); // 打印错误
                        Vec::new() // 如果文件不存在或其他错误，返回一个空的 Vec
                    }
                };
            // 过滤掉已经处理过的arxiv_id
            papers.retain(|paper| !has_get_request_arxiv_ids.contains(&paper.id));
            println!("Processing {} papers", papers.len());

            let results = run_parallel_tasks(config.clone(), papers.clone(), input_token_sz).await.unwrap();

            // 处理结果
            let (successful, failed): (Vec<_>, Vec<_>) =
                results.into_iter().partition(Result::is_ok);
            // 打印统计信息
            let duration = start.elapsed();
            println!("Processing completed in {:?}", duration);
            println!("Successfully processed {} papers", successful.len());
            println!("Failed to process {} papers", failed.len());
            // 将results中的成功结果转换为特定对象
            let results: Vec<PaperRequest> = successful
                .into_iter()
                .map(Result::unwrap)
                .collect::<Vec<PaperRequest>>();

            let completion_requests = results
                .into_iter()
                .map(|request| serde_json::to_string(&request).unwrap()) // 序列化
                .collect::<Vec<String>>()
                .join("\n"); // 用换行符连接每个字符串
                             // 使用 OpenOptions 以追加模式打开文件
            let mut file = OpenOptions::new()
                .create(true) // 如果文件不存在，则创建
                .append(true) // 如果文件存在，则追加内容
                .open(&papers_request_path)
                .unwrap();
            // 写入内容
            writeln!(file, "{}", completion_requests).unwrap();
            println!(
                "Successfully saved completion requests to {}",
                &papers_request_path
            );

            // 将处理过的arxiv_id写入文件
            let mut file = OpenOptions::new()
                .create(true) // 如果文件不存在，则创建
                .append(true) // 如果文件存在，则追加内容
                .open(&has_get_request_path)
                .unwrap();
            // 写入内容
            for paper in papers {
                writeln!(file, "{}", paper.id).unwrap();
            }
        }
        Mode::GetResponse => {
            // 先读取response文件
            let has_get_response_arxiv_ids: Vec<String> = fs::read_to_string(&llm_response_path)
                .map(|content| {
                    if content.trim().is_empty() {
                        eprintln!("File is empty");
                        vec![] // 返回一个空的 Vec
                    } else {
                        content
                            .lines()
                            .enumerate() // 获取行号
                            .map(|(line_number, line)| {
                                serde_json::from_str::<ChatRecord>(line).map_err(|e| {
                                    eprintln!("Error parsing line {}: {}: {}", line_number + 1, line, e);
                                    e
                                })
                            })
                            .filter_map(Result::ok) // 只保留成功解析的行
                            .map(|record| record.arxiv_id)
                            .collect()
                    }
                })
                .unwrap_or_else(|err| {
                    eprintln!("Error reading file: {}", err);
                    vec![] // 如果文件读取失败，返回空 Vec
                });
            // 过滤掉已经处理过的arxiv_id
            paper_requests.retain(|request| !has_get_response_arxiv_ids.contains(&request.arxiv_id));
            println!("Processing {} paper requests", paper_requests.len());
            let results: Vec<_> = paper_requests
                .into_par_iter() // 注意这里使用的是 rayon 提供的 into_par_iter
                .map(|paper_request| {
                    println!("Processing paper request {}", paper_request.arxiv_id);
                    // 调用同步的 chat_with_llm_sync 方法
                    // chat_with_llm_sync(&config, ChatInput::Messages(paper_request.messages.clone()))
                    let mut attempt = 0;
                    let chat_response = loop {
                        let result = chat_with_llm_sync(&config, ChatInput::Messages(paper_request.messages.clone()));
                        if attempt >= max_retries {
                            break result;
                        }

                        match result {
                            Ok(response) => break Ok(response),
                            Err(e) => {
                                // 如果错误信息包括429 Too Many Requests，等待一段时间后重试
                                if e.to_string().contains("429 Too Many Requests") {
                                    println!("Too many requests, waiting for {} seconds... (Attempt {}/{})", wait_time, attempt + 1, max_retries);
                                    thread::sleep(Duration::from_secs(wait_time));
                                    attempt += 1;
                                    continue;
                                }
                                break Err(e);
                            }
                        }
                    }; 
                    println!("Finished processing paper with ID: {}", paper_request.arxiv_id);

                    let chat_record = match chat_response {
                        Ok(chat_response) => {
                            let answer_message = chat_response.get_llm_output(&config).expect("Failed to get LLM output");
                            let success_log = LogEntry {
                                timestamp: Utc::now().to_rfc3339(),
                                paper_id: paper_request.arxiv_id.clone(),
                                status: "success".to_string(),
                                message: "Paper processed successfully".to_string(),
                            };
                            println!("Log: {:?}", success_log);
                            ChatRecord {
                                arxiv_id: paper_request.arxiv_id.clone(),
                                messages: paper_request.messages.clone(),
                                log: success_log,
                                model: config.model.clone(),
                                result: Some(answer_message),
                            }
                        }
                        Err(e) => {
                            let error_log = LogEntry {
                                timestamp: Utc::now().to_rfc3339(),
                                paper_id: paper_request.arxiv_id.clone(),
                                status: "error".to_string(),
                                message: format!("Error processing paper: {}", e),
                            };
                            ChatRecord {
                                arxiv_id: paper_request.arxiv_id.clone(),
                                messages: paper_request.messages.clone(),
                                log: error_log,
                                model: config.model.clone(),
                                result: None,
                            }
                        }
                    };
                    println!("Chat record: {:?}", chat_record);
                    chat_record
                })
                .collect();
            // println!("results: {:?}", results);
            let chat_record_ndjson = results
                .into_iter()
                .map(|record| serde_json::to_string(&record).unwrap())
                .collect::<Vec<_>>()
                .join("\n");
            // 使用 OpenOptions 以追加模式打开文件
            let mut file = OpenOptions::new()
                .create(true) // 如果文件不存在，则创建
                .append(true) // 如果文件存在，则追加内容
                .open(&llm_response_path)
                .unwrap();
            // 写入内容
            writeln!(file, "{}", chat_record_ndjson).unwrap();
            println!("Successfully saved chat records to {}", llm_response_path);
        }
        Mode::ExtractJsonResponse => {
            let chat_records: Vec<ChatRecord> = fs::read_to_string(&llm_response_path)
                .unwrap()
                .lines()
                .enumerate() // 获取行号
                .map(|(line_number, line)| {
                    // 解析每一行，如果出错则返回错误
                    serde_json::from_str::<ChatRecord>(line).map_err(|e| {
                        eprintln!("Error parsing line {}: {}: {}", line_number + 1, line, e);
                        e
                    })
                })
                .filter_map(Result::ok) // 只保留成功解析的行
                .collect();
            // 获取chat_records中的results
            let results: Vec<String> = chat_records
                .into_iter()
                .filter_map(|record| record.result) // 过滤掉 None，只保留 Some 的值
                .collect();
            // 写入pure_response_path
            let pure_response = results
                .into_iter()
                .map(|result| {
                    let processed_result = result.replace("\\n", "\n"); // 将 \n 替换为真正的换行
                    processed_result
                })
                .collect::<Vec<_>>()
                .join("\n");
            // 直接写
            let mut file = OpenOptions::new()
                .create(true) // 如果文件不存在，则创建
                .write(true) // 如果文件存在，则覆盖内容
                .open(&pure_response_path)
                .unwrap();
            // 写入内容
            writeln!(file, "{}", pure_response).unwrap();
            println!("Successfully saved pure responses to {}", pure_response_path);
        }
        _ => {
            todo!()
        }
    }
}
