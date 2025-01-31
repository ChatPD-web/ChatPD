use chrono::prelude::*;
use rand::{seq::SliceRandom, thread_rng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{from_reader, to_writer_pretty};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::fmt;
use async_trait::async_trait;
use tokio::io::AsyncReadExt;


use reqwest;
use scraper::{Html, Selector};

use crate::chat::{CompletionRequest, Config, Message, Tokenizable};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ArxivObject {
    pub id: String,
    pub submitter: String,
    pub authors: String,
    pub title: String,
    pub comments: Option<String>,
    pub journal_ref: Option<String>,
    pub doi: Option<String>,
    pub report_no: Option<String>,
    pub categories: String,
    pub license: Option<String>,
    pub r#abstract: String,
    pub versions: Vec<Version>,
    pub update_date: String,
    pub authors_parsed: Vec<Vec<String>>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Version {
    pub version: String,
    pub created: String, // 或者如果你想自动解析日期，可以使用 DateTime<Utc>
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Paper {
    pub id: String,
    pub sections: Vec<Section>,
    pub arxiv_info: Option<ArxivObject>,
}

#[async_trait::async_trait] // 使用 async-trait crate 来支持异步 trait 方法
pub trait PaperExt {
    async fn get_datasets(&self) -> Result<Vec<String>, reqwest::Error>;
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Section {
    pub section_name: String,
    pub content: Vec<String>, // Paragraphs in the section
}

#[derive(Deserialize, Debug)]
pub struct Dataset {
    url: String,
    name: String,
}

#[derive(Deserialize, Debug)]
pub struct Data {
    pub mentioned: Vec<Dataset>,
}

#[derive(Deserialize, Debug)]
pub struct PWCResponse {
    pub data: Data,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PaperRequest {
    pub arxiv_id: String,
    pub messages: Vec<Message>,
}


pub fn read_and_filter_objects_line(file_path: &Path) -> io::Result<Vec<ArxivObject>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    // 使用并行迭代器读取和过滤
    let objects: Vec<ArxivObject> = reader
        .lines()
        .filter_map(|line| line.ok())
        .filter_map(|line| serde_json::from_str::<ArxivObject>(&line).ok())
        // .filter(|obj| obj.categories.contains("cs.AI"))
        .filter(|obj| obj.categories.contains("cs.CL"))
        .filter(|obj| {
            // println!("obj.update_date: {}", obj.update_date);
            obj.versions.iter().any(|v| {
                if let Ok(date) =
                    NaiveDate::parse_from_str(&obj.update_date, "%Y-%m-%d").map(|date| date.year())
                {
                    date >= 2018 && date <= 2023
                } else {
                    false
                }
            })
        })
        .collect();

    Ok(objects)
}

pub fn read_objects(file_path: &Path, n: Option<usize>) -> io::Result<Vec<ArxivObject>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let objects: Vec<ArxivObject> = serde_json::from_reader(reader)?;

    // 如果 n 有值，则只取前 n 个对象；否则取所有对象
    let len = n.unwrap_or_else(|| objects.len());
    let filtered_objects: Vec<ArxivObject> = objects.into_iter().take(len).collect();

    Ok(filtered_objects)
}

fn clean_text(text: &str) -> String {
    text.replace('\n', " ") // 替换换行符为空格
        .split_whitespace() // 分割所有空白字符，包括连续空格
        .collect::<Vec<&str>>() // 收集分割结果
        .join(" ") // 使用单个空格重新连接
}

pub fn sample_and_write(
    objects: Vec<ArxivObject>,
    output_path: &str,
    sample_size: usize,
) -> io::Result<()> {
    let mut rng = thread_rng();
    // 采样，如果对象数量少于采样大小，则复制全部对象
    let mut sampled_objects: Vec<_> = if objects.len() > sample_size {
        objects
            .choose_multiple(&mut rng, sample_size)
            .cloned()
            .collect()
    } else {
        objects.clone()
    };

    let file = File::create(Path::new(output_path))?;
    let mut writer = BufWriter::new(file);

    // 对每个采样对象的title和abstract字段进行清理
    for obj in sampled_objects.iter_mut() {
        obj.title = clean_text(&obj.title);
        obj.authors = clean_text(&obj.authors);
        obj.r#abstract = clean_text(&obj.r#abstract);
    }

    // 直接将整个采样结果作为一个数组写入
    to_writer_pretty(&mut writer, &sampled_objects)?;

    Ok(())
}

pub fn load_papers_in_json(file_path: &Path) -> Result<Vec<Paper>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let papers: Vec<Paper> = from_reader(reader)?;

    Ok(papers)
}

#[async_trait]
pub trait PaperParser {
    async fn parse_paper_html(&self) -> Result<Paper, Box<dyn Error>>;
}

#[async_trait]
impl PaperParser for ArxivObject {
    async fn parse_paper_html(&self) -> Result<Paper, Box<dyn Error>> {
        // 你的解析逻辑
        println!("Starting to fetch paper with ID: {}", self.id);
        // https://ar5iv.org/html/2105.12092
        let url = format!("https://ar5iv.org/html/{}", self.id);
        println!("Fetching URL: {}", url);
        let response = reqwest::get(&url).await?;

        if !response.status().is_success() {
            eprintln!("Failed to fetch URL {}: {}", url, response.status());
            return Err("Failed to fetch URL".into());
        }
        let html = response.text().await?;
        let document = Html::parse_document(&html);

        // Parse sections
        let section_selector = Selector::parse("section.ltx_section").unwrap();
        let mut sections: Vec<Section> = Vec::new();

        for section in document.select(&section_selector) {
            let section_name_selector = Selector::parse("h2.ltx_title_section").unwrap();
            let section_name_element = section.select(&section_name_selector).next();

            let section_name = match section_name_element {
                Some(element) => element.text().collect::<Vec<_>>().join(" "),
                None => {
                    eprintln!("Error parsing section name for paper ID: {}", self.id);
                    continue;  // Skip this section or you can choose to return an error instead
                }
            };

            let para_selector = Selector::parse("div.ltx_para p").unwrap();
            let content: Vec<String> = section
                .select(&para_selector)
                .map(|para| para.text().collect::<Vec<_>>().join(" "))
                .collect();

            sections.push(Section { section_name, content });
        }

        Ok(Paper {
            id: self.id.to_string(),
            sections: sections,
            arxiv_info: Some(self.clone()),
        })
    }
}

impl Paper {
    pub async fn get_paper_dataset_related_info(&self, config: &Config, input_token_sz: &usize) -> Result<String, Box<dyn Error>> {
        if self.sections.is_empty() {
            return Err("This paper has no sections".into());
        }

        let arxiv_info = self.arxiv_info.as_ref().ok_or("Missing arxiv_info")?;
        let paper_info = format!(
            "arxiv id: {}\ntitle: {}\nabstract: {}\n",
            arxiv_info.id,
            arxiv_info.title,
            arxiv_info.r#abstract
        );

        let keywords = ["dataset", "data", "experiment", "evaluation", "result"];
        let contain_dataset_flag = self.sections.iter().any(|section| {
            keywords.iter().any(|&keyword| section.section_name.to_lowercase().contains(keyword))
        });

        if !contain_dataset_flag {
            return Err("No dataset related section found in paper".into());
        }

        let dataset_info = self
            .sections
            .iter()
            .filter(|section| {
                keywords.iter().any(|&keyword| section.section_name.to_lowercase().contains(keyword))
            })
            .flat_map(|section| section.content.iter().map(|s| s.as_str()))
            .collect::<Vec<&str>>()
            .join("\n");
        
        let dataset_related_info = format!("{}Dataset Related Section Content: {}", paper_info, dataset_info);
        let front_info = dataset_related_info.content_with_max_tokens(config, input_token_sz).to_lowercase();
        if front_info.contains("dataset") || front_info.contains("data set") || front_info.contains("benchmark"){
            Ok(dataset_related_info)
        } else {
            Err("This info doesn't contain dataset description.".into())
        }
    }

    pub async fn generate_paper_message(&self, config: &Config, input_token_sz: &usize) -> Result<CompletionRequest, Box<dyn Error>> {
        let system_message = Message::new("system", "You're a Computer Science researcher. Your task is to extract the dataset related information from the given paper information.");
        let assistant_message = Message::new("assistant", "I'm here to help you extract dataset information. Please provide me the paper information.");
        let dataset_related_info = self.get_paper_dataset_related_info(config, input_token_sz).await?.content_with_max_tokens(config, input_token_sz);
        let user_message = Message::new("user", &format!(r#"I hope you can help me extract information related to the dataset(s) and provide the answers in the following JSON format. If there are multiple datasets, please provide the information for each one separately in JSON format.
{{
    "arxiv id": "xxx",
    "title": "xxx",
    "dataset name": "xxx",
    "dataset summary": "xxx",
    "task": "xxx",
    "data type": "xxx",
    "location": "xxx",
    "time": "xxx",
    "scale": "xxx",
    "dataset citation": ""(like Larson et al., 2016), 
    "dataset provider": "xxx",
    "dataset url": "xxx",
    "dataset publicly available": "xxx",
    "other useful information about this dataset": "xxx"
}}
You should omit fields whose information you cannot extract from the provided materials (but keep the "arxiv id" and "dataset name" fields).
Paper information: {}
Attention: Sometimes a paper may involve multiple datasets, please answer one JSON for one dataset, but in a single response. For example:
{{"dataset name": "A", ...}}
{{"dataset name": "B", ...}} ... .
"#, dataset_related_info));

        let messages = vec![system_message, assistant_message, user_message];

        let completion_request = CompletionRequest::new(config, messages);
        println!("Input Token length: {}", completion_request.num_tokens(config));

        Ok(completion_request)
    }

    pub async fn get_paper_request(&self, config: &Config, input_token_sz: &usize) -> Result<PaperRequest, Box<dyn Error>> {
        println!("Starting to generate paper request for paper ID: {}", self.id);
        let completion_request = self.generate_paper_message(config, input_token_sz).await?;
        Ok(PaperRequest {
            arxiv_id: self.id.clone(),
            messages: completion_request.messages,
        })
    }
}

impl fmt::Display for Paper {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // 这里你可以根据需要自定义输出格式
        write!(f, "Paper ID: {}", self.id)
    }
}

impl ArxivObject {
    pub fn new() -> Self {
        ArxivObject {
            id: String::new(),
            submitter: String::new(),
            authors: String::new(),
            title: String::new(),
            comments: None,
            journal_ref: None,
            doi: None,
            report_no: None,
            categories: String::new(),
            license: None,
            r#abstract: String::new(),
            versions: Vec::new(),
            update_date: String::new(),
            authors_parsed: Vec::new(),
        }
    }
}
