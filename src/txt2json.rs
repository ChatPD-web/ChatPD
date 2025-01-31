use regex::Regex;
use std::fs::{self, File};
use std::io::{self, BufRead, Write};

fn main() -> io::Result<()> {

    // models = ['gpt-3.5-turbo-0125', 'deepseek-chat', 'gemini-1.5-flash-latest', 'GLM-3-Turbo', 'glm-4']
    // let input_path = "data/ie_evaluation/llm_ie_evaluation/release_200/output_paper_dataset_1500_token_results_glm-4.txt";
    // let input_path = "data/ie_evaluation/target_ie_evaluation/complete_request/arxiv_5w_1500_token_results_gpt-4o-mini.txt";
    // let input_path = "data/ie_evaluation/target_ie_evaluation/complete_request/kdd_response_1500_token_gpt-4o-mini.txt";
    let update_month = "2024-12";
    let input_path = format!("data/update_data/{}_arxiv_papers_request_pure_response.ndjson", update_month);
    // let input_path = "data/update_data/2024-02_arxiv_papers_request_response.ndjson";

    let output_path = input_path.replace(".ndjson", ".json");
    // let output_path = "data/ie_evaluation/llm_ie_evaluation/release_200/output_paper_dataset_1500_token_results_gpt-3.5-turbo-0125.json";
    

    let file = File::open(input_path)?;
    let reader = io::BufReader::new(file);

    let mut lines: Vec<String> = Vec::new();

    for line in reader.lines() {
        let mut line = line?.trim_start().to_string(); // 移除每行开头的空格
        // 仅保留以指定字符开头的行
        if line.starts_with('{') || line.starts_with('}') || line.starts_with('"') || line.starts_with('[') || line.starts_with(']') {
            // 检查行中是否有冒号
            if line.clone().contains(":") {
            if let Some((key, value)) = line.split_once(':') {
                    let mut new_value = String::new();
                    let chars: Vec<char> = value.chars().collect();
                    let mut is_first_quote = true;
                    let mut last_quote_index = None;

                    // 首先找到最后一个非转义的双引号的位置
                    for (i, &c) in chars.iter().enumerate().rev() {
                        if c == '"' && chars.get(i.wrapping_sub(1)) != Some(&'\\') {
                            last_quote_index = Some(i);
                            break;
                        }
                    }

                    for (i, &c) in chars.iter().enumerate() {
                        if c == '"' && (i == 0 || chars[i - 1] != '\\') {
                            if is_first_quote {
                                // 第一个双引号，直接添加
                                is_first_quote = false;
                            } else if Some(i) != last_quote_index {
                                // 非最后一个双引号，前面添加反斜线
                                new_value.push('\\');
                            }
                        }
                        new_value.push(c);
                    }
                    line = format!("{}: {}", key, new_value);
                }
            }
            lines.push(line);
        }
    }

    // 处理每个 JSON 对象末尾不需要的逗号
    let mut new_lines = Vec::new();
    let mut iter = lines.iter().peekable();
    while let Some(line) = iter.next() {
        // 如果当前行以"开头，且下一行是"}"
        if line.trim().starts_with('"') && iter.peek().map_or(false, |&next_line| next_line.trim() == "}") {
            // 删除末尾的逗号
            if line.trim_end().ends_with(',') {
                new_lines.push(line.trim_end_matches(',').to_string());
                continue;
            }
        }
        new_lines.push(line.to_string());
    }

    // 将所有的"}"替换成"},"
    let mut content = new_lines.join("\n").replace("}", "},");
    // 将最后一个","替换为"}"
    content = content.trim_end_matches(',').to_string();
    // 将所有的连续2个,,替换为,
    let re = Regex::new(r",,").unwrap();
    content = re.replace_all(&content, ",").to_string();
    // 将所有只有[或者]的行删除
    let re_single_brackets = Regex::new(r"^\[|\]$").unwrap();
    let filtered_lines: Vec<&str> = content
        .lines()
        .filter(|line| !re_single_brackets.is_match(line.trim()))
        .collect();
    content = filtered_lines.join("\n");


    //  第一步：将所有的 '\' 替换为 '\\'
    content = content.replace("\\", "\\\\");
    // 第二步：检查每个 '"' 前的反斜杠，只保留一个
    content = content.replace("\\\\\"", "\\\"");
    // 第三步：将不在""内的//及其后面的内容删除
    let re = Regex::new(r#""(?:\\.|[^"\\])*"|//.*"#).unwrap();
    content = re.replace_all(&content, |caps: &regex::Captures| {
        if caps[0].starts_with('"') {
            caps[0].to_string()
        } else {
            "".to_string()
        }
    }).to_string();


    // 写入新的 JSON 文件
    let mut output_file = File::create(&output_path)?;
    output_file.write_all(content.as_bytes())?;

    // 在内容的开头插入 "[" 并在结尾插入 "]"
    content = format!("[\n{}\n]", content);

    // 写入新的 JSON 文件
    let mut output_file = File::create(&output_path)?;
    output_file.write_all(content.as_bytes())?;
    println!("Successfully convert txt to json: {}", output_path);

    Ok(())
}
