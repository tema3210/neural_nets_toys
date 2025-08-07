use burn::data::dataset::Dataset;
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{Tensor, backend::Backend};
use serde::de::DeserializeOwned;
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::path::Path;

pub struct JsonlDataset<T> {
    items: Vec<T>,
}

#[derive(Debug, Clone)]
pub struct EncodedBatch<B: Backend> {
    pub input: Tensor<B, 2, burn::tensor::Int>,
    pub answer: Tensor<B, 2, burn::tensor::Int>,
}


impl<T: DeserializeOwned + Clone + Send + Sync + 'static> JsonlDataset<T> {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let file = File::open(path).expect("Failed to open data file");
        let reader = BufReader::new(file);
        let items = reader
            .lines()
            .map(|line| {
                serde_json::from_str(&line.expect("Failed to read line"))
                    .expect("Failed to deserialize line")
            })
            .collect();

        Self { items }
    }
}

impl<T: DeserializeOwned + Clone + Send + Sync + 'static> Dataset<T> for JsonlDataset<T> {
    fn get(&self, index: usize) -> Option<T> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

pub struct MyBatcher {
    max_seq_length: usize,
}

impl MyBatcher {
    pub fn new(max_seq_length: usize) -> Self {
        Self { max_seq_length }
    }
}

impl<B: Backend> Batcher<B, DataSample, EncodedBatch<B>> for MyBatcher {
    fn batch(&self, items: Vec<DataSample>, device: &B::Device) -> EncodedBatch<B> {
        let batch_size = items.len();

        let mut inputs = Vec::new();
        let mut target = Vec::new();

        for item in items {
            let text = item.get_text().unwrap_or("");
            let mut input_bytes = text.as_bytes().to_vec();
            input_bytes.resize(self.max_seq_length, 0); // Pad or truncate
            inputs.extend(input_bytes.into_iter().map(|b| b as i32));


            let text = item.get_text().unwrap_or("");
            let mut answer_bytes = text.as_bytes().to_vec();
            answer_bytes.resize(self.max_seq_length, 0); // Pad or truncate
            target.extend(answer_bytes.into_iter().map(|b| b as i32));
        }

        let input_tensor = Tensor::<B, 1, burn::tensor::Int>::from_data(inputs.as_slice(), device)
            .reshape([batch_size, self.max_seq_length]);
        let answer_tensor = Tensor::<B, 2, burn::tensor::Int>::from_data(target.as_slice(), device)
            .reshape([batch_size, self.max_seq_length]);

        EncodedBatch {
            input: input_tensor,
            answer: answer_tensor,
        }
    }
}
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// General dataset sample that can handle multiple data formats
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DataSample {
    CurriculumLearning(CurriculumLearningSample),
    Instruction(InstructionSample),
    Classification(ClassificationSample),
    Evaluation(EvaluationSample),
    Glue(GlueSample),
}

/// Curriculum learning data sample with text and training metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumLearningSample {
    pub text: String,
    pub hash: String,
    pub set_split: String,
    pub loss: f64,
    pub token_num: i32,
    pub hist: Vec<i32>,
    pub average_sentence_tokens: f64,
}

/// Instruction-based training sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionSample {
    pub input: String,
    pub instruction: String,
    pub output: String,
    pub hash: String,
}

/// General classification task sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationSample {
    pub text: String,
    pub label: i32,
    pub idx: i32,
    #[serde(flatten)]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

/// Evaluation/question sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationSample {
    pub question: String,
    pub source: String,
    pub hash: String,
    #[serde(flatten)]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

/// GLUE benchmark specific samples
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "task")]
pub enum GlueSample {
    #[serde(rename = "cola")]
    CoLA(CoLASample),
    #[serde(rename = "mnli")]
    MNLI(MNLISample),
    #[serde(rename = "mrpc")]
    MRPC(MRPCSample),
    #[serde(rename = "qnli")]
    QNLI(QNLISample),
    #[serde(rename = "qqp")]
    QQP(QQPSample),
    #[serde(rename = "rte")]
    RTE(RTESample),
    #[serde(rename = "sst2")]
    SST2(SST2Sample),
    #[serde(rename = "stsb")]
    STSB(STSBSample),
    #[serde(rename = "wnli")]
    WNLI(WNLISample),
}

/// CoLA (Corpus of Linguistic Acceptability) - grammatical acceptability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoLASample {
    pub sentence: String,
    pub label: i32, // 0 = unacceptable, 1 = acceptable
    pub idx: i32,
}

/// MNLI (Multi-Genre Natural Language Inference) - textual entailment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MNLISample {
    pub premise: String,
    pub hypothesis: String,
    pub label: i32, // 0 = entailment, 1 = neutral, 2 = contradiction
    pub idx: i32,
}

/// MRPC (Microsoft Research Paraphrase Corpus) - paraphrase detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MRPCSample {
    pub sentence1: String,
    pub sentence2: String,
    pub label: i32, // 0 = not paraphrases, 1 = paraphrases
    pub idx: i32,
}

/// QNLI (Question Natural Language Inference) - question answering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QNLISample {
    pub question: String,
    pub sentence: String,
    pub label: i32, // 0 = entailment, 1 = not entailment
    pub idx: i32,
}

/// QQP (Quora Question Pairs) - question similarity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QQPSample {
    pub question1: String,
    pub question2: String,
    pub label: i32, // 0 = not duplicate, 1 = duplicate
    pub idx: i32,
}

/// RTE (Recognizing Textual Entailment) - textual entailment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RTESample {
    pub sentence1: String,
    pub sentence2: String,
    pub label: i32, // 0 = entailment, 1 = not entailment
    pub idx: i32,
}

/// SST-2 (Stanford Sentiment Treebank) - sentiment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SST2Sample {
    pub sentence: String,
    pub label: i32, // 0 = negative, 1 = positive
    pub idx: i32,
}

/// STS-B (Semantic Textual Similarity Benchmark) - semantic similarity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STSBSample {
    pub sentence1: String,
    pub sentence2: String,
    pub label: f32, // similarity score 0.0-5.0
    pub idx: i32,
}

/// WNLI (Winograd Natural Language Inference) - coreference resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WNLISample {
    pub sentence1: String,
    pub sentence2: String,
    pub label: i32, // 0 = not entailment, 1 = entailment
    pub idx: i32,
}

/// Configuration for data loading
#[derive(Debug, Clone)]
pub struct DataConfig {
    pub batch_size: usize,
    pub max_seq_length: usize,
    pub shuffle: bool,
    pub drop_last: bool,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            max_seq_length: 512,
            shuffle: true,
            drop_last: false,
        }
    }
}

/// Dataset split types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Split {
    Train,
    Validation,
    Test,
}

impl std::fmt::Display for Split {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Split::Train => write!(f, "train"),
            Split::Validation => write!(f, "validation"),
            Split::Test => write!(f, "test"),
        }
    }
}

/// Helper traits for common data operations
pub trait DataSampleExt {
    fn get_text(&self) -> Option<&str>;
    fn get_label(&self) -> Option<i32>;
    fn get_hash(&self) -> Option<&str>;
}

impl DataSampleExt for DataSample {
    fn get_text(&self) -> Option<&str> {
        match self {
            DataSample::CurriculumLearning(sample) => Some(&sample.text),
            DataSample::Instruction(sample) => Some(&sample.instruction),
            DataSample::Classification(sample) => Some(&sample.text),
            DataSample::Evaluation(sample) => Some(&sample.question),
            DataSample::Glue(sample) => match sample {
                GlueSample::CoLA(s) => Some(&s.sentence),
                GlueSample::MNLI(s) => Some(&s.premise),
                GlueSample::MRPC(s) => Some(&s.sentence1),
                GlueSample::QNLI(s) => Some(&s.question),
                GlueSample::QQP(s) => Some(&s.question1),
                GlueSample::RTE(s) => Some(&s.sentence1),
                GlueSample::SST2(s) => Some(&s.sentence),
                GlueSample::STSB(s) => Some(&s.sentence1),
                GlueSample::WNLI(s) => Some(&s.sentence1),
            },
        }
    }

    fn get_label(&self) -> Option<i32> {
        match self {
            DataSample::Classification(sample) => Some(sample.label),
            DataSample::Glue(sample) => match sample {
                GlueSample::CoLA(s) => Some(s.label),
                GlueSample::MNLI(s) => Some(s.label),
                GlueSample::MRPC(s) => Some(s.label),
                GlueSample::QNLI(s) => Some(s.label),
                GlueSample::QQP(s) => Some(s.label),
                GlueSample::RTE(s) => Some(s.label),
                GlueSample::SST2(s) => Some(s.label),
                GlueSample::STSB(_) => None, // STS-B has float labels
                GlueSample::WNLI(s) => Some(s.label),
            },
            _ => None,
        }
    }

    fn get_hash(&self) -> Option<&str> {
        match self {
            DataSample::CurriculumLearning(sample) => Some(&sample.hash),
            DataSample::Instruction(sample) => Some(&sample.hash),
            DataSample::Evaluation(sample) => Some(&sample.hash),
            _ => None,
        }
    }
}

/// Data processing utilities
pub mod processing {
    use super::*;
    
    /// Convert text to byte sequence for model input
    pub fn text_to_bytes(text: &str, max_length: usize) -> Vec<u8> {
        let mut bytes = text.as_bytes().to_vec();
        
        // Truncate or pad to max_length
        if bytes.len() > max_length {
            bytes.truncate(max_length);
        } else {
            bytes.resize(max_length, 0); // pad with zeros
        }
        
        bytes
    }
    
    /// Tokenize text into simple whitespace tokens
    pub fn simple_tokenize(text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| s.to_lowercase())
            .collect()
    }
    
    /// Build vocabulary from text samples
    pub fn build_vocab(texts: &[&str]) -> HashMap<String, usize> {
        let mut vocab = HashMap::new();
        let mut idx = 0;
        
        // Add special tokens
        vocab.insert("<pad>".to_string(), idx); idx += 1;
        vocab.insert("<unk>".to_string(), idx); idx += 1;
        vocab.insert("<sos>".to_string(), idx); idx += 1;
        vocab.insert("<eos>".to_string(), idx); idx += 1;
        
        // Add words from corpus
        for text in texts {
            for token in simple_tokenize(text) {
                if !vocab.contains_key(&token) {
                    vocab.insert(token, idx);
                    idx += 1;
                }
            }
        }
        
        vocab
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_curriculum_learning_deserialization() {
        let json = r#"{"text": "Hello world", "hash": "abc123", "set_split": "train", "loss": 1.5, "token_num": 100, "hist": [1,2,3], "average_sentence_tokens": 10.5}"#;
        let sample: CurriculumLearningSample = serde_json::from_str(json).unwrap();
        assert_eq!(sample.text, "Hello world");
        assert_eq!(sample.loss, 1.5);
    }
    
    #[test]
    fn test_instruction_deserialization() {
        let json = r#"{"input": "test input", "instruction": "Do something", "output": "result", "hash": "def456"}"#;
        let sample: InstructionSample = serde_json::from_str(json).unwrap();
        assert_eq!(sample.instruction, "Do something");
    }
    
    #[test]
    fn test_cola_deserialization() {
        let json = r#"{"sentence": "This is good.", "label": 1, "idx": 0}"#;
        let sample: CoLASample = serde_json::from_str(json).unwrap();
        assert_eq!(sample.label, 1);
    }
}
