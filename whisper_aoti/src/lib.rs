use anyhow::anyhow;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use itertools::izip;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3_tch::PyTensor;
use std::cmp::max;
use std::collections::HashSet;
use std::sync::LazyLock;

use tch::Tensor;

use indexmap::IndexMap;
use rustc_hash::FxHashMap as HashMap;
use tiktoken_rs::CoreBPE;

// =========================================================
// Tokenizer
// =========================================================

static LANGUAGES: LazyLock<IndexMap<&str, &str>> = LazyLock::new(|| {
    let mut m = IndexMap::default();
    m.insert("en", "english");
    m.insert("zh", "chinese");
    m.insert("de", "german");
    m.insert("es", "spanish");
    m.insert("ru", "russian");
    m.insert("ko", "korean");
    m.insert("fr", "french");
    m.insert("ja", "japanese");
    m.insert("pt", "portuguese");
    m.insert("tr", "turkish");
    m.insert("pl", "polish");
    m.insert("ca", "catalan");
    m.insert("nl", "dutch");
    m.insert("ar", "arabic");
    m.insert("sv", "swedish");
    m.insert("it", "italian");
    m.insert("id", "indonesian");
    m.insert("hi", "hindi");
    m.insert("fi", "finnish");
    m.insert("vi", "vietnamese");
    m.insert("he", "hebrew");
    m.insert("uk", "ukrainian");
    m.insert("el", "greek");
    m.insert("ms", "malay");
    m.insert("cs", "czech");
    m.insert("ro", "romanian");
    m.insert("da", "danish");
    m.insert("hu", "hungarian");
    m.insert("ta", "tamil");
    m.insert("no", "norwegian");
    m.insert("th", "thai");
    m.insert("ur", "urdu");
    m.insert("hr", "croatian");
    m.insert("bg", "bulgarian");
    m.insert("lt", "lithuanian");
    m.insert("la", "latin");
    m.insert("mi", "maori");
    m.insert("ml", "malayalam");
    m.insert("cy", "welsh");
    m.insert("sk", "slovak");
    m.insert("te", "telugu");
    m.insert("fa", "persian");
    m.insert("lv", "latvian");
    m.insert("bn", "bengali");
    m.insert("sr", "serbian");
    m.insert("az", "azerbaijani");
    m.insert("sl", "slovenian");
    m.insert("kn", "kannada");
    m.insert("et", "estonian");
    m.insert("mk", "macedonian");
    m.insert("br", "breton");
    m.insert("eu", "basque");
    m.insert("is", "icelandic");
    m.insert("hy", "armenian");
    m.insert("ne", "nepali");
    m.insert("mn", "mongolian");
    m.insert("bs", "bosnian");
    m.insert("kk", "kazakh");
    m.insert("sq", "albanian");
    m.insert("sw", "swahili");
    m.insert("gl", "galician");
    m.insert("mr", "marathi");
    m.insert("pa", "punjabi");
    m.insert("si", "sinhala");
    m.insert("km", "khmer");
    m.insert("sn", "shona");
    m.insert("yo", "yoruba");
    m.insert("so", "somali");
    m.insert("af", "afrikaans");
    m.insert("oc", "occitan");
    m.insert("ka", "georgian");
    m.insert("be", "belarusian");
    m.insert("tg", "tajik");
    m.insert("sd", "sindhi");
    m.insert("gu", "gujarati");
    m.insert("am", "amharic");
    m.insert("yi", "yiddish");
    m.insert("lo", "lao");
    m.insert("uz", "uzbek");
    m.insert("fo", "faroese");
    m.insert("ht", "haitian creole");
    m.insert("ps", "pashto");
    m.insert("tk", "turkmen");
    m.insert("nn", "nynorsk");
    m.insert("mt", "maltese");
    m.insert("sa", "sanskrit");
    m.insert("lb", "luxembourgish");
    m.insert("my", "myanmar");
    m.insert("bo", "tibetan");
    m.insert("tl", "tagalog");
    m.insert("mg", "malagasy");
    m.insert("as", "assamese");
    m.insert("tt", "tatar");
    m.insert("haw", "hawaiian");
    m.insert("ln", "lingala");
    m.insert("ha", "hausa");
    m.insert("ba", "bashkir");
    m.insert("jw", "javanese");
    m.insert("su", "sundanese");
    m.insert("yue", "cantonese");
    m
});

#[pyclass]
#[derive(Clone)]
struct Encoding {
    name: String,
    _core_bpe: CoreBPE,
    _special_tokens: HashMap<String, usize>,
    _pat_str: String,
    _mergeable_ranks: HashMap<Vec<u8>, usize>,
    _explicit_n_vocab: usize,
    max_token_value: usize,
}

#[pymethods]
impl Encoding {
    #[new]
    fn new(
        name: &str,
        pat_str: &str,
        mergeable_ranks: HashMap<Vec<u8>, usize>,
        special_tokens: HashMap<String, usize>,
        explicit_n_vocab: usize,
    ) -> anyhow::Result<Self> {
        Ok(Encoding {
            name: name.to_string(),
            _core_bpe: CoreBPE::new(mergeable_ranks.clone(), special_tokens.clone(), pat_str)?,
            _special_tokens: special_tokens.clone(),
            _pat_str: pat_str.to_string(),
            _mergeable_ranks: mergeable_ranks.clone(),
            _explicit_n_vocab: explicit_n_vocab,
            max_token_value: max(
                *mergeable_ranks.values().max().unwrap(),
                *special_tokens.values().max().unwrap_or(&0),
            ),
        })
    }

    fn decode(&self, tokens: Vec<usize>) -> anyhow::Result<String> {
        self._core_bpe.decode(tokens)
    }

    fn encode(&self, text: String) -> anyhow::Result<Vec<usize>> {
        Ok(self._core_bpe.encode(&text, HashSet::new()))
    }

    fn encode_single_token(&self, piece: &[u8]) -> anyhow::Result<usize> {
        if let Some(token) = self._mergeable_ranks.get(piece).copied() {
            return Ok(token);
        }
        if let Ok(piece_str) = std::str::from_utf8(piece) {
            if let Some(token) = self._special_tokens.get(piece_str).copied() {
                return Ok(token);
            }
        }
        Err(anyhow!("encode_single_token failed"))
    }

    #[getter]
    fn eot_token(&self) -> usize {
        self._special_tokens["<|endoftext|>"]
    }
}

fn get_encoding(name: &str, num_languages: usize) -> anyhow::Result<Encoding> {
    assert!(name == "multilingual");
    let multilingual_vocab = include_bytes!("multilingual.tiktoken");
    let ranks: HashMap<Vec<u8>, usize> = String::from_utf8_lossy(multilingual_vocab)
        .lines()
        .map(|s| -> (Vec<u8>, usize) {
            let (a, b) = s.split_once(" ").unwrap();
            (
                if a != "=" {
                    BASE64_STANDARD.decode(a).unwrap()
                } else {
                    vec![]
                },
                b.parse().unwrap(),
            )
        })
        .collect();
    let mut n_vocab = ranks.len();
    let mut special_tokens = HashMap::default();
    let mut specials = vec![
        "<|endoftext|>".to_string(),
        "<|startoftranscript|>".to_string(),
    ];
    for (k, _) in LANGUAGES.iter().take(num_languages) {
        specials.push(format!("<|{}|>", k));
    }
    specials.push("<|translate|>".to_string());
    specials.push("<|transcribe|>".to_string());
    specials.push("<|startoflm|>".to_string());
    specials.push("<|startofprev|>".to_string());
    specials.push("<|nospeech|>".to_string());
    specials.push("<|notimestamps|>".to_string());
    for i in 0..1501 {
        specials.push(format!("<|{:.2}|>", f64::from(i) * 0.02));
    }

    for token in &specials {
        special_tokens.insert(token.clone(), n_vocab);
        n_vocab += 1;
    }
    Encoding::new(
        name,
        r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#,
        ranks,
        special_tokens,
        n_vocab,
    )
}

#[pyclass]
#[derive(Clone)]
struct Tokenizer {
    encoding: Encoding,
    num_languages: usize,
    #[pyo3(get)]
    language: String,
    task: String,
    #[pyo3(get)]
    sot_sequence: Vec<usize>,
    special_tokens: HashMap<String, usize>,
}

impl Tokenizer {
    fn new(
        encoding: Encoding,
        num_languages: usize,
        language: &str,
        task: &str,
        special_tokens: HashMap<String, usize>,
    ) -> Self {
        let mut tokenizer = Tokenizer {
            encoding,
            num_languages,
            language: language.to_string(),
            task: task.to_string(),
            sot_sequence: vec![],
            special_tokens,
        };

        for special in tokenizer.encoding._special_tokens.keys() {
            let special_token = tokenizer
                .encoding
                .encode_single_token(special.as_bytes())
                .unwrap();
            tokenizer
                .special_tokens
                .insert(special.clone(), special_token);
        }

        let sot = tokenizer.special_tokens["<|startoftranscript|>"];

        let transcribe = tokenizer.special_tokens["<|transcribe|>"];
        let langs: Vec<&str> = LANGUAGES
            .keys()
            .take(tokenizer.num_languages)
            .map(|x| *x)
            .collect();
        let mut sot_sequence = vec![sot];
        sot_sequence.push(
            sot + 1
                + langs
                    .iter()
                    .position(|&x| x == tokenizer.language.as_str())
                    .unwrap(),
        );
        let task_token = transcribe;
        sot_sequence.push(task_token);
        tokenizer.sot_sequence = sot_sequence;
        tokenizer
    }
}

#[pymethods]
impl Tokenizer {
    #[getter]
    fn language_token(&self) -> usize {
        self.special_tokens[format!("<|{}|>", self.language).as_str()]
    }

    #[getter]
    fn sot(&self) -> usize {
        self.special_tokens["<|startoftranscript|>"]
    }

    #[getter]
    fn eot(&self) -> usize {
        self.encoding.eot_token()
    }

    #[getter]
    fn no_speech(&self) -> usize {
        self.special_tokens["<|nospeech|>"]
    }

    #[getter]
    fn transcribe(&self) -> usize {
        self.special_tokens["<|transcribe|>"]
    }

    #[getter]
    fn translate(&self) -> usize {
        self.special_tokens["<|translate|>"]
    }

    #[getter]
    fn sot_prev(&self) -> usize {
        self.special_tokens["<|startofprev|>"]
    }

    #[getter]
    fn sot_lm(&self) -> usize {
        self.special_tokens["<|startoflm|>"]
    }

    #[getter]
    fn no_timestamps(&self) -> usize {
        self.special_tokens["<|notimestamps|>"]
    }

    #[getter]
    fn all_language_tokens(&self) -> Vec<usize> {
        self.special_tokens
            .iter()
            .filter_map(|(token, &token_id)| {
                if LANGUAGES.contains_key(token.trim_matches(&['<', '|', '>'])) {
                    Some(token_id)
                } else {
                    None
                }
            })
            .take(self.num_languages)
            .collect()
    }

    #[getter]
    fn all_language_codes(&self) -> Vec<String> {
        self.all_language_tokens()
            .iter()
            .map(|&_l| {
                self.encoding
                    .decode(vec![_l])
                    .unwrap()
                    .trim_matches(&['<', '|', '>'])
                    .to_string()
            })
            .collect()
    }

    #[getter]
    fn timestamp_begin(&self) -> usize {
        self.special_tokens["<|0.00|>"]
    }

    fn decode(&self, token_ids: Vec<usize>) -> anyhow::Result<String> {
        let token_ids = token_ids
            .into_iter()
            .filter(|&t| t < self.timestamp_begin())
            .collect();
        self.encoding.decode(token_ids)
    }

    fn encode(&self, text: String) -> anyhow::Result<Vec<usize>> {
        self.encoding.encode(text)
    }

    #[getter]
    fn non_speech_tokens(&self) -> anyhow::Result<Vec<usize>> {
        let mut symbols: Vec<String> = "\"#()*+/:;<=>@[\\]^_`{|}~「」『』;"
            .chars()
            .map(|x| x.to_string())
            .collect();
        symbols.extend(
            "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪"
                .split_whitespace()
                .map(|x| x.to_string()),
        );
        let miscellaneous = "♩♪♫♬♭♮♯"
            .chars()
            .map(|x| x.to_string())
            .collect::<HashSet<String>>();
        // assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)
        let mut result = vec![
            self.encoding.encode(" -".to_string())?[0],
            self.encoding.encode(" '".to_string())?[0],
        ];
        for symbol in symbols.iter().chain(miscellaneous.iter()) {
            let tokens = self.encoding.encode(symbol.clone())?;
            if tokens.len() == 1 || miscellaneous.contains(symbol) {
                result.push(tokens[0])
            }
            let tokens = self.encoding.encode(" ".to_string() + symbol)?;
            if tokens.len() == 1 || miscellaneous.contains(symbol) {
                result.push(tokens[0])
            }
        }
        result.sort();
        Ok(result)
    }
}

#[pyfunction]
fn get_tokenizer() -> anyhow::Result<Tokenizer> {
    let encoding_name = "multilingual";
    let language = "en";
    let task = "transcribe";
    let num_languages = 99;

    let encoding = get_encoding(encoding_name, num_languages)?;
    Ok(Tokenizer::new(
        encoding,
        num_languages,
        language,
        task,
        HashMap::default(),
    ))
}

// =========================================================
// LogitFilter
// =========================================================

trait LogitFilter {
    fn _apply(&self, logits: PyTensor, tokens: PyTensor) -> anyhow::Result<()>;
}

#[pyclass]
struct SuppressBlank {
    tokenizer: Tokenizer,
    sample_begin: i64,
}

#[pymethods]
impl SuppressBlank {
    #[new]
    fn new(tokenizer: Tokenizer, sample_begin: i64) -> Self {
        SuppressBlank {
            tokenizer,
            sample_begin,
        }
    }

    fn apply(&self, mut logits: PyTensor, tokens: PyTensor) -> anyhow::Result<()> {
        self._apply(logits, tokens)
    }
}

impl LogitFilter for SuppressBlank {
    fn _apply(&self, mut logits: PyTensor, tokens: PyTensor) -> anyhow::Result<()> {
        Ok(if tokens.size()[0] == self.sample_begin {
            let mut indices = self.tokenizer.encode(" ".to_string())?;
            indices.push(self.tokenizer.eot());
            let indices: Vec<i64> = indices.iter().map(|&x| i64::try_from(x).unwrap()).collect();
            let _ = logits.0.index_fill_(
                1,
                &Tensor::from_slice(indices.as_slice()).to(logits.device()),
                f64::NEG_INFINITY,
            );
        })
    }
}

#[pyclass]
struct SuppressTokens {
    suppress_tokens: Vec<usize>,
}

#[pymethods]
impl SuppressTokens {
    #[new]
    fn new(suppress_tokens: Vec<usize>) -> Self {
        SuppressTokens { suppress_tokens }
    }

    fn apply(&self, mut logits: PyTensor, tokens: PyTensor) -> anyhow::Result<()> {
        self._apply(logits, tokens)
    }
}

impl LogitFilter for SuppressTokens {
    fn _apply(&self, mut logits: PyTensor, tokens: PyTensor) -> anyhow::Result<()> {
        let indices: Vec<i64> = self
            .suppress_tokens
            .iter()
            .map(|&x| i64::try_from(x).unwrap())
            .collect();
        let _ = logits.0.index_fill_(
            1,
            &Tensor::from_slice(indices.as_slice()).to(logits.device()),
            f64::NEG_INFINITY,
        );
        Ok(())
    }
}

#[pyclass]
struct ApplyTimestampRules {
    tokenizer: Tokenizer,
    sample_begin: i64,
    max_initial_timestamp_index: Option<i64>,
}

#[pymethods]
impl ApplyTimestampRules {
    #[new]
    fn new(
        tokenizer: Tokenizer,
        sample_begin: i64,
        max_initial_timestamp_index: Option<i64>,
    ) -> Self {
        ApplyTimestampRules {
            tokenizer,
            sample_begin,
            max_initial_timestamp_index,
        }
    }

    fn apply(&self, mut logits: PyTensor, tokens: PyTensor) -> anyhow::Result<()> {
        self._apply(logits, tokens)
    }
}

impl LogitFilter for ApplyTimestampRules {
    fn _apply(&self, mut logits: PyTensor, tokens: PyTensor) -> anyhow::Result<()> {
        let _ = logits.0.index_fill_(
            1,
            &Tensor::from_slice(&[i64::try_from(self.tokenizer.no_timestamps()).unwrap()])
                .to(logits.device()),
            f64::NEG_INFINITY,
        );
        let timestamp_begin = i64::try_from(self.tokenizer.timestamp_begin()).unwrap();
        for k in 0..tokens.size()[0] {
            let sampled_tokens = tokens.select(0, k).slice(0, self.sample_begin, None, 1);
            let mut seq: Vec<i64> = vec![0; sampled_tokens.numel()];
            sampled_tokens.copy_data(&mut seq, sampled_tokens.numel());
            let last_was_timestamp = seq.len() >= 1 && *seq.last().unwrap() >= timestamp_begin;
            let penultimate_was_timestamp = seq.len() < 2 || seq[seq.len() - 2] >= timestamp_begin;

            if last_was_timestamp {
                if penultimate_was_timestamp {
                    let _ = logits
                        .select(0, k)
                        .slice(0, timestamp_begin, None, 1)
                        .fill_(f64::NEG_INFINITY);
                } else {
                    let _ = logits.select(0, k).slice(
                        0,
                        None,
                        i64::try_from(self.tokenizer.eot()).unwrap(),
                        1,
                    );
                }
            }

            let timestamps = sampled_tokens.index(&[Some(sampled_tokens.ge(timestamp_begin))]);
            if timestamps.numel() > 0 {
                let timestamp_last = if last_was_timestamp && !penultimate_was_timestamp {
                    timestamps.int64_value(&[-1])
                } else {
                    timestamps.int64_value(&[-1]) + 1
                };
                let _ = logits
                    .select(0, k)
                    .slice(0, timestamp_begin, timestamp_last, 1)
                    .fill_(f64::NEG_INFINITY);
            }
        }

        if tokens.size()[1] == self.sample_begin {
            let _ = logits
                .slice(1, None, timestamp_begin, 1)
                .fill_(f64::NEG_INFINITY);

            if let Some(max_initial_timestamp_index) = self.max_initial_timestamp_index {
                let last_allowed = timestamp_begin + max_initial_timestamp_index;
                let _ = logits
                    .slice(1, last_allowed + 1, None, 1)
                    .fill_(f64::NEG_INFINITY);
            }
        }
        let logprobs = logits.log_softmax(-1, tch::Kind::Float);
        for k in 0..tokens.size()[0] {
            let timestamp_logprob = logprobs
                .select(0, k)
                .slice(0, timestamp_begin, None, 1)
                .logsumexp(-1, false);
            let max_text_token_logprob = logprobs
                .select(0, k)
                .slice(0, None, timestamp_begin, 1)
                .max();
            if timestamp_logprob.double_value(&[]) > max_text_token_logprob.double_value(&[]) {
                let _ = logits
                    .select(0, k)
                    .slice(0, None, timestamp_begin, 1)
                    .fill_(f64::NEG_INFINITY);
            }
        }
        Ok(())
    }
}

// =========================================================
// TokenDecoder
// =========================================================

#[pyclass]
struct GreedyDecoder {
    temperature: f64,
    eot: usize,
}

#[pymethods]
impl GreedyDecoder {
    #[new]
    fn new(temperature: f64, eot: usize) -> Self {
        GreedyDecoder { temperature, eot }
    }

    fn update(
        &self,
        tokens: PyTensor,
        logits: PyTensor,
        mut sum_logprobs: PyTensor,
    ) -> (PyTensor, PyTensor) {
        assert!(self.temperature == 0.);
        let mut next_tokens = logits.argmax(-1, false);
        let logprobs = logits.log_softmax(-1, tch::Kind::Float);
        let current_logprobs = logprobs.index(&[
            Some(Tensor::arange(
                logprobs.size()[0],
                (tch::Kind::Int, tch::Device::Cpu),
            )),
            Some(next_tokens.shallow_clone()),
        ]);
        let eot = i64::try_from(self.eot).unwrap();
        let _ = sum_logprobs
            .0
            .g_add_(&(current_logprobs.g_mul(&tokens.select(1, -1).not_equal(eot))));
        let _ = next_tokens.index_put_(
            &[Some(tokens.select(1, -1).eq(eot))],
            &Tensor::scalar_tensor(eot, (tch::Kind::Int64, tch::Device::Cpu)),
            false,
        );
        let tokens = tch::Tensor::cat(&[tokens.0, next_tokens.unsqueeze(1)], -1);
        let completed = (tokens.select(1, -1).eq(eot)).all();
        (PyTensor(tokens), PyTensor(completed))
    }

    fn finalize(&self, tokens: PyTensor, sum_logprobs: PyTensor) -> (PyTensor, Vec<Vec<f32>>) {
        let tokens = tokens.pad(&[0, 1], "constant", self.eot as f64);
        let (x, y) = sum_logprobs.size2().unwrap();
        (
            PyTensor(tokens),
            (0..x)
                .into_iter()
                .map(|i| {
                    let ysize = usize::try_from(y).unwrap();
                    let mut res: Vec<f32> = vec![0.; ysize];
                    sum_logprobs.select(0, i).copy_data(&mut res, ysize);
                    res
                })
                .collect(),
        )
    }

    fn reset(&self) {}
}

// =========================================================
// SequenceRanker
// =========================================================

#[pyclass]
struct MaximumLikelihoodRanker {
    length_penalty: Option<f32>,
}

#[pymethods]
impl MaximumLikelihoodRanker {
    #[new]
    fn new(length_penalty: Option<f32>) -> Self {
        MaximumLikelihoodRanker { length_penalty }
    }
    fn rank(
        &self,
        tokens: Vec<Vec<PyTensor>>,
        sum_logprobs: Vec<Vec<f32>>,
    ) -> anyhow::Result<Vec<usize>> {
        let scores = |logprobs: Vec<f32>, lengths: Vec<i64>| {
            let mut result = vec![];
            logprobs.iter().zip(lengths).for_each(|(logprob, length)| {
                let penalty = if let Some(length_penalty) = self.length_penalty {
                    ((5. + length as f32) / 6.).powf(length_penalty)
                } else {
                    length as f32
                };
                result.push(logprob / penalty);
            });
            result
        };
        let lengths = tokens
            .iter()
            .map(|s| s.iter().map(|t| t.size()[0]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        Ok(sum_logprobs
            .into_iter()
            .zip(lengths)
            .map(|(p, l)| {
                scores(p, l)
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap()
                    .0
            })
            .collect())
    }
}

// =========================================================
// ModelDimensions
// =========================================================
#[pyclass]
#[derive(Clone)]
struct ModelDimensions {
    n_audio_ctx: i32,
    n_text_ctx: i32,
    n_audio_state: i32,
}
// ModelDimensions(
//   n_mels=80,
//   n_audio_ctx=1500,
//   n_audio_state=512,
//   n_audio_head=8,
//   n_audio_layer=6, n_vocab=51865, n_text_ctx=448, n_text_state=512, n_text_head=8, n_text_layer=6)

impl ModelDimensions {
    fn base() -> Self {
        ModelDimensions {
            n_audio_ctx: 1500,
            n_text_ctx: 448,
            n_audio_state: 512,
        }
    }
}

// =========================================================
// DecodingOptions
// =========================================================

#[pyclass]
#[derive(Clone)]
struct DecodingOptions {
    task: String,
    language: Option<String>,
    sample_len: Option<i32>,
    prefix: Option<String>,
    prompt: Option<String>,
    without_timestamps: bool,
    temperature: f64,
    length_penalty: Option<f32>,
    beam_size: Option<i64>,
    suppress_tokens: Option<Vec<i64>>,
    suppress_blank: bool,
    max_initial_timestamp: Option<f64>,
    fp16: bool,
}

impl DecodingOptions {
    fn default() -> Self {
        DecodingOptions {
            task: "transcribe".to_string(),
            language: None,
            sample_len: None,
            prompt: None,
            prefix: None,
            without_timestamps: false,
            temperature: 0.0,
            length_penalty: None,
            beam_size: None,
            suppress_tokens: Some(vec![-1]),
            suppress_blank: true,
            max_initial_timestamp: Some(1.0),
            fp16: true,
        }
    }
}

// =========================================================
// DecodingTask
// =========================================================

fn _get_initial_tokens(sot_sequence: &Vec<usize>, options: &DecodingOptions) -> Vec<usize> {
    let tokens = sot_sequence.clone();
    assert!(options.prefix.is_none());
    assert!(options.prompt.is_none());
    tokens
}

fn _get_suppress_tokens(tokenizer: &Tokenizer, options: &DecodingOptions) -> Vec<usize> {
    let mut suppress_tokens: Vec<usize> = vec![];
    if let Some(st) = &options.suppress_tokens {
        for &x in st {
            if x >= 0 {
                suppress_tokens.push(x as usize);
            } else {
                suppress_tokens.append(&mut tokenizer.non_speech_tokens().unwrap());
            }
        }
    }
    suppress_tokens.push(tokenizer.transcribe());
    suppress_tokens.push(tokenizer.translate());
    suppress_tokens.push(tokenizer.sot());
    suppress_tokens.push(tokenizer.sot_prev());
    suppress_tokens.push(tokenizer.sot_lm());

    suppress_tokens // TODO sort
}

#[pyclass]
struct DecodingTask {
    encoder: AOTIModel,
    dims: ModelDimensions,
    options: DecodingOptions,
    sequence_ranker: MaximumLikelihoodRanker,
    logit_filters: Vec<Box<dyn LogitFilter>>,
    tokenizer: Tokenizer,
    decoder: GreedyDecoder,
    n_group: i32,
    n_ctx: i32,
    sample_len: i32,
    sot_sequence: Vec<usize>,
    initial_tokens: Vec<usize>,
    sample_begin: usize,
    sot_index: usize,
    inference: PyTorchInference,
}

unsafe impl Send for DecodingTask {}

const CHUNK_LENGTH: i32 = 30;

#[pymethods]
impl DecodingTask {
    #[new]
    fn new(dims: ModelDimensions, options: DecodingOptions) -> anyhow::Result<Self> {
        // TODO verify options
        let tokenizer = get_tokenizer()?;
        let n_group = 1;
        let n_ctx = dims.n_text_ctx;
        let sample_len = if options.sample_len.is_some() {
            options.sample_len.unwrap()
        } else {
            dims.n_text_ctx / 2
        };
        let sot_sequence = tokenizer.sot_sequence.clone();
        assert!(!options.without_timestamps);
        let initial_tokens = _get_initial_tokens(&sot_sequence, &options);
        let sample_begin = initial_tokens.len();
        let sot_index = initial_tokens
            .iter()
            .position(|&r| r == tokenizer.sot())
            .unwrap();
        let inference = PyTorchInference::new(initial_tokens.len() as i64);
        let sequence_ranker = MaximumLikelihoodRanker::new(options.length_penalty);
        assert!(options.beam_size.is_none());
        let decoder = GreedyDecoder::new(options.temperature, tokenizer.eot());
        assert!(options.suppress_blank);
        assert!(options.suppress_tokens.is_some());
        assert!(!options.without_timestamps);
        let mut logit_filters: Vec<Box<dyn LogitFilter>> = vec![];
        logit_filters.push(Box::new(SuppressBlank::new(
            tokenizer.clone(),
            sample_begin as i64,
        )));
        logit_filters.push(Box::new(SuppressTokens::new(_get_suppress_tokens(
            &tokenizer, &options,
        ))));
        let precision = CHUNK_LENGTH as f64 / dims.n_audio_ctx as f64;
        let mut max_initial_timestamp_index = None;
        if let Some(max_initial_timestamp) = options.max_initial_timestamp {
            max_initial_timestamp_index =
                Some((max_initial_timestamp / precision + 0.5).floor() as i64);
        }
        logit_filters.push(Box::new(ApplyTimestampRules::new(
            tokenizer.clone(),
            sample_begin as i64,
            max_initial_timestamp_index,
        )));
        Ok(DecodingTask {
            encoder: aot_load("./generated/encoder.so".to_string(), "cpu".to_string())?,
            dims,
            options,
            sequence_ranker,
            logit_filters,
            tokenizer,
            decoder,
            n_group,
            n_ctx,
            sample_len,
            sot_sequence,
            initial_tokens,
            sample_begin,
            sot_index,
            inference,
        })
    }

    fn _get_audio_features(&self, mel: PyTensor) -> anyhow::Result<PyTensor> {
        assert!(self.options.fp16);
        let mel = mel.to_dtype(tch::Kind::Half, false, false);
        let n_audio_ctx = self.dims.n_audio_ctx as i64;
        let n_audio_state = self.dims.n_audio_state as i64;
        let shapes = mel.size();
        assert_ne!(&shapes[shapes.len() - 2..], &[n_audio_ctx, n_audio_state]);
        let audio_features = self.encoder.call(vec![PyTensor(mel)])?[0].shallow_clone();
        Ok(PyTensor(audio_features))
    }

    fn _detect_language(
        &self,
        audio_features: PyTensor,
        tokens: PyTensor,
    ) -> anyhow::Result<(Vec<String>, Vec<HashMap<String, f64>>)> {
        // FIXME
        // assert!(self.options.language.is_none());
        // let (lang_tokens, lang_probs) = self._model_detect_language(
        //     &model,
        //     audio_features,
        //     &self_.tokenizer,
        // )?;
        // let languages = lang_probs
        //     .iter()
        //     .map(|probs| {
        //         probs
        //             .iter()
        //             .max_by(|(_, x), (_, y)| x.total_cmp(y))
        //             .unwrap()
        //             .0
        //             .clone()
        //     })
        //     .collect();
        // let index = self_.sot_index + 1;
        // let _ = tokens
        //     .slice(0, None, None, 1)
        //     .select(1, index)
        //     .copy_(&lang_tokens);
        // Ok((languages, lang_probs))
        let mut h = HashMap::default();
        h.insert("en".to_string(), 1.0);
        Ok((vec!["en".to_string()], vec![h]))
    }

    // fn _model_detect_language(
    //     &self,
    //     mel: PyTensor,
    //     tokenizer: &Tokenizer,
    // ) -> anyhow::Result<(PyTensor, Vec<HashMap<String, f64>>)> {
    //     let _no_grad_guard = tch::no_grad_guard();
    //     assert_eq!(mel.size().len(), 3);
    //     let n_audio_ctx = self.dims.n_audio_ctx as i64;
    //     let n_audio_state = self.dims.n_audio_state as i64;
    //     let shapes = mel.size();
    //     assert_eq!(&shapes[shapes.len() - 2..], &[n_audio_ctx, n_audio_state]);
    //     let n_audio = mel.size()[0];
    //     let x = Tensor::from_slice(&[i64::try_from(tokenizer.sot()).unwrap()])
    //         .repeat(&[n_audio, 1])
    //         .to(mel.device());
    //     let mut logits = model // FIXME
    //         .getattr("logits")?
    //         .call((PyTensor(x), mel), None)?
    //         .extract::<PyTensor>()?
    //         .select(1, 0);
    //     let mut mask = Tensor::ones(
    //         &[*logits.size().last().unwrap()],
    //         (tch::Kind::Bool, tch::Device::Cpu),
    //     );
    //     let _ = mask.index_put_(
    //         &[Some(Tensor::from_slice(
    //             tokenizer
    //                 .all_language_tokens()
    //                 .iter()
    //                 .map(|&x| i64::try_from(x).unwrap())
    //                 .collect::<Vec<_>>()
    //                 .as_slice(),
    //         ))],
    //         &Tensor::scalar_tensor(0, (tch::Kind::Bool, tch::Device::Cpu)),
    //         false,
    //     );
    //     let _ = logits.index_put_(
    //         &[None, Some(&mask)],
    //         &Tensor::scalar_tensor(f64::NEG_INFINITY, (tch::Kind::Float, tch::Device::Cpu)),
    //         false,
    //     );
    //     let language_tokens = logits.argmax(-1, false);
    //     let language_token_probs = logits.softmax(-1, None).to(tch::Device::Cpu);
    //     let language_probs = (0..n_audio)
    //         .map(|i| {
    //             tokenizer
    //                 .all_language_tokens()
    //                 .iter()
    //                 .zip(tokenizer.all_language_codes())
    //                 .map(|(&j, c)| {
    //                     (
    //                         c,
    //                         language_token_probs
    //                             .select(0, i)
    //                             .select(0, i64::try_from(j).unwrap())
    //                             .double_value(&[]),
    //                     )
    //                 })
    //                 .collect()
    //         })
    //         .collect();
    //     Ok((PyTensor(language_tokens), language_probs))
    // }

    fn _main_loop(
        &mut self,
        audio_features: PyTensor,
        mut tokens: PyTensor,
    ) -> anyhow::Result<(PyTensor, PyTensor, Vec<f32>)> {
        let n_batch = tokens.size()[0];
        let sum_logprobs = Tensor::zeros(&[n_batch], (tch::Kind::Float, audio_features.device()));
        let mut no_speech_probs = vec![f32::NAN; n_batch.try_into().unwrap()];

        for i in 0..self.sample_len {
            let logits = self.inference.logits(
                PyTensor(tokens.shallow_clone()),
                PyTensor(audio_features.shallow_clone()),
            );
            if i == 0
            /* TODO && !tokenizer.no_speech().is_none() */
            {
                let probs_at_sot = logits
                    .select(1, self.sot_index as i64)
                    .softmax(-1, tch::Kind::Float);
                probs_at_sot
                    .select(1, self.tokenizer.no_speech() as i64)
                    .copy_data(&mut no_speech_probs, n_batch.try_into().unwrap());
            }
            let logits = logits.select(1, -1);
            for logit_filter in self.logit_filters.iter() {
                let _ = logit_filter._apply(
                    PyTensor(logits.shallow_clone()),
                    PyTensor(tokens.shallow_clone()),
                );
            }

            let (tokens_next, completed) = self.decoder.update(
                PyTensor(tokens.shallow_clone()),
                PyTensor(logits.shallow_clone()),
                PyTensor(sum_logprobs.shallow_clone()),
            );
            tokens = tokens_next;
            if completed.int64_value(&[]) > 0
                || tokens.size().last().unwrap() > &(self.n_ctx as i64)
            {
                break;
            }
        }

        self.inference.cleanup_caching();
        Ok((tokens, PyTensor(sum_logprobs), no_speech_probs))
    }

    fn run(&mut self, mel: PyTensor) -> anyhow::Result<Vec<DecodingResult>> {
        let _no_grad_guard = tch::no_grad_guard();
        let n_audio = mel.size()[0];
        let audio_features = self._get_audio_features(mel)?;
        let tokens = Tensor::from_slice(
            self.initial_tokens
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .repeat(&[n_audio, 1]);
        let (languages, _language_probs) = self._detect_language(
            PyTensor(audio_features.shallow_clone()),
            PyTensor(tokens.shallow_clone()),
        )?;
        assert_ne!(self.options.task, "lang_id");
        let n_group = self.n_group as i64;
        let tokens = tokens
            .repeat_interleave_self_int(n_group, Some(0), None)
            .to(audio_features.device());

        let (tokens, sum_logprobs, no_speech_probs) =
            self._main_loop(PyTensor(audio_features.shallow_clone()), PyTensor(tokens))?;
        let audio_features = audio_features.slice(0, None, None, n_group);
        let no_speech_probs: Vec<f32> = no_speech_probs
            .into_iter()
            .step_by(n_group.try_into().unwrap())
            .collect();
        assert_eq!(
            usize::try_from(audio_features.size()[0])?,
            no_speech_probs.len()
        );
        assert_eq!(no_speech_probs.len(), usize::try_from(n_audio)?);
        let tokens = tokens.reshape(&[n_audio, n_group, -1]);
        let sum_logprobs = sum_logprobs.reshape(&[n_audio, n_group]);

        let (tokens, sum_logprobs) = self
            .decoder
            .finalize(PyTensor(tokens), PyTensor(sum_logprobs));

        let tokens: Vec<Vec<PyTensor>> = (0..tokens.size()[0])
            .map(|i| {
                let s = tokens.select(0, i);
                (0..s.size()[0])
                    .map(|j| {
                        let t = s.select(0, j);
                        PyTensor(
                            t.slice(
                                0,
                                self.sample_begin as i64,
                                t.eq(i64::try_from(self.tokenizer.eot()).unwrap())
                                    .nonzero()
                                    .int64_value(&[0, 0]),
                                1,
                            ),
                        )
                    })
                    .collect()
            })
            .collect();

        let tokens_clone = tokens
            .iter()
            .map(|x| {
                x.iter()
                    .map(|t| PyTensor(t.shallow_clone()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let selected = self
            .sequence_ranker
            .rank(tokens_clone, sum_logprobs.clone())?; // TODO
        let tokens = selected
            .iter()
            .zip(tokens)
            .map(|(&i, t)| {
                let mut ret: Vec<i64> = vec![0; t[i as usize].size()[0] as usize];
                let len = ret.len();
                t[i as usize].copy_data(&mut ret, len);
                ret.into_iter().map(|t| t as usize).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let texts: Vec<_> = tokens
            .iter()
            .map(|t| self.tokenizer.decode(t.clone()).unwrap().trim().to_string()) // TODO
            .collect();
        let sum_logprobs = selected
            .iter()
            .zip(sum_logprobs)
            .map(|(&i, lp)| lp[i as usize])
            .collect::<Vec<_>>();
        let avg_logprobs = tokens
            .iter()
            .zip(sum_logprobs)
            .map(|(t, lp)| lp / (t.len() as f32 + 1.))
            .collect::<Vec<_>>();

        Ok(
            izip!(texts, languages, tokens, avg_logprobs, no_speech_probs)
                .map(|(text, language, tokens, avg_logprob, no_speech_prob)| {
                    DecodingResult {
                        audio_features_: None, // FIXME
                        language,
                        tokens,
                        text: text.to_string(),
                        avg_logprob,
                        no_speech_prob,
                        temperature: self.options.temperature,
                        compression_ratio: 0., // FIXME
                    }
                })
                .collect::<Vec<_>>(),
        )
    }
}

#[pyfunction]
fn _get_audio_features(self_: &Bound<'_, PyAny>, mel: PyTensor) -> anyhow::Result<PyTensor> {
    assert!(self_
        .getattr("options")?
        .getattr("fp16")?
        .extract::<bool>()?);
    let mel = mel.to_dtype(tch::Kind::Half, false, false);
    let model = self_.getattr("model")?;
    let dims = model.getattr("dims")?;
    let n_audio_ctx = dims.getattr("n_audio_ctx")?.extract::<i64>()?;
    let n_audio_state = dims.getattr("n_audio_state")?.extract::<i64>()?;
    let shapes = mel.size();
    assert_ne!(&shapes[shapes.len() - 2..], &[n_audio_ctx, n_audio_state]);
    let audio_features = model
        .getattr("encoder")?
        .call((PyTensor(mel),), None)?
        .extract::<PyTensor>()?;
    Ok(audio_features)
}

#[pyfunction]
fn _detect_language(
    self_: &Bound<'_, PyAny>,
    audio_features: PyTensor,
    tokens: PyTensor,
) -> anyhow::Result<(Vec<String>, Vec<HashMap<String, f64>>)> {
    assert!(self_.getattr("options")?.getattr("language")?.is_none());
    let model = self_.getattr("model")?;
    let (lang_tokens, lang_probs) = _model_detect_language(
        &model,
        audio_features,
        &self_.getattr("tokenizer")?.extract::<Tokenizer>()?,
    )?;
    let languages = lang_probs
        .iter()
        .map(|probs| {
            probs
                .iter()
                .max_by(|(_, x), (_, y)| x.total_cmp(y))
                .unwrap()
                .0
                .clone()
        })
        .collect();
    let index = self_.getattr("sot_index")?.extract::<i64>()? + 1;
    let _ = tokens
        .slice(0, None, None, 1)
        .select(1, index)
        .copy_(&lang_tokens);
    Ok((languages, lang_probs))
}

fn _model_detect_language(
    model: &Bound<'_, PyAny>,
    mel: PyTensor,
    tokenizer: &Tokenizer,
) -> anyhow::Result<(PyTensor, Vec<HashMap<String, f64>>)> {
    let _no_grad_guard = tch::no_grad_guard();
    assert_eq!(mel.size().len(), 3);
    let dims = model.getattr("dims")?;
    let n_audio_ctx = dims.getattr("n_audio_ctx")?.extract::<i64>()?;
    let n_audio_state = dims.getattr("n_audio_state")?.extract::<i64>()?;
    let shapes = mel.size();
    assert_eq!(&shapes[shapes.len() - 2..], &[n_audio_ctx, n_audio_state]);
    let n_audio = mel.size()[0];
    let x = Tensor::from_slice(&[i64::try_from(tokenizer.sot()).unwrap()])
        .repeat(&[n_audio, 1])
        .to(mel.device());
    let mut logits = model
        .getattr("logits")?
        .call((PyTensor(x), mel), None)?
        .extract::<PyTensor>()?
        .select(1, 0);
    let mut mask = Tensor::ones(
        &[*logits.size().last().unwrap()],
        (tch::Kind::Bool, tch::Device::Cpu),
    );
    let _ = mask.index_put_(
        &[Some(Tensor::from_slice(
            tokenizer
                .all_language_tokens()
                .iter()
                .map(|&x| i64::try_from(x).unwrap())
                .collect::<Vec<_>>()
                .as_slice(),
        ))],
        &Tensor::scalar_tensor(0, (tch::Kind::Bool, tch::Device::Cpu)),
        false,
    );
    let _ = logits.index_put_(
        &[None, Some(&mask)],
        &Tensor::scalar_tensor(f64::NEG_INFINITY, (tch::Kind::Float, tch::Device::Cpu)),
        false,
    );
    let language_tokens = logits.argmax(-1, false);
    let language_token_probs = logits.softmax(-1, None).to(tch::Device::Cpu);
    let language_probs = (0..n_audio)
        .map(|i| {
            tokenizer
                .all_language_tokens()
                .iter()
                .zip(tokenizer.all_language_codes())
                .map(|(&j, c)| {
                    (
                        c,
                        language_token_probs
                            .select(0, i)
                            .select(0, i64::try_from(j).unwrap())
                            .double_value(&[]),
                    )
                })
                .collect()
        })
        .collect();
    Ok((PyTensor(language_tokens), language_probs))
}

#[pyfunction]
fn _main_loop(
    self_: &Bound<'_, PyAny>,
    audio_features: PyTensor,
    mut tokens: PyTensor,
) -> anyhow::Result<(PyTensor, PyTensor, Vec<f32>)> {
    let n_batch = tokens.size()[0];
    let sum_logprobs = Tensor::zeros(&[n_batch], (tch::Kind::Float, audio_features.device()));
    let mut no_speech_probs = vec![f32::NAN; n_batch.try_into().unwrap()];

    let tokenizer = self_.getattr("tokenizer")?;
    for i in 0..self_.getattr("sample_len")?.extract::<i64>()? {
        let logits = self_
            .getattr("inference")?
            .call_method1(
                "logits",
                (
                    PyTensor(tokens.shallow_clone()),
                    PyTensor(audio_features.shallow_clone()),
                ),
            )?
            .extract::<PyTensor>()?;
        if i == 0 && !tokenizer.getattr("no_speech")?.is_none() {
            let probs_at_sot = logits
                .select(1, self_.getattr("sot_index")?.extract::<i64>()?)
                .softmax(-1, tch::Kind::Float);
            probs_at_sot
                .select(1, tokenizer.getattr("no_speech")?.extract::<i64>()?)
                .copy_data(&mut no_speech_probs, n_batch.try_into().unwrap());
        }
        let logits = logits.select(1, -1);
        for logit_filter in self_.getattr("logit_filters")?.iter()? {
            logit_filter?.call_method1(
                "apply",
                (
                    PyTensor(logits.shallow_clone()),
                    PyTensor(tokens.shallow_clone()),
                ),
            )?;
        }

        let (tokens_next, completed) = self_
            .getattr("decoder")?
            .call_method1(
                "update",
                (
                    PyTensor(tokens.shallow_clone()),
                    PyTensor(logits.shallow_clone()),
                    PyTensor(sum_logprobs.shallow_clone()),
                ),
            )?
            .extract::<(PyTensor, PyTensor)>()?;
        tokens = tokens_next;
        if completed.int64_value(&[]) > 0
            || tokens.size().last().unwrap() > &self_.getattr("n_ctx")?.extract::<i64>()?
        {
            break;
        }
    }

    self_
        .getattr("inference")?
        .call_method0("cleanup_caching")?;
    Ok((tokens, PyTensor(sum_logprobs), no_speech_probs))
}

#[pyfunction]
fn run(self_: &Bound<'_, PyAny>, mel: PyTensor) -> anyhow::Result<Vec<DecodingResult>> {
    let _no_grad_guard = tch::no_grad_guard();
    let tokenizer = self_.getattr("tokenizer")?.extract::<Tokenizer>()?;
    let n_audio = mel.size()[0];
    let audio_features = _get_audio_features(self_, mel)?;
    let tokens = Tensor::from_slice(
        self_
            .getattr("initial_tokens")?
            .extract::<Vec<i64>>()?
            .as_slice(),
    )
    .repeat(&[n_audio, 1]);
    let (languages, _language_probs) = _detect_language(
        self_,
        PyTensor(audio_features.shallow_clone()),
        PyTensor(tokens.shallow_clone()),
    )?;
    assert_ne!(
        self_.getattr("options")?.getattr("task")?.to_string(),
        "lang_id"
    );
    let n_group = self_.getattr("n_group")?.extract::<i64>()?;
    let tokens = tokens
        .repeat_interleave_self_int(n_group, Some(0), None)
        .to(audio_features.device());

    let (tokens, sum_logprobs, no_speech_probs) = _main_loop(
        self_,
        PyTensor(audio_features.shallow_clone()),
        PyTensor(tokens),
    )?;
    let audio_features = audio_features.slice(0, None, None, n_group);
    let no_speech_probs: Vec<f32> = no_speech_probs
        .into_iter()
        .step_by(n_group.try_into().unwrap())
        .collect();
    assert_eq!(
        usize::try_from(audio_features.size()[0])?,
        no_speech_probs.len()
    );
    assert_eq!(no_speech_probs.len(), usize::try_from(n_audio)?);
    let tokens = tokens.reshape(&[n_audio, n_group, -1]);
    let sum_logprobs = sum_logprobs.reshape(&[n_audio, n_group]);

    let (tokens, sum_logprobs) = self_
        .getattr("decoder")?
        .call_method1("finalize", (PyTensor(tokens), PyTensor(sum_logprobs)))?
        .extract::<(PyTensor, Vec<Vec<f32>>)>()?;

    let tokens: Vec<Vec<PyTensor>> = (0..tokens.size()[0])
        .map(|i| {
            let s = tokens.select(0, i);
            (0..s.size()[0])
                .map(|j| {
                    let t = s.select(0, j);
                    PyTensor(
                        t.slice(
                            0,
                            self_
                                .getattr("sample_begin")
                                .unwrap()
                                .extract::<i64>()
                                .unwrap(),
                            t.eq(i64::try_from(tokenizer.eot()).unwrap())
                                .nonzero()
                                .int64_value(&[0, 0]),
                            1,
                        ),
                    )
                })
                .collect()
        })
        .collect();

    let tokens_clone = tokens
        .iter()
        .map(|x| {
            x.iter()
                .map(|t| PyTensor(t.shallow_clone()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let selected = self_
        .getattr("sequence_ranker")?
        .call_method1("rank", (tokens_clone, sum_logprobs.clone()))? // TODO
        .extract::<Vec<i64>>()?;
    let tokens = selected
        .iter()
        .zip(tokens)
        .map(|(&i, t)| {
            let mut ret: Vec<i64> = vec![0; t[i as usize].size()[0] as usize];
            let len = ret.len();
            t[i as usize].copy_data(&mut ret, len);
            ret.into_iter().map(|t| t as usize).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let texts: Vec<_> = tokens
        .iter()
        .map(|t| tokenizer.decode(t.clone()).unwrap().trim().to_string()) // TODO
        .collect();
    let sum_logprobs = selected
        .iter()
        .zip(sum_logprobs)
        .map(|(&i, lp)| lp[i as usize])
        .collect::<Vec<_>>();
    let avg_logprobs = tokens
        .iter()
        .zip(sum_logprobs)
        .map(|(t, lp)| lp / (t.len() as f32 + 1.))
        .collect::<Vec<_>>();

    Ok(
        izip!(texts, languages, tokens, avg_logprobs, no_speech_probs)
            .map(|(text, language, tokens, avg_logprob, no_speech_prob)| {
                DecodingResult {
                    audio_features_: None, // FIXME
                    language,
                    tokens,
                    text: text.to_string(),
                    avg_logprob,
                    no_speech_prob,
                    temperature: self_
                        .getattr("options")
                        .unwrap()
                        .getattr("temperature")
                        .unwrap()
                        .extract::<f64>()
                        .unwrap(),
                    compression_ratio: 0., // FIXME
                }
            })
            .collect::<Vec<_>>(),
    )
}

// =========================================================
// PyTorchInference
// =========================================================

#[pyclass]
struct PyTorchInference {
    initial_token_length: i64,
    kv_cache: Vec<PyTensor>,
    prefill: AOTIModel,
    decoder: AOTIModel,
}

#[pymethods]
impl PyTorchInference {
    #[new]
    fn new(initial_token_length: i64) -> Self {
        PyTorchInference {
            initial_token_length,
            kv_cache: vec![],
            prefill: aot_load("./generated/prefill.so".to_string(), "cpu".to_string()).unwrap(),
            decoder: aot_load("./generated/decoder.so".to_string(), "cpu".to_string()).unwrap(),
        }
    }

    fn logits(&mut self, mut tokens: PyTensor, audio_features: PyTensor) -> PyTensor {
        if *tokens.size().last().unwrap() > self.initial_token_length {
            tokens = PyTensor(tokens.slice(1, Some(-1), None, 1));
        }
        let mut args = vec![tokens, audio_features];
        args.append(&mut self.kv_cache);
        let res = if args.len() == 2 {
            self.prefill.call(args)
        } else {
            self.decoder.call(args)
        }
        .unwrap();
        let (ret, new_kv_cache) = res.split_first().unwrap();
        self.kv_cache.clear();
        for t in new_kv_cache {
            self.kv_cache.push(PyTensor(t.shallow_clone()))
        }
        PyTensor(ret.shallow_clone())
    }

    fn cleanup_caching(&mut self) {
        self.kv_cache.clear();
    }
}

// =========================================================
// DecodingResult
// =========================================================

#[pyclass]
struct DecodingResult {
    audio_features_: Option<PyTensor>,
    #[pyo3(get)]
    language: String,
    #[pyo3(get)]
    tokens: Vec<usize>,
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    avg_logprob: f32,
    #[pyo3(get)]
    no_speech_prob: f32,
    #[pyo3(get)]
    temperature: f64,
    #[pyo3(get)]
    compression_ratio: f64,
}

#[pymethods]
impl DecodingResult {
    #[getter]
    fn audio_features(&self) -> Option<PyTensor> {
        None // FIXME
    }
}

// =========================================================
// aot_load
// =========================================================

#[repr(C)]
pub struct ContainerOpaque {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub struct StreamOpaque {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub struct ProxyExecutorOpaque {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub struct AtenTensorOpaque {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

type AOTIRuntimeError = i32;
type ContainerHandle = *mut ContainerOpaque;

type AtenTensorHandle = *mut AtenTensorOpaque;
type StreamHandle = *mut StreamOpaque;
type ProxyExecutorHandle = *mut ProxyExecutorOpaque;

type CreateFn = unsafe extern "C" fn(
    *mut ContainerHandle,
    libc::size_t,
    *const libc::c_char,
    *const libc::c_char,
) -> AOTIRuntimeError;
type DeleteFn = unsafe extern "C" fn(ContainerHandle) -> AOTIRuntimeError;
type RunFn = unsafe extern "C" fn(
    ContainerHandle,
    *mut AtenTensorHandle,
    libc::size_t,
    *mut AtenTensorHandle,
    libc::size_t,
    StreamHandle,
    ProxyExecutorHandle,
) -> AOTIRuntimeError;
type GetNumInputsFn = unsafe extern "C" fn(ContainerHandle, *mut libc::size_t) -> AOTIRuntimeError;
type GetNumOutputsFn = unsafe extern "C" fn(ContainerHandle, *mut libc::size_t) -> AOTIRuntimeError;
type GetCallSpecFn = unsafe extern "C" fn(
    ContainerHandle,
    *mut *const libc::c_char,
    *mut *const libc::c_char,
) -> AOTIRuntimeError;

#[pyclass]
struct AOTIModel {
    path: String,
    device: String,
    lib: libloading::Library,
    container: ContainerHandle,
}

unsafe impl Send for AOTIModel {}

impl AOTIModel {
    unsafe fn new(path: String, device: String) -> anyhow::Result<Self> {
        assert!(std::path::Path::new(path.as_str()).exists());
        let lib = libloading::Library::new(&path)?;
        let create: libloading::Symbol<CreateFn> =
            lib.get(b"AOTInductorModelContainerCreateWithDevice")?;
        let mut container: ContainerHandle = std::ptr::null_mut();
        // TODO cubin dir.
        create(
            &mut container,
            1,
            std::ffi::CString::new(&*device)?.as_ptr(),
            std::ptr::null(),
        ); // TODO error check
        Ok(AOTIModel {
            path,
            device,
            lib,
            container,
        })
    }
}

unsafe fn to_tensor_handle(x: &mut PyTensor) -> AtenTensorHandle {
    torch_sys::at_shallow_clone(x.0.as_mut_ptr()) as AtenTensorHandle
}

unsafe fn to_py_tensor(x: AtenTensorHandle) -> PyTensor {
    // TODO leaky
    PyTensor(Tensor::clone_from_ptr(x as *mut torch_sys::C_tensor)) // TODO buggy if duplicated pointers.
}

#[pymethods]
impl AOTIModel {
    fn call_spec(&self) -> anyhow::Result<(String, String)> {
        unsafe {
            let get_call_spec: libloading::Symbol<GetCallSpecFn> =
                self.lib.get(b"AOTInductorModelContainerGetCallSpec")?;
            let mut in_spec = std::ptr::null();
            let mut out_spec = std::ptr::null();
            // TODO error handling
            get_call_spec(self.container, &mut in_spec, &mut out_spec);
            // TODO leaky
            Ok((
                unsafe { std::ffi::CStr::from_ptr(in_spec) }
                    .to_str()?
                    .to_owned(),
                unsafe { std::ffi::CStr::from_ptr(out_spec) }
                    .to_str()?
                    .to_owned(),
            ))
        }
    }

    fn call(&self, mut inputs: Vec<PyTensor>) -> anyhow::Result<Vec<PyTensor>> {
        unsafe {
            let run: libloading::Symbol<RunFn> = self.lib.get(b"AOTInductorModelContainerRun")?;
            let get_num_inputs: libloading::Symbol<GetNumInputsFn> =
                self.lib.get(b"AOTInductorModelContainerGetNumInputs")?;
            let mut num_inputs: libc::size_t = 0;
            get_num_inputs(self.container, &mut num_inputs);
            assert_eq!(num_inputs, inputs.len());
            let get_num_outputs: libloading::Symbol<GetNumOutputsFn> =
                self.lib.get(b"AOTInductorModelContainerGetNumOutputs")?;
            let mut num_outputs: libc::size_t = 0;
            get_num_outputs(self.container, &mut num_outputs); // TODO error check
            let mut input_handles: Vec<AtenTensorHandle> = inputs
                .iter_mut()
                .map(|x| unsafe { to_tensor_handle(x) })
                .collect();
            let mut output_handles = vec![std::ptr::null_mut(); num_outputs];
            let _no_grad_guard = tch::no_grad_guard();
            let json_path = self.path.replace(".so", ".json"); // TODO
            println!("11111111111111111");
            let proxy_executor = if std::path::Path::new(json_path.as_str()).exists() {
                assert_eq!(self.device, "cpu".to_string());
                let tmp = std::ffi::CString::new(&*json_path)?;
                torch_sys::torch_aoti_make_proxy_executor(tmp.as_ptr(), true) as ProxyExecutorHandle // TODO is_cpu
            } else {
                std::ptr::null_mut()
            };
            println!("222222222222222222222222");
            // TODO proxy executor, cuda stream.
            run(
                self.container,
                input_handles.as_mut_ptr(),
                inputs.len(),
                output_handles.as_mut_ptr(),
                num_outputs,
                std::ptr::null_mut(),
                proxy_executor,
            ); // TODO error check
            println!("333333333333333333333333333333");
            if proxy_executor != std::ptr::null_mut() {
                torch_sys::torch_aoti_delete_proxy_executor(proxy_executor as *mut libc::c_void); // TODO caching
            }
            println!("444444444444444444444444");
            Ok(output_handles
                .into_iter()
                .map(|x| unsafe { to_py_tensor(x) })
                .collect())
        }
    }

    #[pyo3(signature = (*py_args))]
    fn __call__(&self, mut py_args: &Bound<'_, PyTuple>) -> anyhow::Result<Vec<PyTensor>> {
        // TODO tree flatten
        let mut inputs = vec![];
        for a in py_args {
            inputs.push(a.extract::<PyTensor>()?);
        }
        self.call(inputs)
    }
}

impl Drop for AOTIModel {
    fn drop(&mut self) {
        unsafe {
            let delete: libloading::Symbol<DeleteFn> =
                self.lib.get(b"AOTInductorModelContainerDelete").unwrap();
            delete(self.container);
        }
    }
}

#[pyfunction]
fn aot_load(path: String, device: String) -> anyhow::Result<AOTIModel> {
    unsafe { AOTIModel::new(path, device) }
}

#[pyfunction]
fn make_decoding_task(py_dims: &Bound<'_, PyAny>) -> anyhow::Result<DecodingTask> {
    let dims = ModelDimensions::base();
    assert_eq!(
        py_dims.getattr("n_audio_ctx")?.extract::<i32>()?,
        dims.n_audio_ctx
    );
    assert_eq!(
        py_dims.getattr("n_text_ctx")?.extract::<i32>()?,
        dims.n_text_ctx
    );
    DecodingTask::new(dims, DecodingOptions::default())
}

#[pyfunction]
fn log_mel_spectrogram(audio: PyTensor) -> PyTensor {
    let mel = aot_load("./generated/mel.so".to_string(), "cpu".to_string()).unwrap();
    PyTensor(mel.call(vec![audio]).unwrap()[0].shallow_clone()) // FIXME
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn whisper_aoti(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Encoding>()?;
    m.add_class::<Tokenizer>()?;
    m.add_class::<SuppressBlank>()?;
    m.add_class::<SuppressTokens>()?;
    m.add_class::<ApplyTimestampRules>()?;
    m.add_class::<GreedyDecoder>()?;
    m.add_class::<DecodingResult>()?;
    m.add_class::<MaximumLikelihoodRanker>()?;
    m.add_class::<AOTIModel>()?;
    m.add_class::<DecodingTask>()?;
    m.add_function(wrap_pyfunction!(get_tokenizer, m)?).unwrap();
    m.add_function(wrap_pyfunction!(_get_audio_features, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(_detect_language, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(_main_loop, m)?).unwrap();
    m.add_function(wrap_pyfunction!(run, m)?).unwrap();
    m.add_function(wrap_pyfunction!(aot_load, m)?).unwrap();
    m.add_function(wrap_pyfunction!(make_decoding_task, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(log_mel_spectrogram, m)?).unwrap();
    Ok(())
}