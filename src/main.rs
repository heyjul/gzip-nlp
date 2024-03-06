use flate2::write::GzEncoder;
use flate2::Compression;
use rayon::prelude::*;
use serde::Deserialize;
use std::{collections::HashMap, env, fs, io::Write};
use rand::prelude::*;

const K: usize = 3;
const MIN_OCCURENCE: usize = 1000;
const TEST_RATIO: usize = 10; // 10%
const TEST_COUNT: usize = MIN_OCCURENCE / 100 * TEST_RATIO;

// const SEPARATOR: fn(char) -> bool = |x| x.is_ascii_whitespace();
const SEPARATOR: &str = ",";

fn main() {

    clean_file();
    
    let files: Vec<_> = env::args().skip(1).take(2).collect();

    let training_set = fs::read_to_string(&files[0]).unwrap();
    let training_set: Vec<_> = training_set
        .lines()
        .filter_map(|x| x.split_once(SEPARATOR))
        .collect();

    let test_set = fs::read_to_string(&files[1]).unwrap();
    let test_set: Vec<_> = test_set
        .lines()
        .filter_map(|x| x.split_once(SEPARATOR))
        .collect();

    let mut ok = 0;
    let mut executed = 0;
    let total = test_set.len();

    for (expected, x1) in test_set.iter().take(10) {
        let mut distance_from_x1 = Vec::with_capacity(training_set.len());
        let cx1 = zip_len(x1.as_bytes());

        training_set
            .par_iter()
            .enumerate()
            .map(move |(i, (_, x2))| {
                let cx2 = zip_len(x2.as_bytes());
                let cx1x2 = zip_len(format!("{x1}{x2}").as_bytes());

                let ncd = (cx1x2 - cx1.min(cx2)) as f64 / cx1.max(cx2) as f64;
                (i, ncd)
            })
            .collect_into_vec(&mut distance_from_x1);

        distance_from_x1.sort_unstable_by(|(_, a), (_, b)| a.total_cmp(b));
        let top_k_class: Vec<_> = distance_from_x1
            .into_iter()
            .take(K)
            .map(|(i, _)| training_set[i].0)
            .collect();

        let mut counts = std::collections::HashMap::new();
        for class in &top_k_class {
            *counts.entry(class).or_insert(0) += 1;
        }

        // let predict_class = counts
        //     .iter()
        //     .max_by_key(|&(_, count)| count)
        //     .map(|(class, _)| *class)
        //     .unwrap();

        // executed += 1;
        // if &expected == predict_class {
        //     ok += 1;
        // }

        executed += 1;
        if top_k_class.contains(&expected) {
            ok += 1;
        }

        println!("Total : {total} : {ok}/{executed} => {:.2}%", ok as f32 / executed as f32 * 100_f32);
        println!("Expected : {expected} Predicted (top {K}) : {top_k_class:?}")
    }
}

fn clean_file() {
    let file = fs::File::open("./data/oe_noe_tokenized (3).csv").unwrap();
    let mut rdr = csv::ReaderBuilder::new().delimiter(b',').from_reader(file);

    let records: Vec<_> = rdr.deserialize().filter_map(|x| x.ok()).map(Record::clean).collect();

    let grouped_records = records.iter().fold(HashMap::new(), |mut acc, record| {
        let entry = acc.entry(&record.rome_v3_mon_profil).or_insert(Vec::new());
        entry.push(record);
        acc
    });

    let mut rng = rand::thread_rng();
    let group_with_min_occurence: Vec<Vec<_>> = grouped_records.iter().filter(|x| x.1.len() >= MIN_OCCURENCE).map(|x| x.1.iter().take(MIN_OCCURENCE).collect()).collect();
    
    let mut tests: Vec<_> = group_with_min_occurence.iter().map(|x| x.iter().take(TEST_COUNT).collect::<Vec<_>>()).flat_map(|x| x.iter().map(|x| format!("{}{SEPARATOR}{}", x.rome_v3_mon_profil, x.text.as_deref().unwrap_or(""))).collect::<Vec<_>>()).collect();
    tests.shuffle(&mut rng);

    let mut test_file = fs::File::create("./data/oe_noe_tokenized_test.csv").unwrap();
    test_file.write_all(tests.join("\n").as_bytes()).unwrap();
    
    let mut training: Vec<_> = group_with_min_occurence.iter().map(|x| x.iter().skip(TEST_COUNT).collect::<Vec<_>>()).flat_map(|x| x.iter().map(|x| format!("{}{SEPARATOR}{}", x.rome_v3_mon_profil, x.text.as_deref().unwrap_or(""))).collect::<Vec<_>>()).collect();
    training.shuffle(&mut rng);

    let mut training_file = fs::File::create("./data/oe_noe_tokenized_training.csv").unwrap();
    training_file.write_all(training.join("\n").as_bytes()).unwrap();
}

fn zip_len(content: &[u8]) -> usize {
    let mut zip = GzEncoder::new(Vec::new(), Compression::default());
    zip.write_all(content).unwrap();

    zip.finish().unwrap().len()
}

#[derive(Debug, Deserialize)]
struct Record {
    rome_v3_mon_profil: String,
    text: Option<String>,
    description_libre_offre: Option<String>,
    intitule_libre_metier: Option<String>,
}

impl Record {
    fn clean(self) -> Self {
        Self {
            text: self.text.map(|x| x.replace("nan", "").replace("(H/F)", "")),
            ..self
        }
    }

    fn to_string(&self) -> String {
        [&self.text, &self.description_libre_offre, &self.intitule_libre_metier].map(|x| x.as_deref().unwrap_or("")).join(SEPARATOR)
    }
}
