use flate2::write::GzEncoder;
use flate2::Compression;
use rayon::prelude::*;
use std::{env, fs, io::Write};

const K: usize = 2;

// const SEPARATOR: fn(char) -> bool = |x| x.is_ascii_whitespace();
const SEPARATOR: char = ',';

fn main() {
    let files: Vec<_> = env::args().skip(1).take(2).collect();

    let training_set = fs::read_to_string(&files[0]).unwrap();
    let training_set: Vec<_> = training_set
        .lines()
        .filter_map(|x| x.split_once(SEPARATOR))
        .collect();

    let test_set = fs::read_to_string(&files[1]).unwrap();
    let test_set = test_set
        .lines()
        .take(1)
        .filter_map(|x| x.split_once(SEPARATOR));

    let mut ok = 0;
    let mut total = 0;
    for (expected, x1) in test_set {
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

        let predict_class = counts
            .iter()
            .max_by_key(|&(_, count)| count)
            .map(|(class, _)| *class)
            .unwrap();

        total += 1;
        if &expected == predict_class {
            ok += 1;
        }

        println!("{ok}/{total} : {:2}%", ok as f32 / total as f32 * 100_f32);
    }
}

fn zip_len(content: &[u8]) -> usize {
    let mut zip = GzEncoder::new(Vec::new(), Compression::default());
    zip.write_all(content).unwrap();

    zip.finish().unwrap().len()
}
