//! Uzbek output cleanup, the rules Kotib applies before every paste
//! (`references/kotib/src/text_format.swift:23-125`, MIT; our own code, with
//! Kotib's test vectors from `tests/test_text_format.swift:20-56`):
//! 1. collapse Whisper's phrase loops: a 1–6 word phrase repeated 4+ times in
//!    a row stays once ("oʻzbekiston respublikasi" ×15 → once);
//! 2. unify apostrophes: ʻ (U+02BB) after o/g, ʼ (U+02BC) elsewhere.

const PHRASE_MAX_WORDS: usize = 6;
const MIN_REPEATS: usize = 4;
/// The apostrophe forms Whisper mixes.
const APOSTROPHES: [char; 7] = [
    '\'', '\u{2018}', '\u{2019}', '`', '\u{00B4}', '\u{02BB}', '\u{02BC}',
];

/// Both rules, loops first (as Kotib's `matnniTayyorla`).
pub fn clean(text: &str) -> String {
    unify_apostrophes(&collapse_phrase_loops(text))
}

/// Words are split on ASCII spaces and compared exactly. Text without a loop
/// comes back unchanged (double spaces too); with one, it is rejoined with
/// single spaces.
fn collapse_phrase_loops(text: &str) -> String {
    let words: Vec<&str> = text.split(' ').filter(|w| !w.is_empty()).collect();
    let mut kept: Vec<&str> = Vec::with_capacity(words.len());
    let mut collapsed = false;
    let mut i = 0;
    while i < words.len() {
        let looped = (1..=PHRASE_MAX_WORDS)
            .take_while(|k| i + k * MIN_REPEATS <= words.len())
            .find_map(|k| {
                let mut times = 1;
                while i + (times + 1) * k <= words.len()
                    && words[i + times * k..i + (times + 1) * k] == words[i..i + k]
                {
                    times += 1;
                }
                (times >= MIN_REPEATS).then_some((k, times))
            });
        match looped {
            Some((k, times)) => {
                kept.extend_from_slice(&words[i..i + k]);
                i += times * k;
                collapsed = true;
            }
            None => {
                kept.push(words[i]);
                i += 1;
            }
        }
    }
    if collapsed {
        kept.join(" ")
    } else {
        text.to_string()
    }
}

fn unify_apostrophes(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut previous = None;
    for c in text.chars() {
        if APOSTROPHES.contains(&c) {
            let after_o_or_g = matches!(previous, Some('o' | 'O' | 'g' | 'G'));
            out.push(if after_o_or_g { '\u{02BB}' } else { '\u{02BC}' });
        } else {
            out.push(c);
        }
        previous = Some(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_loops_like_kotib() {
        let cases = [
            (
                "bu oʻzbekiston respublikasi oʻzbekiston respublikasi oʻzbekiston respublikasi oʻzbekiston respublikasi oʻzbekiston respublikasi oʻzbekiston respublikasi",
                "bu oʻzbekiston respublikasi",
            ),
            (
                "qaynon shirin toʻla toʻla toʻla toʻla toʻla toʻla toʻla toʻla",
                "qaynon shirin toʻla",
            ),
            (
                "oʻn toʻrt yuz ellik ming toʻrt yuz ellik ming toʻrt yuz ellik ming toʻrt yuz ellik ming oʻn olti",
                "oʻn toʻrt yuz ellik ming oʻn olti",
            ),
            ("ha ha ha keldik", "ha ha ha keldik"),
            ("toʻla toʻla idish", "toʻla toʻla idish"),
            ("Salom,  dunyo!  Bugun", "Salom,  dunyo!  Bugun"),
            ("ha ha ha ha.", "ha ha ha ha."),
            ("", ""),
        ];
        for (input, expected) in cases {
            assert_eq!(collapse_phrase_loops(input), expected, "{input}");
        }
    }

    #[test]
    fn unifies_apostrophes_like_kotib() {
        let cases = [
            ("o'zbek", "oʻzbek"),
            ("g`alaba", "gʻalaba"),
            ("o´zbek", "oʻzbek"),
            ("ma'no", "maʼno"),
            ("o'zbek ma'nosi", "oʻzbek maʼnosi"),
            ("oʻzbek gʻalaba maʼno", "oʻzbek gʻalaba maʼno"),
            ("O\u{2019}zbekiston", "Oʻzbekiston"),
        ];
        for (input, expected) in cases {
            assert_eq!(unify_apostrophes(input), expected, "{input}");
        }
    }

    #[test]
    fn clean_collapses_before_unifying() {
        assert_eq!(clean("men o'zbek o'zbek o'zbek o'zbek"), "men oʻzbek");
    }
}
