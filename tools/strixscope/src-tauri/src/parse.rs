//! Classifies a StrixFold log line into a typed event, or `None`; patterns search, not anchor, as writers can interleave.

use std::sync::LazyLock;

use regex::Regex;

#[derive(Debug, Clone, PartialEq)]
pub enum Line {
    /// `== <iso> strixfold rank R on HOST: run NAME, conf PATH (HASH), deploy PATH`
    RunStart {
        rank: u8,
        name: String,
        conf: String,
        deploy: String,
    },
    /// `== argv: ... --port P --parallel N --context C ...`
    Argv {
        port: Option<u32>,
        parallel: Option<u32>,
        context: Option<u64>,
        prompt_cache_gib: Option<f64>,
    },
    /// `[tensorfold] serving MODEL at URL (...; loaded in 51.0s)`
    Serving {
        model: String,
        loaded_s: Option<f64>,
    },
    /// `== <iso> rank R exited with status N`
    Exit { rank: u8, status: i32 },
    /// `info: lanes: step S rank 0 prompt pass: N tokens, slot S, ...; K other stream(s) live`
    PromptPass {
        tokens: u64,
        slot: u32,
        resume: Resume,
        others: u32,
    },
    /// `info: prompt cache: N tokens, resumed at R, kept K; S states, M MiB ... (hits..)`
    PromptCache {
        tokens: u64,
        resumed_at: u64,
        states: u64,
        mib: u64,
        cap_mib: u64,
        hits: u64,
        misses: u64,
        evicted: u64,
        refused: u64,
        failed: u64,
    },
    /// `[tensorfold] done <id> prompt= cached= thinking= effort= tokens= ... finish= rounds= accepted=a/d`
    Done {
        id: String,
        prompt: u64,
        cached: u64,
        thinking: bool,
        effort: String,
        tokens: u64,
        finish: String,
        rounds: u64,
        accepted: u64,
        drafted: u64,
    },
    /// `[tensorfold] ended <id> reason=R prompt=N tokens=T after=S s`
    Ended {
        id: String,
        reason: String,
        prompt: u64,
        tokens: u64,
        after_s: f64,
    },
    /// `[tensorfold] refused <id>: why`
    Refused { id: String, why: String },
    /// `[tensorfold] stream error: X`
    StreamError(String),
    /// `[tensorfold] IP "METHOD PATH HTTP/1.1" STATUS -`
    Access {
        client: String,
        method: String,
        path: String,
        status: u16,
    },
    /// `info: sf-disk: rank=R event=save result=written tokens= blocks= new_blocks= bytes= ms=`
    DiskSave {
        rank: u8,
        tokens: u64,
        bytes: u64,
        ms: f64,
    },
    /// `info: sf-disk: rank=R event=blocks result=written tokens= count= bytes= ms=`
    DiskBlocks {
        rank: u8,
        count: u64,
        bytes: u64,
        ms: f64,
    },
    /// `info: sf-disk: rank=R event=resume result=hit tokens= bytes= ms=`
    DiskResume {
        rank: u8,
        hit: bool,
        tokens: u64,
        bytes: u64,
        ms: f64,
    },
    /// `info: dsv41: the kept state of N tokens at P goes stale: ...`
    Stale { tokens: u64 },
    /// `info: dsv41: a prompt of N tokens does not resume the kept state ...`
    NoResume { tokens: u64 },
    /// `info: lanes: step S a prompt of N tokens cuts in; the pass in slot X waits at A of B`
    CutIn { tokens: u64, waits_at: u64, of: u64 },
    /// `info: lanes: step S rank 1 restores state N at P into slot S, ...`
    Restore,
    /// `info: image memo: pictures= hits= misses= entries=`
    ImageMemo {
        pictures: u64,
        hits: u64,
        misses: u64,
    },
    /// `info: encoder: pictures= encoded= cached= skipped= ms=`
    Encoder {
        pictures: u64,
        encoded: u64,
        cached: u64,
        ms: f64,
    },
    /// `info: structured reply: N rows drawn under its grammar, X us a row ...`
    Structured { rows: u64, us_per_row: f64 },
    /// `[tensorfold] warning: a reply reached max_tokens while still thinking ...`
    MaxTokensThinking,
    /// `warning: collectives: rank R tcp_host waited N ms for its GPU (...)`
    GpuWait { rank: u8, ms: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resume {
    /// Nothing kept matched: the whole prompt is computed.
    None,
    /// A RAM-kept state resumed in place.
    Ram,
    /// A RAM-kept state resumed after moving slots.
    RamMoved,
    /// The disk prefix cache supplied the state.
    Disk,
}

impl Resume {
    pub fn as_str(self) -> &'static str {
        match self {
            Resume::None => "none",
            Resume::Ram => "ram",
            Resume::RamMoved => "ram-moved",
            Resume::Disk => "disk",
        }
    }
}

macro_rules! re {
    ($name:ident, $pat:expr) => {
        static $name: LazyLock<Regex> = LazyLock::new(|| Regex::new($pat).unwrap());
    };
}

re!(
    RUN,
    r"^== \S+ strixfold rank (\d) on \S+: run (\S+), conf (\S+) \([0-9a-f]+\), deploy (\S+)"
);
re!(EXIT, r"^== \S+ rank (\d) exited with status (-?\d+)");
re!(PORT, r"--port (\d+)");
re!(PARALLEL, r"--parallel (\d+)");
re!(CONTEXT, r"--context (\d+)");
re!(PCGIB, r"--prompt-cache-gib (\d+(?:\.\d+)?)");
re!(
    SERVING,
    r"^\[tensorfold\] serving (\S+) at .*?(?:loaded in (\d+(?:\.\d+)?)s)?\)?$"
);
re!(
    PASS,
    r"prompt pass: (\d+) tokens, slot (\d+), extent at \d+, \d+ rows; ([^;]*); (\d+) other stream"
);
re!(
    PCACHE,
    r"^info: prompt cache: (\d+) tokens, resumed at (\d+), kept \d+; (\d+) states, (\d+) MiB and \d+ MiB spare of (\d+) MiB \(hits (\d+), misses (\d+), evicted (\d+), refused (\d+), failed (\d+)\)"
);
re!(
    DONE,
    r"\[tensorfold\] done (\S+) prompt=(\d+) cached=(\d+) thinking=(\w+) effort=(\S+) tokens=(\d+) .*?finish=(\S+) (?:.*? )?rounds=(\d+) accepted=(\d+)/(\d+)"
);
re!(
    ENDED,
    r"\[tensorfold\] ended (\S+) reason=(\S+) prompt=(\d+) tokens=(\d+) after=(\d+(?:\.\d+)?)s"
);
re!(REFUSED, r"^\[tensorfold\] refused ([^:\s]+): (.*)$");
re!(STREAMERR, r"^\[tensorfold\] stream error: (.*)$");
re!(
    ACCESS,
    r#"\[tensorfold\] (\S+) "([A-Z]+) (\S+) HTTP/[\d.]+" (\d{3})"#
);
re!(
    DSAVE,
    r"sf-disk: rank=(\d) event=save result=written tokens=(\d+) .*?bytes=(\d+) ms=(\d+(?:\.\d+)?)"
);
re!(
    DBLOCKS,
    r"sf-disk: rank=(\d) event=blocks result=written tokens=\d+ count=(\d+) bytes=(\d+) ms=(\d+(?:\.\d+)?)"
);
re!(
    DRESUME,
    r"sf-disk: rank=(\d) event=resume result=(\w+) tokens=(\d+) bytes=(\d+) ms=(\d+(?:\.\d+)?)"
);
re!(STALE, r"the kept state of (\d+) tokens at \d+ goes stale");
re!(
    NORESUME,
    r"a prompt of (\d+) tokens does not resume the kept state"
);
re!(
    CUTIN,
    r"a prompt of (\d+) tokens cuts in; the pass in slot \d+ waits at (\d+) of (\d+)"
);
re!(RESTORE, r"^info: lanes: step \d+ rank \d restores state");
re!(
    IMEMO,
    r"^info: image memo: pictures=(\d+) hits=(\d+) misses=(\d+)"
);
re!(
    ENCODER,
    r"^info: encoder: pictures=(\d+) encoded=(\d+) cached=(\d+) skipped=\d+ ms=(\d+(?:\.\d+)?)"
);
re!(
    STRUCT,
    r"^info: structured reply: (\d+) rows drawn under its grammar, (\d+(?:\.\d+)?) us a row"
);
re!(
    GPUWAIT,
    r"^warning: collectives: rank (\d) tcp_host waited (\d+(?:\.\d+)?) ms for its GPU"
);

fn n(s: &str) -> u64 {
    s.parse().unwrap_or(0)
}
fn f(s: &str) -> f64 {
    s.parse().unwrap_or(0.0)
}

pub fn parse(line: &str) -> Option<Line> {
    // Cheap prefix dispatch first: most lines are access logs and the lanes/cache/done trio.
    if line.starts_with("[tensorfold] ") {
        if let Some(c) = DONE.captures(line) {
            return Some(Line::Done {
                id: c[1].to_string(),
                prompt: n(&c[2]),
                cached: n(&c[3]),
                thinking: &c[4] == "True",
                effort: c[5].to_string(),
                tokens: n(&c[6]),
                finish: c[7].to_string(),
                rounds: n(&c[8]),
                accepted: n(&c[9]),
                drafted: n(&c[10]),
            });
        }
        if let Some(c) = ACCESS.captures(line) {
            return Some(Line::Access {
                client: c[1].to_string(),
                method: c[2].to_string(),
                path: c[3].to_string(),
                status: c[4].parse().unwrap_or(0),
            });
        }
        if let Some(c) = ENDED.captures(line) {
            return Some(Line::Ended {
                id: c[1].to_string(),
                reason: c[2].to_string(),
                prompt: n(&c[3]),
                tokens: n(&c[4]),
                after_s: f(&c[5]),
            });
        }
        if let Some(c) = REFUSED.captures(line) {
            return Some(Line::Refused {
                id: c[1].to_string(),
                why: c[2].to_string(),
            });
        }
        if let Some(c) = STREAMERR.captures(line) {
            return Some(Line::StreamError(c[1].to_string()));
        }
        if line.contains("reached max_tokens while still thinking") {
            return Some(Line::MaxTokensThinking);
        }
        if let Some(c) = SERVING.captures(line) {
            return Some(Line::Serving {
                model: c[1].to_string(),
                loaded_s: c.get(2).map(|m| f(m.as_str())),
            });
        }
        return None;
    }
    if line.starts_with("info: ") {
        if let Some(c) = PASS.captures(line) {
            let how = &c[3];
            let resume = if how.starts_with("resumes disk state") {
                Resume::Disk
            } else if how.starts_with("resumes state") && how.contains("moved from") {
                Resume::RamMoved
            } else if how.starts_with("resumes state") {
                Resume::Ram
            } else {
                Resume::None
            };
            return Some(Line::PromptPass {
                tokens: n(&c[1]),
                slot: c[2].parse().unwrap_or(0),
                resume,
                others: c[4].parse().unwrap_or(0),
            });
        }
        if let Some(c) = PCACHE.captures(line) {
            return Some(Line::PromptCache {
                tokens: n(&c[1]),
                resumed_at: n(&c[2]),
                states: n(&c[3]),
                mib: n(&c[4]),
                cap_mib: n(&c[5]),
                hits: n(&c[6]),
                misses: n(&c[7]),
                evicted: n(&c[8]),
                refused: n(&c[9]),
                failed: n(&c[10]),
            });
        }
        if line.starts_with("info: sf-disk: ") {
            if let Some(c) = DSAVE.captures(line) {
                return Some(Line::DiskSave {
                    rank: c[1].parse().unwrap_or(0),
                    tokens: n(&c[2]),
                    bytes: n(&c[3]),
                    ms: f(&c[4]),
                });
            }
            if let Some(c) = DBLOCKS.captures(line) {
                return Some(Line::DiskBlocks {
                    rank: c[1].parse().unwrap_or(0),
                    count: n(&c[2]),
                    bytes: n(&c[3]),
                    ms: f(&c[4]),
                });
            }
            if let Some(c) = DRESUME.captures(line) {
                return Some(Line::DiskResume {
                    rank: c[1].parse().unwrap_or(0),
                    hit: &c[2] == "hit",
                    tokens: n(&c[3]),
                    bytes: n(&c[4]),
                    ms: f(&c[5]),
                });
            }
            return None;
        }
        if let Some(c) = STALE.captures(line) {
            return Some(Line::Stale { tokens: n(&c[1]) });
        }
        if let Some(c) = NORESUME.captures(line) {
            return Some(Line::NoResume { tokens: n(&c[1]) });
        }
        if let Some(c) = CUTIN.captures(line) {
            return Some(Line::CutIn {
                tokens: n(&c[1]),
                waits_at: n(&c[2]),
                of: n(&c[3]),
            });
        }
        if RESTORE.is_match(line) {
            return Some(Line::Restore);
        }
        if let Some(c) = IMEMO.captures(line) {
            return Some(Line::ImageMemo {
                pictures: n(&c[1]),
                hits: n(&c[2]),
                misses: n(&c[3]),
            });
        }
        if let Some(c) = ENCODER.captures(line) {
            return Some(Line::Encoder {
                pictures: n(&c[1]),
                encoded: n(&c[2]),
                cached: n(&c[3]),
                ms: f(&c[4]),
            });
        }
        if let Some(c) = STRUCT.captures(line) {
            return Some(Line::Structured {
                rows: n(&c[1]),
                us_per_row: f(&c[2]),
            });
        }
        return None;
    }
    if line.starts_with("== ") {
        if let Some(c) = RUN.captures(line) {
            return Some(Line::RunStart {
                rank: c[1].parse().unwrap_or(0),
                name: c[2].to_string(),
                conf: c[3].trim_end_matches(',').to_string(),
                deploy: c[4].to_string(),
            });
        }
        if line.starts_with("== argv:") {
            let cap = |r: &Regex| r.captures(line).map(|c| c[1].to_string());
            return Some(Line::Argv {
                port: cap(&PORT).and_then(|s| s.parse().ok()),
                parallel: cap(&PARALLEL).and_then(|s| s.parse().ok()),
                context: cap(&CONTEXT).and_then(|s| s.parse().ok()),
                prompt_cache_gib: cap(&PCGIB).and_then(|s| s.parse().ok()),
            });
        }
        if let Some(c) = EXIT.captures(line) {
            return Some(Line::Exit {
                rank: c[1].parse().unwrap_or(0),
                status: c[2].parse().unwrap_or(0),
            });
        }
        return None;
    }
    if let Some(c) = GPUWAIT.captures(line) {
        return Some(Line::GpuWait {
            rank: c[1].parse().unwrap_or(0),
            ms: f(&c[2]),
        });
    }
    // Interleaved writers can push a lanes line behind another prefix.
    if let Some(c) = PASS.captures(line) {
        return Some(Line::PromptPass {
            tokens: n(&c[1]),
            slot: c[2].parse().unwrap_or(0),
            resume: Resume::None,
            others: c[4].parse().unwrap_or(0),
        });
    }
    None
}

/// The deploy hash at the end of a deploy path (`.../deploy/ad367d8a2ede`).
pub fn deploy_hash(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_header_and_argv() {
        let l = "== 2026-11-21T22:08:56Z strixfold rank 0 on node0: run strixfold-20261121T220855Z, conf /home/u/.config/strixfold/serve.conf (070eb2add70b), deploy /home/u/tfp/deploy/ad367d8a2ede";
        assert_eq!(
            parse(l),
            Some(Line::RunStart {
                rank: 0,
                name: "strixfold-20261121T220855Z".into(),
                conf: "/home/u/.config/strixfold/serve.conf".into(),
                deploy: "/home/u/tfp/deploy/ad367d8a2ede".into(),
            })
        );
        assert_eq!(
            deploy_hash("/home/u/tfp/deploy/ad367d8a2ede"),
            "ad367d8a2ede"
        );
        let a = "== argv: /x/tensorfold-native serve /m --host 0.0.0.0 --port 8731 --name d --context 1048527 --parallel 4 --max-tokens 32768 --prompt-cache-gib 2 --thinking";
        assert_eq!(
            parse(a),
            Some(Line::Argv {
                port: Some(8731),
                parallel: Some(4),
                context: Some(1048527),
                prompt_cache_gib: Some(2.0),
            })
        );
        assert_eq!(
            parse("== 2026-11-10T07:45:59Z rank 0 exited with status 0"),
            Some(Line::Exit { rank: 0, status: 0 })
        );
    }

    #[test]
    fn serving() {
        let l = "[tensorfold] serving deepseek-v4.1-flash at http://0.0.0.0:8731/v1 (sampling: temperature 1.0, draft_temperature 0.9; drafts: on; context: 1046510; loaded in 51.0s)";
        assert_eq!(
            parse(l),
            Some(Line::Serving {
                model: "deepseek-v4.1-flash".into(),
                loaded_s: Some(51.0)
            })
        );
    }

    #[test]
    fn request_trio() {
        let p = "info: lanes: step 76 rank 0 prompt pass: 6847 tokens, slot 2, extent at 0, 8192 rows; from 0: no kept state; 1 other stream(s) live; collectives (stream idle): staged 408, cable1 408, cable2 391, host 63";
        assert_eq!(
            parse(p),
            Some(Line::PromptPass {
                tokens: 6847,
                slot: 2,
                resume: Resume::None,
                others: 1
            })
        );
        let r = "info: lanes: step 9 rank 0 prompt pass: 9000 tokens, slot 0, extent at 0, 2048 rows; resumes state 3 at 8192, moved from 1; 0 other stream(s) live; collectives";
        assert!(matches!(
            parse(r),
            Some(Line::PromptPass {
                resume: Resume::RamMoved,
                ..
            })
        ));
        let d = "info: lanes: step 9 rank 0 prompt pass: 9000 tokens, slot 0, extent at 0, 2048 rows; resumes disk state at 8192; 0 other stream(s) live; collectives";
        assert!(matches!(
            parse(d),
            Some(Line::PromptPass {
                resume: Resume::Disk,
                ..
            })
        ));
        let c = "info: prompt cache: 7 tokens, resumed at 0, kept 0; 232 states, 1479 MiB and 0 MiB spare of 2048 MiB (hits 306, misses 556, evicted 0, refused 0, failed 0)";
        assert_eq!(
            parse(c),
            Some(Line::PromptCache {
                tokens: 7,
                resumed_at: 0,
                states: 232,
                mib: 1479,
                cap_mib: 2048,
                hits: 306,
                misses: 556,
                evicted: 0,
                refused: 0,
                failed: 0
            })
        );
        let done = "[tensorfold] done chatcmpl-abf7 prompt=16 cached=0 thinking=False effort=none tokens=8 sha=6243895c315c finish=stop rounds=2 accepted=6/10";
        assert_eq!(
            parse(done),
            Some(Line::Done {
                id: "chatcmpl-abf7".into(),
                prompt: 16,
                cached: 0,
                thinking: false,
                effort: "none".into(),
                tokens: 8,
                finish: "stop".into(),
                rounds: 2,
                accepted: 6,
                drafted: 10
            })
        );
        let looped = "[tensorfold] done cmpl-1 prompt=5 cached=0 thinking=True effort=high tokens=3 sha=ab finish=length loop=2 rounds=1 accepted=1/2";
        assert!(matches!(
            parse(looped),
            Some(Line::Done {
                rounds: 1,
                accepted: 1,
                drafted: 2,
                thinking: true,
                ..
            })
        ));
    }

    #[test]
    fn access_ended_refused() {
        assert_eq!(
            parse(r#"[tensorfold] 10.0.0.5 "POST /v1/chat/completions HTTP/1.1" 200 -"#),
            Some(Line::Access {
                client: "10.0.0.5".into(),
                method: "POST".into(),
                path: "/v1/chat/completions".into(),
                status: 200
            })
        );
        assert_eq!(
            parse(
                "[tensorfold] ended chatcmpl-2a reason=client-left prompt=60 tokens=18 after=0.79s"
            ),
            Some(Line::Ended {
                id: "chatcmpl-2a".into(),
                reason: "client-left".into(),
                prompt: 60,
                tokens: 18,
                after_s: 0.79
            })
        );
        assert!(matches!(
            parse("[tensorfold] refused chatcmpl-8b: image input: image file is truncated"),
            Some(Line::Refused { .. })
        ));
        assert_eq!(
            parse("[tensorfold] stream error: PeerLost"),
            Some(Line::StreamError("PeerLost".into()))
        );
    }

    #[test]
    fn disk_and_misc() {
        assert_eq!(
            parse(
                "info: sf-disk: rank=0 event=save result=written tokens=12288 blocks=6 new_blocks=6 bytes=17621800 ms=77 key=d4"
            ),
            Some(Line::DiskSave {
                rank: 0,
                tokens: 12288,
                bytes: 17621800,
                ms: 77.0
            })
        );
        assert!(
            parse(
                "info: sf-disk: rank=0 event=save result=queued tokens=12288 new_blocks=6 key=d4"
            )
            .is_none()
        );
        assert_eq!(
            parse(
                "info: sf-disk: rank=0 event=blocks result=written tokens=0 count=6 bytes=10936848 ms=48"
            ),
            Some(Line::DiskBlocks {
                rank: 0,
                count: 6,
                bytes: 10936848,
                ms: 48.0
            })
        );
        assert_eq!(
            parse(
                "info: sf-disk: rank=0 event=resume result=hit tokens=24576 bytes=28558840 ms=88 rank0=ok rank1=ok key=7b"
            ),
            Some(Line::DiskResume {
                rank: 0,
                hit: true,
                tokens: 24576,
                bytes: 28558840,
                ms: 88.0
            })
        );
        assert_eq!(
            parse(
                "info: dsv41: the kept state of 65536 tokens at 40960 goes stale: a placement needed its room (oldest first)"
            ),
            Some(Line::Stale { tokens: 65536 })
        );
        assert_eq!(
            parse(
                "info: lanes: step 2311 a prompt of 476 tokens cuts in; the pass in slot 0 waits at 4096 of 15863"
            ),
            Some(Line::CutIn {
                tokens: 476,
                waits_at: 4096,
                of: 15863
            })
        );
        assert_eq!(
            parse("info: encoder: pictures=1 encoded=1 cached=0 skipped=0 ms=312"),
            Some(Line::Encoder {
                pictures: 1,
                encoded: 1,
                cached: 0,
                ms: 312.0
            })
        );
        assert_eq!(
            parse(
                "info: structured reply: 28 rows drawn under its grammar, 145.4 us a row on the host (796 token checks, 5 full masks)"
            ),
            Some(Line::Structured {
                rows: 28,
                us_per_row: 145.4
            })
        );
        assert_eq!(
            parse(
                "warning: collectives: rank 0 tcp_host waited 21 ms for its GPU (chunk, prompt_engram layer 1): Aborted"
            ),
            Some(Line::GpuWait { rank: 0, ms: 21.0 })
        );
        assert_eq!(
            parse(
                "info: lanes: step 2404 rank 1 restores state 3 at 14336 into slot 0, extent 40960 from 40960"
            ),
            Some(Line::Restore)
        );
        assert_eq!(parse(r#"{"round_logits": 1, "round": 2}"#), None);
    }

    #[test]
    fn cut_lanes_line_is_dropped() {
        let l = r#"info: lanes: step 5 rank 0 prompt pass: 300 tokens, slot 1, extent at [tensorfold] 1.2.3.4 "POST /v1/completions HTTP/1.1" 200 -"#;
        // The access part wins only if the pass pattern cannot match; here the pass is cut, so it is dropped.
        assert!(!matches!(parse(l), Some(Line::PromptPass { .. })));
    }
}
