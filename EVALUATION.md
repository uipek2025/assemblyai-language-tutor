# How I'd win the customer's technical evaluation

The deal is a rip-and-replace: a language-learning voice agent moving from another provider to AssemblyAI. Generic benchmarks don't win deals like this. What wins is proving, on the customer's own learners, that AssemblyAI fixes the problem that made them look in the first place, with a method their data team trusts.

## 0. Start with why they're switching

Nobody rips out a working provider for fun. In the first call with my AE, I'd find the pain and build the evaluation around it. For a language tutor, these are my hypotheses going in:

| Likely pain | How it shows up for learners |
|---|---|
| Foreign words get mangled mid-English ("what does *me gustaría* mean?") | The tutor answers the wrong question |
| Learners are cut off while hesitating | They feel rushed and quit the lesson |
| Mistakes get "auto-corrected" into the right phrase | The tutor praises errors, and learning value drops |
| Slow turn-taking | It feels like a phone menu, not a conversation |
| Language coverage | They can't launch the languages on their roadmap |
| Cost or concurrency at scale | Growth hurts their margins |

I'd also find out what they run today, because it decides the conversation. If they use a component STT inside their own stack, we compare STT to STT and the migration is a provider swap. If they use an all-in-one speech-to-speech agent, the Voice Agent API is the like-for-like comparison.

## 1. Agree success criteria before anyone runs anything

Agree these in writing with their data-science and product leads, including the bars. That prevents goalposts moving once the results are in.

| Metric | Definition | Why it matters here |
|---|---|---|
| Target-span accuracy | Foreign word or phrase transcribed exactly (with and without accents) | The part of the utterance that matters. WER hides it behind carrier words |
| Premature cut-off rate | Turn ended before the learner finished, reported overall and on hesitations | The most visible learner failure |
| End of speech → final transcript | p50 and **p90**, on the audio timeline | Tail latency is what breaks conversation flow |
| Grader false-accept / false-reject | Wrong attempts praised; right attempts rejected | Learning value versus learner frustration |
| Language-ID accuracy | Per turn | Telling an attempt from a question |
| WER | Normalised, one normaliser for every vendor | Secondary, but their DS team will want it |
| Cost per learner-hour | STT session + LLM tokens + TTS characters | The business case |
| Reliability | Errors and disconnects at production concurrency | The risk case |

## 2. A test set that looks like their learners

- **Source.** Production recordings, used with consent and under their retention policy. Not clean read speech, and not TTS-generated audio. Neither sounds like a nervous beginner.
- **Stratify** by native × target language pair, proficiency level (weighted to beginners), device and noise, and utterance type: question, code-switched question, practice attempt, hesitation. If they serve children, that changes the compliance conversation too.
- **Size.** A few hundred utterances per key stratum gives roughly ±3–5 points on rates. Every number gets a confidence interval.
- **Ground truth.** Humans transcribe what was *said*, not what should have been said. Code-switched spans stay in the original language: AssemblyAI's own evaluation guide warns that transcribers tend to translate them. Use two annotators plus adjudication, and audit insertions before trusting WER.
- **Labelled attempts.** Native speakers mark practice attempts correct or incorrect, including deliberate classic mistakes (*librería* for *biblioteca*). Grader false-accept and false-reject rates come from these labels.

## 3. Flawed methodology I'd catch before it shapes the outcome

| Flaw | Why it distorts | Fix |
|---|---|---|
| Incumbent's transcripts used as "ground truth" | Biased toward the incumbent by construction | Human references |
| WER only | Carrier words dominate and the foreign word disappears | Target-span accuracy as the headline metric |
| Different text normalisation per vendor | Formatting differences show up as fake accuracy gaps | One normaliser, applied to both sides |
| Uploading files faster than real time | Hides latency and changes streaming behaviour | Real-time pacing, 50 ms chunks, trailing silence |
| Time-to-first-token as "latency" | Gameable: some systems emit before speech | Speech end → final transcript, p90 |
| Speech end taken from each vendor's own timestamps | Each vendor grades its own homework | Vendor-independent detector or manual annotation |
| One vendor tuned, the other on defaults | Not a fair fight, and DS teams can tell | Each vendor's recommended settings, and sweep both |
| Boosting/context judged on WER alone | WER improves while the tutor praises mistakes | Report the false-accept rate alongside |
| Small samples, no intervals | Noise read as signal | Confidence intervals and enough data |

## 4. Run it

The harness in this repo does the mechanics:
- one provider interface
- identical audio, pacing, normaliser and scoring for every vendor
- per-utterance CSV, so every number can be traced back to a recording

The plan:
1. **Baselines.** AssemblyAI out of the box, AssemblyAI tuned for learners, and the incumbent on its best settings.
2. **Find the operating point.** Sweep turn detection (`practice.max_turn_silence`) and show the curve of cut-off rate against p90 latency. Their product team picks the point. Boosting is decided the same way, on false-accept rates.
3. **End to end.** Run scripted lessons through the full agent and measure voice-to-voice latency and lesson completion, not just transcripts.
4. **Mid-point readout.** No surprises at the end; adjust scope if something unexpected shows up.

## 5. Prove it in production

Shadow mode first: dual-stream the same audio and compare offline. Then an A/B on 5–10% of sessions behind a feature flag, then ramp. Product metrics decide the result:
- lesson completion
- "say that again" rate
- complaints about wrong "try again"s
- session length
- D7 retention

Rollback is flipping the flag.

## 6. Make switching low-risk, and the business case obvious

- **Migration.** The provider sits behind an interface (as in this repo). Write a parity checklist of every feature they use today, run shadow mode, and keep a flag-based rollback.
- **Cost per learner-hour.**
  - STT-3.7-Realtime-Agent: $0.45/hr of session, keyterms included
  - TTS-1: $15 per million characters
  - LLM Gateway: per token, at no markup
  - TTS-1 has no spend commitment or concurrency tier

  Compare that against the incumbent's TCO, including the engineering time to maintain multiple vendors.
- **Security and compliance.** SOC 2 report, DPA, GDPR, EU endpoints for STT, LLM and TTS, data-retention controls, and how minors' data is handled if relevant. Run this track in parallel with the commercial track, not after it.
- **Honest gaps.** TTS-1 speaks 6 languages today, while the STT understands 32. If their roadmap has Japanese, we plan for it up front rather than let it surface in week 3. Being straight about gaps is what gets a data team to trust the rest of the readout.

## 7. Multi-thread the account, with my AE

| Stakeholder | What they need to see | Proof |
|---|---|---|
| Developers | It's easy to build with and fast in their stack | A working reference agent, a swap guide, SDK and docs walkthrough |
| Data science / evaluation | A fair, reproducible method on their data | Harness, report with confidence intervals, per-item CSV |
| Product / learning science | Better learner outcomes, and a choice of operating point | Sweep curve, false-accept/false-reject rates, A/B design |
| Security / legal | Compliance and data residency | Completed questionnaire, EU endpoints, DPA |
| CTO / exec | Lower risk and cost, a roadmap fit | TCO, migration plan, multilingual roadmap |

## 8. Timeline (about three weeks)

| Week | Focus |
|---|---|
| 0 | Discovery, written success criteria, data access, security questionnaire started |
| 1 | Test set and labels; harness on their data; baselines |
| 2 | Tuning, sweeps, end-to-end runs; mid-point readout |
| 3 | Final readout with a recommendation, A/B plan and commercial proposal |
