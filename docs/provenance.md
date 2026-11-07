# Source Code and Historical Provenance Audit

This document establishes the digital chain of custody, authorship, and provenance classification for all assets, source code, and configurations in the **StethoSpeech** research repository.

---

## 1. Provenance Classification Framework

To uphold scientific and engineering integrity, all assets are partitioned into four distinct provenance categories:

| Category | Definition | Repository Assets |
|---|---|---|
| **1. Surviving Original Source** | Code and files written directly during the original research study (April–May 2026) that survived verbatim on local disk or authoritative archive. | - [`src/stethospeech/data/mfa_corpus.py`](../src/stethospeech/data/mfa_corpus.py) (refactored from `C:\mini-projet-dl\prepare_mfa_corpus.py`, authored April 29, 2026)<br>- Original May 2026 course report (`StethoSpeech_Deep_Learning_Report.md`)<br>- Raw dataset zip archives in `Downloads` (`Seated-*.zip`, `Mobility-*.zip`)<br>- Pretrained LJSpeech vocoder checkpoint (`900000.pth.tar`) |
| **2. Historical Code Evidenced by Execution Logs** | Code cells authored and executed inside ephemeral Kaggle Linux GPU sessions, whose exact AST, parameters, console stdout, and numerical metrics were recorded in primary conversation event logs. | - 6-layer Transformer training loop (Steps 34, 40, 54, 74, 81 of `CONV_236512e1`)<br>- 29.26M StethoSpeech FFT architecture (Steps 408–438 of `CONV_236512e1`)<br>- Meta CDN model downloader (Step 340)<br>- SpeechT5 synthetic speech generation with deterministic seed (Steps 295, 307)<br>- Fairseq-free mHuBERT Layer 11 extraction (Step 337)<br>- Unit HiFi-GAN vocoder pure-PyTorch implementation (Steps 228, 274, 343) |
| **3. Newly Reconstructed Implementations** | Python scripts reconstructed during the November 2026 recovery phase to restore modular execution from historical logs. | - Recovery scripts in `CAREER-RESEARCH-RECOVERY-R2.zip`<br>- Initial standalone verification scripts (`01_download_pretrained_models.py` through `05_inference_and_unit_vocoding.py`) |
| **4. New Improvements Made for This Release** | Engineering refactoring and additions created specifically for this public research repository release. | - Modular Python package architecture (`stethospeech`) with typed interfaces<br>- Configuration-driven experiment management (`configs/*.yaml`)<br>- Automated test suite (37 passing CPU tests covering data splitting, models, vocoding, CLI integration, and error handling)<br>- GitHub Actions CI pipeline (`.github/workflows/ci.yml`) with strict lint enforcement<br>- Vector architecture schematics (`assets/figures/*.svg`)<br>- Unified CLI entry points (`scripts/train.py`, `scripts/synthesize.py`, `scripts/prepare_corpus.py`)<br>- Hardened checkpoint validation preventing silent corruption or uninitialized random inference<br>- Security sanitization and credential scrubbing |

---

## 2. Forensic Analysis of Historical Workspace (`C:\mini-projet-dl`)

Physical inspection of the local path `C:\mini-projet-dl` confirms:
1. **Total Files:** 2 files (`prepare_mfa_corpus.py`, `README.md`).
2. **Version Control:** No `.git` directory exists locally (`fatal: not a git repository`).
3. **Reason for Decoupling:** The owner faced local Windows compilation hurdles with Montreal Forced Aligner C++ binaries in April 2026 and moved all model training to ephemeral Kaggle cloud GPU containers (`/kaggle/working/`). Local storage was used solely for preliminary file staging and unzipping.
4. **Authentic Historical Code:** `prepare_mfa_corpus.py` (2,970 bytes, SHA-256 `DC4BC371...`) was authored from scratch by **Achouri Youcef Islem** on April 29, 2026 to bridge the gap between raw nested speaker folders and the strict directory pair requirements of Montreal Forced Aligner.

---

## 3. Existing Remote GitHub Repositories Audit

Two GitHub repositories associated with the owner's account (`youcef-ach`) were audited via GitHub API:

1. **`youcef-ach/mini_projet_dl`:**
   - Created: `2026-05-02T22:40:04Z`
   - Commits: Exactly one commit (`b19acaaf`: *"Initial commit"*).
   - Contents: Contains solely `.gitattributes` (66 bytes).
   - Finding: No substantive research code was ever pushed to this repository.
2. **`youcef-ach/dl_mini_project`:**
   - Status: HTTP 404 (Not Found / Non-existent).

**Conclusion:** Neither existing remote repository contains meaningful commit history or historical codebase branches. A clean initial release on a dedicated repository (`stethospeech-nam-reproduction`) represents the honest, technically grounded path.

---

## 4. Preservation of Historical Source Archives

Original research logs, forensic transcripts, and the quarantined `CAREER-RESEARCH-RECOVERY-R1.zip` archive (which contained historical unredacted credentials) remain quarantined in private local storage and are strictly excluded from the release package.

