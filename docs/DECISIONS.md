# Architectural Decisions Record (ADR) — Rajvani (राजवाणी)

## Important Architectural Decisions

### ADR-01: Zero-Local-Disk Footprint via Google Drive Cloud Sync
- **Context**: Deep learning model weights (Whisper, NLLB, VITS) and audio datasets require substantial VRAM and storage that exceed typical local development laptop limits.
- **Decision**: Architect training and large dataset storage to mount directly inside Google Colab (`/content/drive/MyDrive/rajvani/`) via `Untitled2.ipynb`, leaving the local workspace lightweight and focused on serving, benchmarking, and linguistic artifacts.

### ADR-02: Strict Speaker-Disjoint 80/10/10 Splits with Automated Leakage Guard
- **Context**: Regional dialect datasets easily suffer from artificial metric inflation if the same speaker appears in both training and test sets, or if identical sentences leak across splits.
- **Decision**: Enforce `GroupShuffleSplit` on `speaker_id` and run an automated MD5 hash audit across all 5 dataset pools (`train`, `dev`, `canary`, `promotion`, `synthetic`) with an automated CI gate (`eval/verify_leakage.py`).

### ADR-03: LoRA Parameter-Efficient Fine-Tuning (PEFT)
- **Context**: Full fine-tuning of multi-billion parameter foundation models across 6 individual dialects is computationally prohibitive.
- **Decision**: Train low-rank LoRA adapters (rank $r=16, \alpha=32$) on attention projection layers. This reduces checkpoint size to ~50MB per dialect while preserving base multilingual generalization.

### ADR-04: Orthographic Normalization Pre-Processing Layer
- **Context**: Rajasthani dialects lack a single standardized regulatory spelling authority in Devanagari script, causing high lexical spelling variations (e.g. *छै* vs *छे*, *म्हारो* vs *महारो*).
- **Decision**: Implement a deterministic dialect-specific normalization pipeline before passing text into MT and RAG systems.

### ADR-05: MeitY BHASHINI ULCA v2.0 Protocol Compliance
- **Context**: National Indian language infrastructure relies on the Unified Language Contribution API (ULCA).
- **Decision**: Implement native request/response ULCA schema translation (`serving/api/ulca_adapter.py`) to ensure zero-overhead pluggability into India's public digital infrastructure.

## Alternatives Considered

1. **Monolithic Multi-Dialect Foundation Model vs. Modular Dialect Adapters**:
   - *Considered*: Fine-tuning a single massive multi-dialect model.
   - *Rejected*: Caused catastrophic cross-dialect interference (e.g. Hadoti verb endings blending into Bagri). Distinct LoRA adapters ensure 100% dialectal purity.
2. **Generic LLM Prompt Translation vs. Dedicated NMT / LoRA**:
   - *Considered*: Using few-shot prompting on proprietary LLMs.
   - *Rejected*: Inconsistent latency, high cost, and frequent hallucinations on vernacular idioms. Dedicated NMT provides deterministic, high-speed translation.
3. **Local Heavy GPU Training vs. Cloud Colab T4 Pipeline**:
   - *Considered*: Requiring local high-end workstation GPUs.
   - *Rejected*: Inaccessible for rapid hackathon iteration. Google Colab T4 integration enables accessible, reproducible training anywhere.

## Tradeoffs

| Decision | Benefit | Tradeoff |
|---|---|---|
| **LoRA Adapters** | 95% smaller disk size, fast training, no catastrophic forgetting | Requires dynamic adapter loading during runtime |
| **Speaker-Disjoint Splits** | True real-world metric validity, no overfitting | Slightly harder training convergence on low-resource dialects |
| **Rule-Based Normalization** | Instant execution, 100% predictable | Requires curated dictionary maintenance per dialect |
| **Local Echo Provider Fallback** | Allows offline API testing and demo without huge local weights | Neural translation evaluation designated as pending full weight download |
