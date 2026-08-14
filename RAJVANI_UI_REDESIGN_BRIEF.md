# Rajvani Demo App — UI Redesign Brief

**For:** whichever coding agent (Claude Code, Codex, Gemini CLI) owns `serving/demo_app/`
**Context:** this is the Stage 4 live-panel demo surface for a ₹90L BHASHINI hackathon prize. Right now it reads as a generic dark admin-dashboard template with a broken data-rendering bug in it. This brief fixes both: a P0 bug pass, then a full visual identity redesign.

---

## 0. P0 — Fix before any redesign work starts

**The transfer-matrix heatmap and every results table are rendering the literal string `[object Object]` instead of data** (visible across the Transfer Matrix, Evaluation & Human Feedback, and Model Improvement panels). This is not a styling issue — somewhere a component is being handed a JS object and rendering it with `{value}` instead of accessing the actual field (e.g. `{value.wer}` or a proper cell renderer / `JSON.stringify` fallback). Find every table/matrix component currently doing this and fix the data mapping before any visual work below — a beautifully redesigned table that still shows `[object Object]` is worse than the current version, because it'll look intentional.

**Acceptance check:** every cell in the Transfer Matrix heatmap and every row in the Evaluation tables renders an actual number, percentage, or explicit `N/A`/`—` — never a stringified object.

---

## 1. Why the current UI reads as generic (diagnosis, not just "make it prettier")

- **No visual identity tied to the subject matter.** Navy-on-black + default badge pills is the default look of every AI-dashboard template — nothing here signals Rajasthan, dialect diversity, or cultural preservation specifically.
- **Empty states dominate the layout.** The Live Pipeline tab (Image 1) is five large bordered rectangles, most of them blank, waiting for a pipeline run — the *absence* of content is the first thing a viewer sees, not the product.
- **Status signaling is inconsistent and alarming.** A red "Bhashini Offline" badge sits in the header of every single tab, permanently, at the same visual weight as "System Ready." A judge's eye goes straight to the red dot before anything else.
- **Honesty about the TTS fallback is presented as a warning, not a design choice.** "Local TTS Ready (Hindi Fallback)" and "TTS Voice MOS: Pending" are correct and important to disclose (see the earlier limitations review) — but right now they read as *broken*, not *transparently in-progress*. Good design can make "still training" look like a deliberate, tracked state instead of an error.
- **Typography has no hierarchy.** Section labels, body copy, badge text, and table headers are all close to the same small size and weight — nothing tells the eye where to look first.
- **The one genuinely good screen (Proverb & Idiom KB, Image 3) proves the rest can look this good** — card-based, real content, actual Devanagari script given room to breathe. Use it as the internal reference point for the other three tabs, not an outlier.

---

## 2. Visual identity direction

**Concept:** *"Desert manuscript, not SaaS dashboard."* Draw restrained visual cues from Rajasthani miniature painting, block-print textiles, and haveli architecture — warm earth tones instead of generic tech-navy, a serif/display pairing for headers that nods to manuscript typography, and Devanagari script treated as a first-class visual element rather than small gray text. Keep it professional and judge-appropriate — this is restraint applied to a specific cultural palette, not a kitsch theme with camels and turbans.

### 2.1 Color palette (replace the current navy/slate system entirely)

| Role | Color | Notes |
|---|---|---|
| Background (base) | Deep indigo-charcoal `#1A1523` (not pure black/navy) | Evokes Jodhpur's "blue city" without being literal |
| Surface / card | `#241C30` with a 1px warm-toned hairline border `#3D2F4A` | Replace the current flat slate-800 boxes |
| Primary accent | Terracotta / sindoor red `#C4502A` | Primary buttons, active tab underline, key numbers |
| Secondary accent | Marigold gold `#E8A83C` | Highlights, "field-verified" badges, hover states |
| Tertiary accent | Desert sand `#D9B48F` | Muted labels, secondary text on dark |
| Success | Muted sage `#7A9B76` (not saturated green) | "Ready" states — calmer than a bright green dot |
| Pending/in-progress | Marigold gold, not red | "TTS fine-tuning in progress" should read as *active work*, not failure |
| Critical/error only | Reserved terracotta-red `#B33A2E`, used sparingly | Only for actual failures, not for expected/flagged fallback states |
| Text primary | Warm off-white `#F2E9DD` | Not pure white — reduces the "cold dashboard" feel |
| Text secondary | `#A99A8C` | |

Do not use bright saturated red for anything that is a known, disclosed, in-progress state (like the Hindi TTS fallback) — reserve red exclusively for genuine failures. This single change fixes most of the "looks broken" impression.

### 2.2 Typography

- **Headers/section titles:** a serif or slab-serif display face with some warmth — e.g. `"Fraunces"`, `"Lora"`, or `"Tiro Devanagari Hindi"` for anything mixing Latin+Devanagari headers. Avoid another default geometric sans like Inter/Roboto for headers — that's most of what makes this look templated.
- **Body/UI text:** a clean, highly-legible sans for Latin text (`"Inter"` or `"IBM Plex Sans"` is fine here — the *header* face is what needs to change).
- **Devanagari/dialect script text:** use `"Noto Sans Devanagari"` or `"Tiro Devanagari"` at a **minimum 1.15× the size of adjacent Latin text** — Devanagari needs more room to read clearly than Latin at the same pixel size, and right now dialect text (Image 3) is being set at the same size as English labels around it, which under-serves the actual content this project is about.
- Establish a real type scale (e.g. 12/14/16/20/28/36px) and stick to it — right now most text in Images 1, 2, 4 sits in a narrow 12–14px band with no hierarchy.

### 2.3 Motif (use sparingly — accents, not backgrounds)

- A thin block-print-inspired border pattern (simple repeating geometric motif, single color, low opacity) as a top border on major section cards — not a full background texture, which would hurt legibility.
- Card corner treatment: small consistent radius (8px), not the current mix of sharp and rounded corners across different components.

---

## 3. Global layout changes

- **Collapse the header.** Currently the full title + subtitle + status-badge row + model/dataset chips repeat identically at the top of every tab (Images 1–4), consuming ~15% of vertical space on every screen. Make it a single persistent slim top bar (~56px) with the product name + tab nav; move the status badges and model/dataset metadata into a collapsible "System Info" panel, not permanent header real estate.
- **Status badges → single compact health strip**, ordered by actual importance, using the new color system (2.1): `● ASR: Ready` `● MT: Ready` `◐ TTS: Fine-tuning (Hindi fallback active)` — the last one in gold, not red, with a tooltip explaining the state rather than a bare "Offline."
- **Tab navigation** should be visually the most prominent nav element on the page (currently it's a thin underlined row easy to miss under the badge clutter) — increase weight, use the terracotta accent for the active tab.

---

## 4. Per-screen redesign

### 4.1 Live Pipeline tab (currently Image 1)

- Replace the five stacked empty bordered rectangles (`01–05`) with a **horizontal step tracker** (like a progress rail) at the top: Speech → ASR → Normalization → Cultural Match → Translation → Synthesis, each step showing a status icon (pending / running / done) — this turns "five blank boxes" into "a visible pipeline," even before a run starts.
- Below the tracker, show step *output* only for steps that have run — collapse/hide empty future steps entirely instead of rendering them as empty bordered boxes waiting to be filled. An empty state should say something (e.g. a light icon + "Run the pipeline to see ASR output here"), never just be blank whitespace inside a border.
- Human-in-the-loop transcript correction section: keep the function, but visually separate it as a distinct card with its own header treatment — right now it reads as a continuation of the pipeline rather than a separate reviewer tool.
- Latency stats: move from a prominent 2×2 grid at the bottom to a slim inline strip — these numbers matter for engineers, not for a judge's first impression.

### 4.2 Transfer Matrix tab (currently Image 2)

- Fix the `[object Object]` bug first (Section 0).
- Once fixed: render the 6×6 matrix as an actual **color-coded heatmap**, WER mapped to a color gradient from sage (low/good) through gold through terracotta (high/bad), with the numeric WER inside each cell — not a plain data table.
- `N/A` cells (speaker-disjoint constraint) get a distinct muted diagonal-hatch or dotted-border treatment, with the existing explanatory tooltip kept.
- Add a compact legend under the heatmap (color → WER range).
- Keep the "Zero-Shot / Fine-Tuned Cross-Dialect" toggle, but give it a proper segmented-control look, not a two-button pair with unclear active state (Image 2's toggle is genuinely hard to read which mode is selected).

### 4.3 Proverb & Idiom KB tab (currently Image 3 — the strongest screen)

- Keep the card-grid structure — it works. Enhancements only:
  - Increase Devanagari script size per Section 2.2.
  - Color-code the dialect tag chip (MWR/MTR/DHD/HDT/MWT/BGR) consistently per dialect across the entire app (same color for "MWR" everywhere it appears, in every tab) — right now dialect identity has no consistent color coding anywhere in the UI, which is a missed navigation aid across all four tabs, not just this one.
  - "Field Verified" badge: switch from generic green to the sage success color, and give it a small checkmark icon rather than text-only.

### 4.4 Evaluation & Human Feedback tab (currently Images 4–5)

- Fix all `[object Object]` tables (Section 0).
- The four headline metric cards (WER, CSR, BLEU, chrF, MOS, Latency) are the right idea — keep the card format, but visually distinguish **real numbers** from **provisional/pending** ones: the current "8.4%*" with a tiny asterisk and separate footnote is easy to miss. Instead, give provisional metrics (n=8 dev set) a visible gold "provisional" corner tag directly on the card, not a footnote — see the honest-numbers discussion from the project's own limitations doc; the UI should make the n=8 caveat impossible to miss, not a small disclaimer underneath.
- "TTS Voice MOS: Pending" card — style this as an active/in-progress state (gold, with a small "in progress" icon), not a blank/grayed-out card that looks like a missing feature.
- The architecture diagram (Image 5, top) is currently ASCII text in a plain box (`Dialect Detection → Whisper ASR Model → ...`). Replace with an actual simple SVG/diagram component — boxes and arrows, using the new color system — this is a two-hour fix for a huge visual-credibility jump; ASCII-art-in-a-textbox is one of the more obvious "unfinished prototype" signals in the current UI.
- Human Evaluator sliders (Image 5, bottom): currently truncated labels ("Correctness (0 = Not Rated)"), inconsistent slider-track styling, and a value readout box that's visually disconnected from the slider itself. Redesign as: full untruncated label above each slider, the numeric value shown as a filled pill that sits directly on the slider handle (not a separate box to the side), and a clear 0–5 tick scale beneath.

---

## 5. Component conventions to establish once, reuse everywhere

- **Badge/status pill:** one component, three states (ready/sage, in-progress/gold, error/terracotta-red), consistent padding and icon placement — audit every badge in the current app (there are at least 4 different pill styles across Images 1–5) and consolidate to this one.
- **Card:** one card component (radius, border, padding, header treatment) reused for every panel — pipeline steps, metric cards, proverb entries, evaluation tables all currently use subtly different bordered-box styles.
- **Empty state:** one pattern (icon + one line of guidance text, muted color) used everywhere a panel has no data yet — never a blank bordered rectangle.
- **Dialect color coding:** six fixed colors, one per dialect (MWR/MTR/DHD/HDT/MWT/BGR), defined once in a shared config/theme file and used consistently for every chip, tag, and heatmap axis label across all four tabs.

---

## 6. What NOT to change

- Don't add literal cultural clip-art (camels, forts, turbans) — the goal is a restrained palette/typography identity, not a tourist-brochure aesthetic. A judge who's a linguist or technical reviewer should read this as "considered," not "themed."
- Don't hide or remove the honest provisional-data disclosures (n=8 dev sets, TTS fallback state, pending MOS) — the fix is presenting them with better visual design, not making them less visible. Overcorrecting into hiding known limitations would undo the credibility work from the project's own `LIMITATIONS.md`.

---

## 7. Acceptance checklist

- [ ] Zero instances of `[object Object]` anywhere in the rendered UI.
- [ ] No panel renders as an empty bordered box with no icon/label — every empty state has explicit guidance text.
- [ ] Color palette fully replaced per Section 2.1 — no remaining default slate/navy dashboard colors.
- [ ] Devanagari text renders at ≥1.15× adjacent Latin text size everywhere it appears.
- [ ] Header/status-badge real estate reduced to a single slim bar + collapsible system-info panel, consistent across all four tabs.
- [ ] Transfer Matrix renders as a real color-coded heatmap, not a data table.
- [ ] Architecture diagram is a real SVG diagram, not ASCII text in a box.
- [ ] Dialect color coding is consistent across every tab (same 6 colors, same dialect, everywhere).
- [ ] Provisional/in-progress states (n=8 metrics, TTS fallback, pending MOS) are visually distinct (gold "in progress" treatment) but never hidden or downplayed to the point of being missable.
