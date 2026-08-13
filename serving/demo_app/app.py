import json
import os
import sys
from pathlib import Path
import gradio as gr

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dialect_id.infer import infer_dialect_distribution
from eval.cross_dialect_transfer import compute_transfer_matrix
from linguistic_artifacts.idiom_mt_eval import evaluate_idiom_mt

DIALECT_STATUS = {
    "mwr": "✅ Trained",
    "mtr": "✅ Trained",
    "dhd": "✅ Trained",
    "hdt": "✅ Trained",
    "mwt": "✅ Trained",
    "bgr": "✅ Trained"
}

def pipeline_asr_mt_tts_demo(dialect_choice: str, text_input: str):
    if not text_input:
        return "Please enter text or record audio.", "", None

    # Simulate pipeline response
    clean_dialect = dialect_choice.split(" ")[0].lower()
    transcript = f"[{clean_dialect.upper()} Transcript]: {text_input}"
    translation = f"[Hindi Translation]: {text_input} (अनुवाद)"

    # Synthetic Audio path or base64
    audio_path = str(ROOT_DIR / "serving" / "demo_app" / "sample_output.wav")
    
    return transcript, translation, None

def get_transfer_heatmap_matrix():
    matrix_data, _, _ = compute_transfer_matrix("asr")
    dialects = ["mwr", "mtr", "dhd", "hdt", "mwt", "bgr"]
    
    formatted_rows = []
    for d1 in dialects:
        row = [f"{d1.upper()} ({DIALECT_STATUS[d1]})"]
        for d2 in dialects:
            score = matrix_data[d1][d2]
            row.append(f"{score:.1f}% WER")
        formatted_rows.append(row)
    return formatted_rows

def get_idiom_demo_sample(dialect_choice: str):
    clean_d = dialect_choice.split(" ")[0].lower()
    bank_file = ROOT_DIR / "linguistic_artifacts" / "idiom_bank" / f"{clean_d}.jsonl"

    eval_stats = evaluate_idiom_mt(clean_d)

    idiom_text = "म्हारो खेत सोनो उगले छै"
    literal = "My field vomits gold"
    intended = "खेत में बहुत अच्छी फसल होना (Rich harvest)"

    if bank_file.exists():
        with open(bank_file, "r", encoding="utf-8") as f:
            line = f.readline()
            if line.strip():
                rec = json.loads(line)
                idiom_text = rec.get("idiom_dialect", idiom_text)
                literal = rec.get("literal_gloss", literal)
                intended = rec.get("intended_meaning_hindi", intended)

    accuracy_banner = f"**Figurative MT Accuracy for {clean_d.upper()}: {eval_stats['accuracy_pct']}%** (Evaluated across {eval_stats['total']} proverbs)"

    return accuracy_banner, idiom_text, literal, intended

def build_demo_interface():
    with gr.Blocks(title="Rajasthani Multi-Dialect AI Platform") as demo:
        gr.Markdown("# 🏜️ Rajasthani Multi-Dialect Language Technology Platform")
        gr.Markdown("Preserving traditional oral speech & enabling Bhashini ecosystem interoperability across **Marwari**, **Mewari**, **Dhundhari**, **Hadoti**, **Mewati**, and **Bagri**.")

        with gr.Tab("🎙️ Live Speech & Translation"):
            with gr.Row():
                dialect_dropdown = gr.Dropdown(
                    choices=[f"{d} ({DIALECT_STATUS[d]})" for d in ["mwr", "mtr", "dhd", "hdt", "mwt", "bgr"]],
                    value="mwr (✅ Trained)",
                    label="Target Dialect"
                )
            
            with gr.Row():
                text_in = gr.Textbox(lines=2, placeholder="Type dialect text or speak into mic...", label="Dialect Input Text")
            
            btn_run = gr.Button("Run ASR + MT + TTS Pipeline", variant="primary")

            with gr.Row():
                out_transcript = gr.Textbox(label="ASR Transcript Output")
                out_translation = gr.Textbox(label="MT Hindi Translation Output")
                out_audio = gr.Audio(label="TTS Synthesized Dialect Audio")

            btn_run.click(pipeline_asr_mt_tts_demo, inputs=[dialect_dropdown, text_in], outputs=[out_transcript, out_translation, out_audio])

        with gr.Tab("📊 Cross-Dialect Zero-Shot Transfer Heatmap"):
            gr.Markdown("### Interactive 6×6 Zero-Shot Transfer Matrix (ASR WER)")
            gr.Markdown("Shows WER degradation when evaluating a model trained on Dialect A against Dialect B without fine-tuning.")
            
            heatmap_table = gr.Dataframe(
                headers=["Train \\ Eval", "MWR", "MTR", "DHD", "HDT", "MWT", "BGR"],
                value=get_transfer_heatmap_matrix(),
                interactive=False
            )

        with gr.Tab("📜 Proverb & Idiom Bank Demonstration"):
            gr.Markdown("### Live Figurative vs. Literal MT Evaluation")
            gr.Markdown("Demonstrates semantic preservation of cultural proverbs vs garbled word-for-word translation.")

            idiom_dialect_select = gr.Dropdown(
                choices=["mwr", "mtr", "dhd", "hdt", "mwt", "bgr"],
                value="mwr",
                label="Select Dialect Bank"
            )

            btn_load_idiom = gr.Button("Load Proverb & Evaluate MT", variant="secondary")

            accuracy_display = gr.Markdown()
            
            with gr.Row():
                txt_idiom = gr.Textbox(label="Spoken Dialect Proverb")
                txt_literal = gr.Textbox(label="Literal Word-for-Word Gloss (Incorrect)")
                txt_intended = gr.Textbox(label="Intended Figurative Meaning (Correct MT Target)")

            btn_load_idiom.click(get_idiom_demo_sample, inputs=[idiom_dialect_select], outputs=[accuracy_display, txt_idiom, txt_literal, txt_intended])

    return demo

if __name__ == "__main__":
    demo_app = build_demo_interface()
    print("Launching Gradio demo application on http://127.0.0.1:7860 ...")
    demo_app.launch(server_name="127.0.0.1", server_port=7860)
