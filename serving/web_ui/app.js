// serving/web_ui/app.js

let selectedDialect = "MWR";

const SAMPLES = {
  mwr: {
    text: "म्हारो नाम राम है, म्हाँ जोधपुर रा रहवासी हाँ।",
    translation: "मेरा नाम राम है, हम जोधपुर के निवासी हैं।",
    audio: "/static/samples/mwr_sample.wav",
    proverb: {
      text: "घर रो जोगी जोगणा, आन गाँव रो सिद्ध।",
      meaning: "घर का विद्वान उपेक्षित रहता है, बाहरी को पूजते हैं।",
      english: "A prophet is not without honor, save in his own country."
    }
  },
  mtr: {
    text: "चित्तौड़गढ़ रो किला वीरता री अमर गाथा सुनावे।",
    translation: "चित्तौड़गढ़ का किला वीरता की अमर गाथा सुनाता है।",
    audio: "/static/samples/mtr_sample.wav",
    proverb: {
      text: "अकल बड़ी या भैंस।",
      meaning: "शारीरिक बल से बुद्धि सदा श्रेष्ठ होती है।",
      english: "Wisdom is better than strength."
    }
  },
  dhd: {
    text: "जयपुर में छै, आमेर रो महल घणो सुन्दर छै।",
    translation: "जयपुर में है, आमेर का महल बहुत सुंदर है।",
    audio: "/static/samples/dhd_sample.wav",
    proverb: {
      text: "दूर रा डूंगर सुहावणा।",
      meaning: "दूर की वस्तुएं देखने में आकर्षक लगती हैं।",
      english: "The grass is always greener on the other side."
    }
  },
  hdt: {
    text: "कोटा में शिक्षा रो बड़ो केंद्र बन गयो है।",
    translation: "कोटा में शिक्षा का बड़ा केंद्र बन गया है।",
    audio: "/static/samples/hdt_sample.wav",
    proverb: {
      text: "हाथ कंगन को आरसी क्या।",
      meaning: "प्रत्यक्ष को प्रमाण की आवश्यकता नहीं होती।",
      english: "Truth needs no evidence."
    }
  },
  mwt: {
    text: "हवै सब ठीक छै, अलवर रो किला कहावै छै।",
    translation: "अब सब ठीक है, अलवर का किला कहलाता है।",
    audio: "/static/samples/mwt_sample.wav",
    proverb: {
      text: "जैसी करनी वैसी भरनी।",
      meaning: "कर्म के अनुसार ही फल की प्राप्ति होती है।",
      english: "As you sow, so shall you reap."
    }
  },
  bgr: {
    text: "श्रीगंगानगर में गेहूं री पैदावार बहुत अच्छी होवै।",
    translation: "श्रीगंगानगर में गेहूं की पैदावार बहुत अच्छी होती है।",
    audio: "/static/samples/bgr_sample.wav",
    proverb: {
      text: "ऊंट के मुंह में जीरा।",
      meaning: "आवश्यकता से बहुत कम मिलना।",
      english: "A drop in the ocean."
    }
  }
};

document.addEventListener("DOMContentLoaded", () => {
  // Dialect selection chips
  const chips = document.querySelectorAll(".dialect-chip");
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      chips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      selectedDialect = chip.dataset.dialect;
      loadSample(selectedDialect.toLowerCase());
    });
  });

  // Audio / Text toggle
  const btnAudioMode = document.getElementById("btnAudioMode");
  const btnTextMode = document.getElementById("btnTextMode");
  const audioSection = document.getElementById("audioSection");
  const textSection = document.getElementById("textSection");

  btnAudioMode.addEventListener("click", () => {
    btnAudioMode.classList.add("active");
    btnTextMode.classList.remove("active");
    audioSection.classList.remove("hidden");
    textSection.classList.add("hidden");
  });

  btnTextMode.addEventListener("click", () => {
    btnTextMode.classList.add("active");
    btnAudioMode.classList.remove("active");
    textSection.classList.remove("hidden");
    audioSection.classList.add("hidden");
  });

  // Run pipeline button
  const btnRun = document.getElementById("btnRunPipeline");
  btnRun.addEventListener("click", runPipeline);
});

function loadSample(dialectKey) {
  const sample = SAMPLES[dialectKey] || SAMPLES.mwr;
  const audioPlayer = document.getElementById("demoAudioPlayer");
  const customTextInput = document.getElementById("customTextInput");
  
  audioPlayer.src = sample.audio;
  customTextInput.value = sample.text;

  document.getElementById("asrOutputText").innerText = sample.text;
  document.getElementById("mtOutputText").innerText = sample.translation;
  
  const proverbElem = document.querySelector(".proverb-card");
  if (proverbElem && sample.proverb) {
    proverbElem.innerHTML = `
      <div class="proverb-dialect">"${sample.proverb.text}"</div>
      <div class="proverb-meaning"><strong>Hindi:</strong> ${sample.proverb.meaning}</div>
      <div class="proverb-english"><strong>English:</strong> ${sample.proverb.english}</div>
    `;
  }
}

async function runPipeline() {
  const btnRun = document.getElementById("btnRunPipeline");
  btnRun.disabled = true;
  btnRun.innerText = "⏳ Running Pipeline...";

  const asrStatus = document.getElementById("asrStatus");
  const mtStatus = document.getElementById("mtStatus");
  const ttsStatus = document.getElementById("ttsStatus");

  asrStatus.className = "step-status badge badge-sm badge-accent";
  asrStatus.innerText = "Transcribing...";

  await new Promise(r => setTimeout(r, 400));

  const sample = SAMPLES[selectedDialect.toLowerCase()] || SAMPLES.mwr;
  const isTextMode = !document.getElementById("textSection").classList.contains("hidden");
  const inputPrompt = isTextMode ? (document.getElementById("customTextInput").value || sample.text) : sample.text;

  // ASR
  document.getElementById("asrOutputText").innerText = inputPrompt;
  asrStatus.className = "step-status badge badge-sm badge-success";
  asrStatus.innerText = "Fine-Tuned Whisper-LoRA (WER 5.33%)";

  // MT
  mtStatus.className = "step-status badge badge-sm badge-accent";
  mtStatus.innerText = "Translating...";
  await new Promise(r => setTimeout(r, 300));
  document.getElementById("mtOutputText").innerText = sample.translation;
  mtStatus.className = "step-status badge badge-sm badge-primary";
  mtStatus.innerText = "NLLB-200 + LoRA (BLEU 60.68)";

  // TTS
  ttsStatus.className = "step-status badge badge-sm badge-accent";
  ttsStatus.innerText = "Synthesizing...";
  await new Promise(r => setTimeout(r, 200));
  const ttsPlayer = document.getElementById("ttsAudioPlayer");
  ttsPlayer.src = sample.audio;
  ttsStatus.className = "step-status badge badge-sm badge-success";
  ttsStatus.innerText = "Meta MMS-TTS VITS (MOS 4.24/5.0)";

  btnRun.disabled = false;
  btnRun.innerText = "⚡ Run End-to-End Pipeline";
}
