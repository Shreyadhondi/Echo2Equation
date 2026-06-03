// frontend/script.js
(function () {
  // =========================
  // Config
  // =========================
  const API_BASE = "http://localhost:8000"; // change if needed

  // =========================
  // DOM refs  (NOTE: matches updated index.html)
  // =========================
  const body = document.body;
  const btnRecord = document.getElementById("btn-record");
  const btnPredict = document.getElementById("btn-predict");
  const btnCorrect = document.getElementById("btn-correct");
  const btnRetry = document.getElementById("btn-retry");
  const btnRecordAgain = document.getElementById("btn-record-again");

  const asrInput = document.getElementById("asr-input");     // <input>
  const latexRawEl = document.getElementById("latex-raw");    // <textarea>
  const latexRenderBox = document.getElementById("latex-render");

  const errorBox = document.getElementById("error-box");
  const errorText = document.getElementById("error-text");

  // =========================
  // App state
  // =========================
  let mediaStream = null;
  let mediaRecorder = null;
  let chunks = [];

  let recording = false;
  let lastAudioBlob = null;
  let lastAudioPath = null; // path returned by /transcribe
  let lastTranscript = "";
  let lastLatex = "";
  let lastCorpusId = null;
  let wasRetried = false;

  // =========================
  // Helpers (UI)
  // =========================
  function setState(s) {
    body.dataset.state = s; // "idle" | "listening" | "processing" | "success" | "error"
  }

  function showError(msg) {
    errorText.textContent = msg || "Something went wrong.";
    errorBox.classList.remove("hidden");
    setState("error");
  }

  function hideError() {
    errorBox.classList.add("hidden");
  }

  function setTranscript(text) {
    lastTranscript = text || "";
    asrInput.value = lastTranscript;
    btnPredict.disabled = !lastTranscript.trim();
  }

  function setButtonsDisabled(disabled) {
    btnRecord.disabled = disabled;
    btnPredict.disabled = disabled || !getTranscript().trim();
    btnCorrect.disabled = disabled || !getLatex().trim();
    btnRetry.disabled = disabled || !getTranscript().trim();
    btnRecordAgain.disabled = disabled;
  }

  function getTranscript() {
    // always read fresh from the input (user can edit)
    return (asrInput.value || "").trim();
  }

  function getLatex() {
    // always read fresh from the textarea
    return (latexRawEl.value || "").trim();
  }

  // Put LaTeX into textarea and render below with MathJax
  function setLatex(latex) {
    lastLatex = (latex || "").trim();
    latexRawEl.value = lastLatex;

    const toRender = ensureDollars(lastLatex);
    latexRenderBox.innerHTML = toRender || "";
    hideError();

    if (toRender && window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetPromise([latexRenderBox]).catch(() => {
        showError(
          "We couldn’t render this LaTeX. Please press Retry or Record Again."
        );
      });
    }
  }

  function ensureDollars(s) {
    if (!s) return "";
    // If already wrapped with $...$ or $$...$$ leave as-is; else wrap with $$...$$
    if (/^\$[\s\S]*\$$/.test(s)) return s;
    return `$$${s}$$`;
  }

  function toast(msg) {
    // quick feedback (replace with fancier toast if you like)
    console.log("[toast]", msg);
  }

  // =========================
  // Media recording
  // =========================
  async function startRecording() {
    chunks = [];
    lastAudioBlob = null;
    wasRetried = false; // reset for a new recording

    // Attempt preferred MIME
    let mime = "audio/webm;codecs=opus";
    if (!MediaRecorder.isTypeSupported(mime)) {
      mime = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
    }

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(
        mediaStream,
        mime ? { mimeType: mime } : undefined
      );

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        mediaStream.getTracks().forEach((t) => t.stop());
        mediaStream = null;

        lastAudioBlob = new Blob(chunks, { type: mime || "audio/webm" });
        await uploadAndTranscribe(lastAudioBlob, mime);

        setState("idle");
        setButtonsDisabled(false);
      };

      mediaRecorder.start();
      recording = true;
      btnRecord.setAttribute("aria-pressed", "true");
      btnRecord.querySelector(".btn-label").textContent = "Stop Recording";
      setState("listening");
      setButtonsDisabled(true); // lock other buttons while recording
      btnRecord.disabled = false; // keep record button enabled to stop
    } catch (err) {
      console.error(err);
      showError("Microphone access failed. Please allow mic permissions and try again.");
      setState("idle");
      setButtonsDisabled(false);
    }
  }

  function stopRecording() {
    if (mediaRecorder && recording) {
      mediaRecorder.stop();
      recording = false;
      btnRecord.setAttribute("aria-pressed", "false");
      btnRecord.querySelector(".btn-label").textContent = "Start Recording";
      setState("processing");
    }
  }

  async function uploadAndTranscribe(blob, mime) {
    try {
      const fd = new FormData();
      const ext = mime && mime.includes("webm") ? "webm" : "wav";
      fd.append("audio", blob, `recording.${ext}`);
      fd.append("ext", ext);

      const res = await fetch(`${API_BASE}/transcribe`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) {
        const j = await safeJson(res);
        throw new Error(j?.detail || `Transcribe failed (${res.status})`);
      }
      const data = await res.json();
      lastAudioPath = data.audio_path || null;
      setTranscript(data.transcript || "");
    } catch (e) {
      console.error(e);
      showError(
        "We couldn’t transcribe the audio. Please try again or check your microphone."
      );
    }
  }

  // =========================
  // API helpers
  // =========================
  async function callToLatex(text) {
    const res = await fetch(`${API_BASE}/to_latex`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      const j = await safeJson(res);
      throw new Error(j?.detail || `to_latex failed (${res.status})`);
    }
    return res.json();
  }

  async function callCorpusSearch(q, limit = 5) {
    const res = await fetch(
      `${API_BASE}/corpus/search?` +
        new URLSearchParams({ q, limit: String(limit) })
    );
    if (!res.ok) {
      const j = await safeJson(res);
      throw new Error(j?.detail || `search failed (${res.status})`);
    }
    return res.json();
  }

  async function postFeedback(payload) {
    const res = await fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const j = await safeJson(res);
      throw new Error(j?.detail || `feedback failed (${res.status})`);
    }
    return res.json();
  }

  async function safeJson(res) {
    try {
      return await res.json();
    } catch {
      return null;
    }
  }

  // =========================
  // Button handlers
  // =========================
  async function onRecordClick() {
    hideError();
    if (!recording) {
      await startRecording();
    } else {
      stopRecording();
    }
  }

  async function onGoClick() {
    hideError();
    const current = getTranscript();
    if (!current) return;

    try {
      setState("processing");
      setButtonsDisabled(true);

      const data = await callToLatex(current);
      const cleaned = data.cleaned_latex || data.raw_latex || "";
      wasRetried = false; // fresh prediction
      lastCorpusId = null;

      setLatex(cleaned);
      setState("success");
      setButtonsDisabled(false);
      btnCorrect.disabled = !getLatex();
    } catch (e) {
      console.error(e);
      showError(
        "We couldn’t produce a valid LaTeX output. Please press Retry or Record Again."
      );
      setButtonsDisabled(false);
    }
  }

  async function onCorrectClick() {
    // allow manual edits to both fields
    lastTranscript = getTranscript();
    lastLatex = getLatex();
    if (!lastLatex && !lastTranscript) return;

    try {
      await postFeedback({
        transcript_text: lastTranscript || null,
        generated_latex: lastLatex || null,
        correct: true,
        retried: wasRetried,
        record_again: false,
        audio_path: lastAudioPath,
        visual_path: null,
        corpus_id: lastCorpusId,
      });
      toast("Saved. Thanks!");

      // Reset just like “Record Again”
      onRecordAgainClick();
    } catch (e) {
      console.error(e);
      showError("Could not save feedback. Please try again.");
    }
  }

  async function onRetryClick() {
    hideError();
    const current = getTranscript();
    if (!current) return;

    try {
      setState("processing");
      setButtonsDisabled(true);

      const hits = await callCorpusSearch(current, 5);
      if (!hits || hits.length === 0) {
        showError(
          "Sorry! Not able to fetch the request, please try to record again or try some other equation."
        );
        setButtonsDisabled(false);
        setState("idle");
        return;
      }

      // Take top hit for now (could add a chooser later)
      const best = hits[0];
      lastCorpusId = best.corpus_id || null;
      wasRetried = true;

      setLatex(best.latex || "");
      setButtonsDisabled(false);
      setState("success");

      // Log retry suggestion used (non-blocking)
      try {
        await postFeedback({
          transcript_text: current || null,
          generated_latex: getLatex() || null,
          correct: null,
          retried: true,
          record_again: false,
          audio_path: lastAudioPath,
          visual_path: null,
          corpus_id: lastCorpusId,
        });
      } catch (_) {}
    } catch (e) {
      console.error(e);
      showError(
        "Sorry! Not able to fetch the request, please try to record again or try some other equation."
      );
      setButtonsDisabled(false);
      setState("idle");
    }
  }

  async function onRecordAgainClick() {
    // optional: log the reset action
    try {
      const t = getTranscript();
      const l = getLatex();
      if (t || l) {
        await postFeedback({
          transcript_text: t || null,
          generated_latex: l || null,
          correct: null,
          retried: wasRetried,
          record_again: true,
          audio_path: lastAudioPath,
          visual_path: null,
          corpus_id: lastCorpusId,
        });
      }
    } catch (_) {}

    // reset UI
    setTranscript("");
    setLatex("");
    lastAudioPath = null;
    lastCorpusId = null;
    wasRetried = false;
    setState("idle");
    setButtonsDisabled(false);
  }

  // =========================
  // Live input handlers (editable fields)
  // =========================
  function onTranscriptInput(e) {
    setTranscript(e.target.value || "");
  }

  function onLatexInput(e) {
    // re-render as the user types
    setLatex(e.target.value || "");
  }

  // Enter in transcript input triggers GO
  function onTranscriptKeydown(e) {
    if (e.key === "Enter" && !btnPredict.disabled) {
      onGoClick();
    }
  }

  // =========================
  // Init
  // =========================
  function init() {
    setState("idle");
    setTranscript("");
    setLatex("");
    hideError();

    btnRecord.addEventListener("click", onRecordClick);
    btnPredict.addEventListener("click", onGoClick);
    btnCorrect.addEventListener("click", onCorrectClick);
    btnRetry.addEventListener("click", onRetryClick);
    btnRecordAgain.addEventListener("click", onRecordAgainClick);

    asrInput.addEventListener("input", onTranscriptInput);
    asrInput.addEventListener("keydown", onTranscriptKeydown);
    latexRawEl.addEventListener("input", onLatexInput);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
