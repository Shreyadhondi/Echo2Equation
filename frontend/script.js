// frontend/script.js
(function () {
  const API_BASE = "http://localhost:8000";
  const REQUEST_TIMEOUT_MS = 45000;
  const AUTO_CLEAR_AFTER_NO_MS = 2500;

  const body = document.body;

  const btnRecord = document.getElementById("btn-record");
  const btnPredict = document.getElementById("btn-predict");
  const btnCopyLatex = document.getElementById("btn-copy-latex");

  const btnCorrect = document.getElementById("btn-correct");
  const btnIncorrect = document.getElementById("btn-incorrect");
  const btnClearMain = document.getElementById("btn-clear-main");

  const btnKnowYes = document.getElementById("btn-know-yes");
  const btnKnowNo = document.getElementById("btn-know-no");

  const btnPreviewCorrection = document.getElementById("btn-preview-correction");
  const btnSubmitCorrection = document.getElementById("btn-submit-correction");
  const btnClearCorrection = document.getElementById("btn-clear-correction");
  const btnCopyCorrection = document.getElementById("btn-copy-correction");

  const asrInput = document.getElementById("asr-input");
  const latexRawEl = document.getElementById("latex-raw");
  const latexRenderBox = document.getElementById("latex-render");

  const feedbackSection = document.getElementById("feedback-section");
  const incorrectQuestionPanel = document.getElementById("incorrect-question-panel");
  const correctionPanel = document.getElementById("correction-panel");

  const correctionInput = document.getElementById("correction-input");
  const correctionRender = document.getElementById("correction-render");

  const statusMessage = document.getElementById("status-message");
  const errorBox = document.getElementById("error-box");
  const errorText = document.getElementById("error-text");

  const exampleButtons = document.querySelectorAll(".example-chip");

  let mediaStream = null;
  let mediaRecorder = null;
  let chunks = [];
  let recording = false;

  let lastAudioPath = null;
  let feedbackSaved = false;
  let autoClearTimer = null;

  function setState(state) {
    body.dataset.state = state;
  }

  function setStatus(message) {
    statusMessage.textContent = message || "Ready.";
  }

  function showError(message) {
    errorText.textContent = message || "Something went wrong. Please try again.";
    errorBox.classList.remove("hidden");
    setState("error");
    setStatus("Error. Please check the message below.");
  }

  function hideError() {
    errorBox.classList.add("hidden");
  }

  function clearAutoClearTimer() {
    if (autoClearTimer) {
      clearTimeout(autoClearTimer);
      autoClearTimer = null;
    }
  }

  function getTranscript() {
    return (asrInput.value || "").trim();
  }

  function getGeneratedLatex() {
    return (latexRawEl.value || "").trim();
  }

  function getCorrectedLatex() {
    return (correctionInput.value || "").trim();
  }

  function ensureDollars(text) {
    const value = (text || "").trim();

    if (!value) {
      return "";
    }

    if (/^\${1,2}[\s\S]*\${1,2}$/.test(value)) {
      return value;
    }

    return `$$${value}$$`;
  }

  function renderLatex(targetEl, latex, emptyMessage) {
    const value = ensureDollars(latex);

    if (!value) {
      targetEl.innerHTML = `<span class="muted">${emptyMessage}</span>`;
      return;
    }

    targetEl.innerHTML = value;

    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetPromise([targetEl]).catch(() => {
        showError("The LaTeX could not be rendered. Please check the syntax.");
      });
    }
  }

  async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      return await fetch(url, {
        ...options,
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }
  }

  async function safeJson(response) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }

  function isProbablyRandomText(text) {
    const value = text.trim();

    if (value.length < 2) {
      return true;
    }

    const letters = value.match(/[a-zA-Z]/g) || [];
    return letters.length === 0;
  }

  function updateButtons() {
    const hasTranscript = Boolean(getTranscript());
    const hasGeneratedLatex = Boolean(getGeneratedLatex());
    const hasCorrectedLatex = Boolean(getCorrectedLatex());
    const isBusy = body.dataset.state === "processing";

    btnPredict.disabled = isBusy || !hasTranscript;
    btnCopyLatex.disabled = isBusy || !hasGeneratedLatex;

    btnCorrect.disabled = isBusy || !hasGeneratedLatex || feedbackSaved;
    btnIncorrect.disabled = isBusy || !hasGeneratedLatex || feedbackSaved;

    btnKnowYes.disabled = isBusy || feedbackSaved;
    btnKnowNo.disabled = isBusy || feedbackSaved;

    btnPreviewCorrection.disabled = isBusy || feedbackSaved;
    btnSubmitCorrection.disabled = isBusy || feedbackSaved;
    btnCopyCorrection.disabled = isBusy || !hasCorrectedLatex;

    // Clear must always be available so the user is never trapped.
    btnClearMain.disabled = false;
    btnClearCorrection.disabled = false;
  }

  function markFeedbackSaved(message) {
    feedbackSaved = true;

    btnCorrect.disabled = true;
    btnIncorrect.disabled = true;
    btnKnowYes.disabled = true;
    btnKnowNo.disabled = true;
    btnPreviewCorrection.disabled = true;
    btnSubmitCorrection.disabled = true;

    // Keep Clear and Copy usable after saving.
    btnClearMain.disabled = false;
    btnClearCorrection.disabled = false;
    btnCopyLatex.disabled = !getGeneratedLatex();
    btnCopyCorrection.disabled = !getCorrectedLatex();

    setStatus(message);
  }

  function resetFeedbackState() {
    feedbackSaved = false;
    updateButtons();
  }

  function setTranscript(text) {
    asrInput.value = text || "";
    updateButtons();
  }

  function setGeneratedLatex(latex) {
    const value = (latex || "").trim();
    latexRawEl.value = value;

    renderLatex(
      latexRenderBox,
      value,
      "Your rendered equation will appear here."
    );

    if (value) {
      feedbackSection.classList.remove("hidden");
    } else {
      feedbackSection.classList.add("hidden");
      incorrectQuestionPanel.classList.add("hidden");
      correctionPanel.classList.add("hidden");
    }

    updateButtons();
  }

  async function callToLatex(text) {
    const response = await fetchWithTimeout(`${API_BASE}/to_latex`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const data = await safeJson(response);
      throw new Error(data?.detail || `Conversion failed with status ${response.status}`);
    }

    return response.json();
  }

  async function postFeedback(payload) {
    const response = await fetchWithTimeout(`${API_BASE}/feedback`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const data = await safeJson(response);
      throw new Error(data?.detail || `Feedback failed with status ${response.status}`);
    }

    return response.json();
  }

  async function uploadAndTranscribe(blob, mime) {
    try {
      const formData = new FormData();
      const ext = mime && mime.includes("webm") ? "webm" : "wav";

      formData.append("audio", blob, `recording.${ext}`);
      formData.append("ext", ext);

      const response = await fetchWithTimeout(`${API_BASE}/transcribe`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const data = await safeJson(response);
        throw new Error(data?.detail || `Transcription failed with status ${response.status}`);
      }

      const data = await response.json();
      lastAudioPath = data.audio_path || null;

      setTranscript(data.transcript || "");

      if (data.transcript) {
        setStatus("Transcription ready. Press Convert.");
      } else {
        setStatus("No speech detected. Please record again or type the expression.");
      }
    } catch (error) {
      console.error(error);

      if (error.name === "AbortError") {
        showError("Transcription is taking too long. Please try again with a shorter recording.");
      } else {
        showError("We could not transcribe the audio. Please try again or type the expression.");
      }
    }
  }

  async function startRecording() {
    chunks = [];
    hideError();
    resetFeedbackState();
    clearAutoClearTimer();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError("Microphone recording is not supported in this browser. Please type the expression instead.");
      return;
    }

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

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        if (mediaStream) {
          mediaStream.getTracks().forEach((track) => track.stop());
          mediaStream = null;
        }

        const audioBlob = new Blob(chunks, { type: mime || "audio/webm" });

        if (audioBlob.size === 0) {
          showError("No audio was recorded. Please try again.");
          recording = false;
          setState("idle");
          updateButtons();
          return;
        }

        await uploadAndTranscribe(audioBlob, mime);

        recording = false;
        btnRecord.setAttribute("aria-pressed", "false");
        btnRecord.querySelector(".btn-label").textContent = "Start Recording";

        setState("idle");
        updateButtons();
      };

      mediaRecorder.start();
      recording = true;

      btnRecord.setAttribute("aria-pressed", "true");
      btnRecord.querySelector(".btn-label").textContent = "Stop Recording";

      setState("listening");
      setStatus("Recording... Press Stop Recording when done.");
      updateButtons();
    } catch (error) {
      console.error(error);
      showError("Microphone access failed. Please allow microphone permission and try again.");
      recording = false;
      setState("idle");
      updateButtons();
    }
  }

  function stopRecording() {
    if (!mediaRecorder || !recording) {
      return;
    }

    setState("processing");
    setStatus("Transcribing audio...");
    updateButtons();
    mediaRecorder.stop();
  }

  async function onRecordClick() {
    hideError();

    if (!recording) {
      await startRecording();
    } else {
      stopRecording();
    }
  }

  async function onConvertClick() {
    hideError();
    resetFeedbackState();
    clearAutoClearTimer();

    const transcript = getTranscript();

    if (!transcript) {
      showError("Please type or record a math expression first.");
      return;
    }

    if (isProbablyRandomText(transcript)) {
      showError("Please enter a valid spoken math expression, for example: x square plus y square equals z square.");
      return;
    }

    try {
      setState("processing");
      setStatus("Converting spoken math to LaTeX...");
      updateButtons();

      const data = await callToLatex(transcript);
      const latex = data.cleaned_latex || data.raw_latex || "";

      if (!latex.trim()) {
        showError("The model returned an empty LaTeX result. Please try a clearer expression.");
        return;
      }

      setGeneratedLatex(latex);

      incorrectQuestionPanel.classList.add("hidden");
      correctionPanel.classList.add("hidden");
      correctionInput.value = "";
      renderLatex(correctionRender, "", "Corrected preview will appear here.");

      setState("success");
      setStatus("Conversion complete. Please tell us if the LaTeX is correct.");
      updateButtons();
    } catch (error) {
      console.error(error);

      if (error.name === "AbortError") {
        showError("The server is taking too long to respond. Please try again.");
      } else {
        showError("Server failure or connection issue. Please check if the backend is running and try again.");
      }

      setState("idle");
      updateButtons();
    }
  }

  async function onCorrectClick() {
    hideError();

    if (feedbackSaved) {
      setStatus("Feedback is already saved. Press Clear to start again.");
      return;
    }

    const transcript = getTranscript();
    const generatedLatex = getGeneratedLatex();

    if (!transcript || !generatedLatex) {
      showError("There is no generated LaTeX to save.");
      return;
    }

    try {
      setState("processing");
      setStatus("Saving accepted LaTeX...");
      updateButtons();

      await postFeedback({
        transcript_text: transcript,
        generated_latex: generatedLatex,
        corrected_latex: null,
        correct: true,
        retried: false,
        record_again: false,
        audio_path: lastAudioPath,
        visual_path: null,
        corpus_id: null,
      });

      setState("success");
      incorrectQuestionPanel.classList.add("hidden");
      correctionPanel.classList.add("hidden");
      markFeedbackSaved("Thank you. Your feedback was saved successfully. Press Clear to start again.");
    } catch (error) {
      console.error(error);

      if (error.name === "AbortError") {
        showError("Saving feedback took too long. Please try again.");
      } else {
        showError("Could not save feedback. Please check the server and try again.");
      }

      setState("success");
      updateButtons();
    }
  }

  function onIncorrectClick() {
    hideError();

    if (feedbackSaved) {
      setStatus("Feedback is already saved. Press Clear to start again.");
      return;
    }

    if (!getGeneratedLatex()) {
      showError("There is no generated LaTeX to mark as incorrect.");
      return;
    }

    incorrectQuestionPanel.classList.remove("hidden");
    correctionPanel.classList.add("hidden");

    setStatus("The output was marked incorrect. Please tell us if you know the correct LaTeX.");
    updateButtons();
  }

  function onKnowYesClick() {
    hideError();

    if (feedbackSaved) {
      setStatus("Feedback is already saved. Press Clear to start again.");
      return;
    }

    correctionPanel.classList.remove("hidden");
    correctionInput.value = "";
    renderLatex(correctionRender, "", "Corrected preview will appear here.");

    setStatus("Please type the correct LaTeX, preview it, then press OK.");
    correctionInput.focus();
    updateButtons();
  }

  async function onKnowNoClick() {
    hideError();

    if (feedbackSaved) {
      setStatus("Feedback is already saved. Press Clear to start again.");
      return;
    }

    const transcript = getTranscript();
    const generatedLatex = getGeneratedLatex();

    if (!transcript || !generatedLatex) {
      showError("Missing input or generated LaTeX.");
      return;
    }

    try {
      setState("processing");
      setStatus("Saving incorrect feedback...");
      updateButtons();

      await postFeedback({
        transcript_text: transcript,
        generated_latex: generatedLatex,
        corrected_latex: null,
        correct: false,
        retried: false,
        record_again: false,
        audio_path: lastAudioPath,
        visual_path: null,
        corpus_id: null,
      });

      setState("success");
      markFeedbackSaved("Sorry, we could not solve it this time. Your feedback has been saved and will help us improve Echo2Equation.");

      autoClearTimer = setTimeout(() => {
        clearAll();
      }, AUTO_CLEAR_AFTER_NO_MS);
    } catch (error) {
      console.error(error);
      showError("Could not save incorrect feedback. Please check the server and try again.");
      setState("success");
      updateButtons();
    }
  }

  function onPreviewCorrectionClick() {
    hideError();

    if (feedbackSaved) {
      setStatus("Correction is already saved. Press Clear to start again.");
      return;
    }

    const correctedLatex = getCorrectedLatex();

    if (!correctedLatex) {
      showError("Please enter corrected LaTeX before previewing.");
      return;
    }

    renderLatex(correctionRender, correctedLatex, "Corrected preview will appear here.");
    setStatus("Preview updated. Press OK if this corrected LaTeX is right.");
    updateButtons();
  }

  async function onSubmitCorrectionClick() {
    hideError();

    if (feedbackSaved) {
      setStatus("Correction is already saved. Press Clear to start again.");
      return;
    }

    const transcript = getTranscript();
    const generatedLatex = getGeneratedLatex();
    const correctedLatex = getCorrectedLatex();

    if (!transcript || !generatedLatex) {
      showError("Missing original input or generated LaTeX.");
      return;
    }

    if (!correctedLatex) {
      showError("Please enter the corrected LaTeX before pressing OK.");
      return;
    }

    try {
      setState("processing");
      setStatus("Saving corrected LaTeX...");
      updateButtons();

      await postFeedback({
        transcript_text: transcript,
        generated_latex: generatedLatex,
        corrected_latex: correctedLatex,
        correct: false,
        retried: false,
        record_again: false,
        audio_path: lastAudioPath,
        visual_path: null,
        corpus_id: null,
      });

      renderLatex(correctionRender, correctedLatex, "Corrected preview will appear here.");

      setState("success");
      markFeedbackSaved("Thank you. Your correction was saved and will help improve Echo2Equation. Press Clear to start again.");
    } catch (error) {
      console.error(error);

      if (error.name === "AbortError") {
        showError("Saving correction took too long. Please try again.");
      } else {
        showError("Could not save corrected feedback. Please check the server and try again.");
      }

      setState("success");
      updateButtons();
    }
  }

  async function copyText(text, successMessage, emptyMessage) {
    if (!text) {
      showError(emptyMessage);
      return;
    }

    try {
      await navigator.clipboard.writeText(text);
      setStatus(successMessage);
    } catch (error) {
      console.error(error);
      showError("Could not copy automatically. Please select and copy manually.");
    }
  }

  function onCopyLatexClick() {
    copyText(
      getGeneratedLatex(),
      "Generated LaTeX copied to clipboard.",
      "There is no generated LaTeX to copy."
    );
  }

  function onCopyCorrectionClick() {
    copyText(
      getCorrectedLatex(),
      "Corrected LaTeX copied to clipboard.",
      "There is no corrected LaTeX to copy."
    );
  }

  function clearAll() {
    clearAutoClearTimer();
    hideError();

    setTranscript("");
    setGeneratedLatex("");

    correctionInput.value = "";
    renderLatex(correctionRender, "", "Corrected preview will appear here.");

    feedbackSection.classList.add("hidden");
    incorrectQuestionPanel.classList.add("hidden");
    correctionPanel.classList.add("hidden");

    lastAudioPath = null;
    feedbackSaved = false;

    setState("idle");
    setStatus("Ready.");
    updateButtons();
  }

  function onExampleClick(event) {
    clearAll();

    const example = event.currentTarget.dataset.example || "";
    setTranscript(example);
    setStatus("Example loaded. Press Convert.");
    asrInput.focus();
  }

  function onTranscriptInput(event) {
    setTranscript(event.target.value || "");
    feedbackSaved = false;
    clearAutoClearTimer();

    if (!getTranscript()) {
      clearAll();
    } else {
      updateButtons();
    }
  }

  function onLatexInput(event) {
    setGeneratedLatex(event.target.value || "");
    feedbackSaved = false;
    clearAutoClearTimer();

    if (!getGeneratedLatex()) {
      incorrectQuestionPanel.classList.add("hidden");
      correctionPanel.classList.add("hidden");
    }

    updateButtons();
  }

  function onCorrectionInput() {
    updateButtons();
  }

  function onTranscriptKeydown(event) {
    if (event.key === "Enter" && !btnPredict.disabled) {
      onConvertClick();
    }
  }

  function init() {
    setState("idle");
    setStatus("Ready.");

    setTranscript("");
    setGeneratedLatex("");
    correctionInput.value = "";

    feedbackSection.classList.add("hidden");
    incorrectQuestionPanel.classList.add("hidden");
    correctionPanel.classList.add("hidden");
    hideError();

    btnRecord.addEventListener("click", onRecordClick);
    btnPredict.addEventListener("click", onConvertClick);
    btnCopyLatex.addEventListener("click", onCopyLatexClick);

    btnCorrect.addEventListener("click", onCorrectClick);
    btnIncorrect.addEventListener("click", onIncorrectClick);
    btnClearMain.addEventListener("click", clearAll);

    btnKnowYes.addEventListener("click", onKnowYesClick);
    btnKnowNo.addEventListener("click", onKnowNoClick);

    btnPreviewCorrection.addEventListener("click", onPreviewCorrectionClick);
    btnSubmitCorrection.addEventListener("click", onSubmitCorrectionClick);
    btnClearCorrection.addEventListener("click", clearAll);
    btnCopyCorrection.addEventListener("click", onCopyCorrectionClick);

    exampleButtons.forEach((button) => {
      button.addEventListener("click", onExampleClick);
    });

    asrInput.addEventListener("input", onTranscriptInput);
    asrInput.addEventListener("keydown", onTranscriptKeydown);
    latexRawEl.addEventListener("input", onLatexInput);
    correctionInput.addEventListener("input", onCorrectionInput);

    updateButtons();
  }

  document.addEventListener("DOMContentLoaded", init);
})();