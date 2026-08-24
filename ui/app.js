// VoiceClone AI Studio Frontend Logic

document.addEventListener("DOMContentLoaded", () => {
  const btnSynthesize = document.getElementById("btnSynthesize");
  const promptText = document.getElementById("promptText");
  const tempSlider = document.getElementById("tempSlider");
  const tempVal = document.getElementById("tempVal");
  const odeSlider = document.getElementById("odeSlider");
  const odeVal = document.getElementById("odeVal");
  const audioPlayer = document.getElementById("audioPlayer");
  
  const valDuration = document.getElementById("valDuration");
  const valLatency = document.getElementById("valLatency");
  const valRTF = document.getElementById("valRTF");
  const valSNR = document.getElementById("valSNR");

  const canvas = document.getElementById("waveformCanvas");
  const ctx = canvas.getContext("2d");

  // Selected Voice ID tracking
  let selectedVoiceId = "default_voice";
  const voiceList = document.getElementById("voiceList");

  // Voice Selection Handler
  voiceList.addEventListener("click", (e) => {
    const item = e.target.closest(".voice-item");
    if (!item) return;
    document.querySelectorAll(".voice-item").forEach((el) => el.classList.remove("active"));
    item.classList.add("active");
    selectedVoiceId = item.getAttribute("data-voice");
    console.log("Selected voice:", selectedVoiceId);
  });

  // Slider bindings
  tempSlider.addEventListener("input", (e) => tempVal.textContent = e.target.value);
  odeSlider.addEventListener("input", (e) => odeVal.textContent = e.target.value);

  // Resize Canvas
  function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    drawIdleWaveform();
  }
  window.addEventListener("resize", resizeCanvas);
  setTimeout(resizeCanvas, 100);

  // Animated Waveform Canvas
  let animationId = null;
  let isPlaying = false;

  function drawIdleWaveform() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(99, 102, 241, 0.4)";
    ctx.beginPath();
    const sliceWidth = canvas.width / 60;
    let x = 0;
    for (let i = 0; i < 60; i++) {
      const y = canvas.height / 2 + Math.sin(i * 0.2) * 6;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      x += sliceWidth;
    }
    ctx.stroke();
  }

  function drawPlayingWaveform(timestamp) {
    if (!isPlaying) return;
    animationId = requestAnimationFrame(drawPlayingWaveform);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const numBars = 48;
    const barWidth = canvas.width / numBars;
    const t = (timestamp || performance.now()) * 0.005;

    for (let i = 0; i < numBars; i++) {
      const heightFrac = Math.abs(Math.sin(t + i * 0.35) * Math.cos(t * 0.7 + i * 0.15)) * 0.75 + 0.1;
      const barHeight = heightFrac * canvas.height;

      const grad = ctx.createLinearGradient(0, canvas.height, 0, 0);
      grad.addColorStop(0, "#6366f1");
      grad.addColorStop(0.5, "#a855f7");
      grad.addColorStop(1, "#ec4899");

      ctx.fillStyle = grad;
      ctx.fillRect(i * barWidth, (canvas.height - barHeight) / 2, barWidth - 2, barHeight);
    }
  }

  audioPlayer.addEventListener("play", () => {
    isPlaying = true;
    if (animationId) cancelAnimationFrame(animationId);
    drawPlayingWaveform();
  });

  audioPlayer.addEventListener("pause", () => {
    isPlaying = false;
    if (animationId) cancelAnimationFrame(animationId);
    drawIdleWaveform();
  });

  audioPlayer.addEventListener("ended", () => {
    isPlaying = false;
    if (animationId) cancelAnimationFrame(animationId);
    drawIdleWaveform();
  });

  function base64ToBlob(base64, mimeType = "audio/wav") {
    const byteCharacters = atob(base64);
    const byteNumbers = new Uint8Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    return new Blob([byteNumbers], { type: mimeType });
  }

  // Voice Registration Handler
  const btnRegisterVoice = document.getElementById("btnRegisterVoice");
  const newVoiceId = document.getElementById("newVoiceId");
  const refAudioInput = document.getElementById("refAudioInput");
  const consentAudioInput = document.getElementById("consentAudioInput");
  const bypassConsentCheck = document.getElementById("bypassConsentCheck");
  const consentStatus = document.getElementById("consentStatus");

  btnRegisterVoice.addEventListener("click", async () => {
    const voiceId = newVoiceId.value.trim();
    if (!voiceId) {
      alert("Please enter a voice identifier (e.g. alam).");
      return;
    }

    if (!refAudioInput.files.length) {
      alert("Please select a Reference Audio file.");
      return;
    }

    const bypass = bypassConsentCheck.checked;
    if (!bypass && !consentAudioInput.files.length) {
      alert("Please select a Spoken Consent Audio file, or check 'Bypass Consent (Dev Mode)'.");
      return;
    }

    btnRegisterVoice.disabled = true;
    btnRegisterVoice.textContent = "Verifying & Extracting Timbre...";
    consentStatus.innerHTML = `<span style="color: #818cf8;">Processing speaker embedding...</span>`;

    const formData = new FormData();
    formData.append("voice_id", voiceId);
    formData.append("reference_file", refAudioInput.files[0]);
    formData.append("bypass_consent", bypass ? "true" : "false");
    formData.append("consent_phrase", "I hereby authorize the cloning and synthetic generation of my voice for authorized application use.");

    if (consentAudioInput.files.length) {
      formData.append("consent_file", consentAudioInput.files[0]);
    }

    try {
      const resp = await fetch("/api/register_voice", {
        method: "POST",
        body: formData
      });

      const res = await resp.json();

      if (!resp.ok) {
        throw new Error(res.detail || "Registration failed");
      }

      // Success
      consentStatus.innerHTML = `
        <span style="color: #34d399; font-weight: 600;">✔ Voice '${voiceId}' Registered Successfully!</span>
        <br><span style="color: #94a3b8; font-size: 0.75rem;">Timbre: ${res.embedding_dim}-dim embedding cached</span>
      `;

      // Add to voice list if not already present
      let existingItem = document.querySelector(`.voice-item[data-voice="${voiceId}"]`);
      if (!existingItem) {
        const initials = voiceId.substring(0, 2).toUpperCase();
        const div = document.createElement("div");
        div.className = "voice-item active";
        div.setAttribute("data-voice", voiceId);
        div.innerHTML = `
          <div class="voice-meta">
            <div class="voice-avatar">${initials}</div>
            <div>
              <div style="font-weight: 600;">${voiceId}</div>
              <div style="font-size: 0.75rem; color: var(--text-muted);">Custom Cloned Profile</div>
            </div>
          </div>
        `;
        document.querySelectorAll(".voice-item").forEach((el) => el.classList.remove("active"));
        voiceList.appendChild(div);
      } else {
        document.querySelectorAll(".voice-item").forEach((el) => el.classList.remove("active"));
        existingItem.classList.add("active");
      }

      selectedVoiceId = voiceId;

    } catch (err) {
      consentStatus.innerHTML = `
        <span style="color: #f87171; font-weight: 600;">✖ Registration Refused:</span>
        <br><span style="color: #cbd5e1; font-size: 0.78rem;">${err.message}</span>
        <br><span style="color: #94a3b8; font-size: 0.72rem;">Tip: If testing with different audio files, check 'Bypass Consent (Dev Mode)'.</span>
      `;
    } finally {
      btnRegisterVoice.disabled = false;
      btnRegisterVoice.textContent = "Verify & Register Profile";
    }
  });

  // Synthesis Execution
  btnSynthesize.addEventListener("click", async () => {
    const text = promptText.value.trim();
    if (!text) {
      alert("Please enter text to synthesize.");
      return;
    }

    btnSynthesize.disabled = true;
    btnSynthesize.innerHTML = `
      <span style="display:inline-block; animation: spin 1s linear infinite; margin-right: 8px;">⏳</span>
      Synthesizing Voice (${selectedVoiceId})...
    `;

    const emotionSelect = document.getElementById("emotionSelect");
    const emotionVal = emotionSelect ? emotionSelect.value : "conversational";

    const genderToneSelect = document.getElementById("genderToneSelect");
    const genderToneVal = genderToneSelect ? genderToneSelect.value : "auto";

    try {
      const response = await fetch("/api/clone", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          voice_id: selectedVoiceId,
          emotion_style: emotionVal,
          gender_tone: genderToneVal,
          temperature: parseFloat(tempSlider.value),
          ode_steps: parseInt(odeSlider.value),
          apply_watermark: true
        })
      });



      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Synthesis failed");
      }

      const data = await response.json();

      // Convert Base64 to Blob URL for instant audio playback
      const audioBlob = base64ToBlob(data.audio_base64, "audio/wav");
      const audioUrl = URL.createObjectURL(audioBlob);
      
      audioPlayer.src = audioUrl;
      audioPlayer.load();

      // Play audio
      const playPromise = audioPlayer.play();
      if (playPromise !== undefined) {
        playPromise.catch(() => {
          console.log("Autoplay was blocked by browser. User can press the play button.");
        });
      }

      // Update telemetry cards
      valDuration.textContent = `${data.duration_sec.toFixed(1)}s`;
      valLatency.textContent = `${data.profiling.total_time_sec < 1 ? (data.profiling.total_time_sec * 1000).toFixed(0) + 'ms' : data.profiling.total_time_sec.toFixed(2) + 's'}`;
      valRTF.textContent = `${data.profiling.rtf.toFixed(2)}x`;
      if (data.watermark_meta && data.watermark_meta.snr_db) {
        valSNR.textContent = `${data.watermark_meta.snr_db.toFixed(1)} dB`;
      }

    } catch (e) {
      alert(`Synthesis Error: ${e.message}`);
    } finally {
      btnSynthesize.disabled = false;
      btnSynthesize.innerHTML = `
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
        Synthesize Cloned Voice
      `;
    }
  });

  // Watermark Inspector
  const btnInspectWatermark = document.getElementById("btnInspectWatermark");
  const inspectAudioInput = document.getElementById("inspectAudioInput");
  const inspectResult = document.getElementById("inspectResult");

  btnInspectWatermark.addEventListener("click", async () => {
    if (!inspectAudioInput.files.length) {
      alert("Please choose an audio file to inspect.");
      return;
    }

    const formData = new FormData();
    formData.append("audio_file", inspectAudioInput.files[0]);

    inspectResult.textContent = "Scanning watermark bits...";
    try {
      const resp = await fetch("/api/detect_watermark", {
        method: "POST",
        body: formData
      });
      const res = await resp.json();
      if (res.is_ai_generated) {
        inspectResult.innerHTML = `<span style="color: #34d399; font-weight: bold;">✔ AudioSeal Watermark Detected</span><br>Confidence: ${(res.confidence_score * 100).toFixed(1)}%<br>Payload: [${res.recovered_payload.join('')}]`;
      } else {
        inspectResult.innerHTML = `<span style="color: #94a3b8;">✖ No AI Watermark Found (Confidence: ${(res.confidence_score * 100).toFixed(1)}%)</span>`;
      }
    } catch (err) {
      inspectResult.textContent = `Inspection error: ${err.message}`;
    }
  });
});
