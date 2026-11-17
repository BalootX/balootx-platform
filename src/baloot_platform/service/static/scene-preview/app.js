(() => {
  "use strict";
  const data = window.BALOOTX_QA;
  const byId = id => document.getElementById(id);
  const copy = key => {
    const value = data.copy?.[key];
    if (typeof value !== "string" || !value.trim()) throw new Error(`missing governed preview copy: ${key}`);
    return value;
  };
  const video = byId("video");
  const still = byId("still");
  const phone = byId("phone");
  const scene = byId("scene");
  const layers = byId("layers");
  const layerList = byId("layerList");
  const chapters = byId("chapters");
  let artifactKey = "flagship";

  const active = () => data.artifacts[artifactKey];
  const videoScene = item => item.scenes.find(value => value.kind === "video") || item.scenes[0];
  const formatTime = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

  function configureCopy() {
    document.querySelectorAll("[data-copy-key]").forEach(element => {
      element.textContent = copy(element.dataset.copyKey);
    });
    byId("brand").setAttribute("aria-label", copy("brand_aria"));
    byId("artifact").setAttribute("aria-label", copy("artifact_switcher_aria"));
    byId("viewer").setAttribute("aria-label", copy("viewer_aria"));
    byId("captionsTrack").setAttribute("label", copy("captions_label"));
    still.alt = copy("still_alt");
    byId("utility-controls").setAttribute("aria-label", copy("playback_options_aria"));
    byId("speed").setAttribute("aria-label", copy("speed_aria"));
    chapters.setAttribute("aria-label", copy("chapters_aria"));
  }

  function renderSceneOptions() {
    scene.replaceChildren();
    active().scenes.forEach(item => {
      const option = document.createElement("option");
      option.value = item.kind;
      option.textContent = item.label;
      scene.appendChild(option);
    });
    scene.value = videoScene(active()).kind;
  }

  function renderChapters() {
    chapters.replaceChildren();
    (active().chapters || []).forEach((item, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `chapter${index === 0 ? " is-active" : ""}`;
      button.dataset.start = String(item.startSeconds);
      const time = document.createElement("time");
      time.textContent = formatTime(item.startSeconds);
      const label = document.createElement("span");
      label.textContent = item.label;
      button.append(time, label);
      button.addEventListener("click", () => {
        video.currentTime = item.startSeconds;
        video.play().catch(() => undefined);
      });
      chapters.appendChild(button);
    });
  }

  function renderArtifact() {
    const item = active();
    renderSceneOptions();
    renderChapters();
    document.querySelectorAll("[data-artifact]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.artifact === artifactKey)));
    byId("formatLabel").textContent = copy(artifactKey === "flagship" ? "flagship_label" : "decision_label");
    byId("title").textContent = copy(artifactKey === "flagship" ? "flagship_title" : "decision_title");
    byId("resolution").textContent = `${item.width} × ${item.height}`;
    byId("identity").textContent = item.identity;
    layerList.replaceChildren();
    item.layers.forEach(layer => {
      const row = document.createElement("li");
      const name = document.createElement("b");
      const disclosure = document.createElement("span");
      name.textContent = layer.name;
      disclosure.textContent = layer.disclosure;
      row.append(name, disclosure);
      layerList.appendChild(row);
    });
    renderSelectedScene();
  }

  function renderSelectedScene() {
    const selected = active().scenes.find(value => value.kind === scene.value) || videoScene(active());
    const isVideo = selected.kind === "video";
    video.hidden = !isVideo;
    still.hidden = isVideo;
    if (isVideo) {
      video.src = selected.src;
      const track = byId("captionsTrack");
      if (active().captions) {
        track.src = active().captions;
        byId("captions").disabled = false;
      } else {
        track.removeAttribute("src");
        byId("captions").disabled = true;
      }
      video.load();
      video.playbackRate = Number(byId("speed").value);
      if (video.textTracks.length) {
        video.textTracks[0].mode = byId("captions").getAttribute("aria-pressed") === "true" ? "showing" : "hidden";
      }
    } else {
      still.src = selected.src;
      still.alt = selected.label;
      video.pause();
    }
  }

  function toggleCaptions() {
    if (!video.textTracks.length) return;
    const enabled = video.textTracks[0].mode !== "showing";
    video.textTracks[0].mode = enabled ? "showing" : "hidden";
    byId("captions").setAttribute("aria-pressed", String(enabled));
  }

  document.querySelectorAll("[data-artifact]").forEach(button => button.addEventListener("click", () => {
    artifactKey = button.dataset.artifact;
    renderArtifact();
  }));
  scene.addEventListener("change", renderSelectedScene);
  byId("speed").addEventListener("change", event => { video.playbackRate = Number(event.target.value); });
  byId("captions").addEventListener("click", toggleCaptions);
  byId("fullscreen").addEventListener("click", () => phone.requestFullscreen?.().catch(() => undefined));
  byId("safeArea").addEventListener("click", event => {
    const enabled = event.currentTarget.getAttribute("aria-pressed") !== "true";
    event.currentTarget.setAttribute("aria-pressed", String(enabled));
    phone.classList.toggle("show-safe", enabled);
  });
  byId("disclosure").addEventListener("click", event => {
    const enabled = event.currentTarget.getAttribute("aria-pressed") !== "true";
    event.currentTarget.setAttribute("aria-pressed", String(enabled));
    document.body.classList.toggle("disclosure-on", enabled);
    layers.hidden = !enabled;
  });
  video.addEventListener("timeupdate", () => {
    const buttons = [...chapters.querySelectorAll(".chapter")];
    let current = buttons[0];
    buttons.forEach(button => { if (Number(button.dataset.start) <= video.currentTime) current = button; });
    buttons.forEach(button => button.classList.toggle("is-active", button === current));
  });
  document.addEventListener("keydown", event => {
    const activeTag = document.activeElement?.tagName;
    if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(activeTag)) return;
    if (event.code === "Space" && activeTag !== "VIDEO") { event.preventDefault(); video.paused ? video.play().catch(() => undefined) : video.pause(); }
    if (event.key === "ArrowRight") video.currentTime = Math.min(video.duration || Infinity, video.currentTime + 5);
    if (event.key === "ArrowLeft") video.currentTime = Math.max(0, video.currentTime - 5);
    if (event.key.toLowerCase() === "f") phone.requestFullscreen?.().catch(() => undefined);
    if (event.key.toLowerCase() === "c") toggleCaptions();
  });

  configureCopy();
  renderArtifact();
})();
