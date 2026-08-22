function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

export default {
  render({ model, el }) {
    el.innerHTML = `
      <style>
        .alignment-review { font-family: sans-serif; max-width: 100%; }
        .alignment-map { display: flex; flex-wrap: wrap; gap: 4px; margin: 8px 0; }
        .alignment-box { width: 36px; height: 30px; border: 0; border-radius: 4px; color: white; cursor: pointer; font-weight: 700; }
        .alignment-box.keep { background: #188038; } .alignment-box.flip { background: #d93025; }
        .alignment-box.playing { outline: 3px solid #1a73e8; outline-offset: 2px; }
        .alignment-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 10px 0; }
        .alignment-controls button { padding: 6px 10px; cursor: pointer; }
        .alignment-finish { margin-left: auto; background: #174ea6; border: 1px solid #174ea6; border-radius: 4px; color: white; font-weight: 700; }
        .alignment-player { width: min(720px, 100%); margin-top: 8px; }
        .alignment-status { min-height: 1.3em; color: #444; }
      </style>
      <section class="alignment-review">
        <h3>UNMIXX chunk alignment review</h3>
        <div>Green keeps the online alignment. Red swaps that chunk's online alignment.</div>
        <div class="alignment-map"></div><div class="alignment-now">Playback: stopped</div>
        <div class="alignment-controls">
          <label>Identity <select class="alignment-identity"><option value="0">1 / stem 1</option><option value="1">2 / stem 2</option></select></label>
          <label>Second <input class="alignment-second" type="range" min="0" step="0.1"><output class="alignment-second-value">0.0</output></label>
          <button class="alignment-play-selected">Play selected stem</button><button class="alignment-play-both">Play both stems</button><button class="alignment-save">Save corrected stems</button><button class="alignment-finish" title="Save corrections and replace the final singer stems">Finish alignment</button>
        </div>
        <audio class="alignment-player selected" controls></audio>
        <div class="alignment-both" hidden><audio class="alignment-player one" controls></audio><audio class="alignment-player two" controls></audio></div>
        <div class="alignment-status"></div>
      </section>`;

    const map = el.querySelector(".alignment-map");
    const now = el.querySelector(".alignment-now");
    const slider = el.querySelector(".alignment-second");
    const secondValue = el.querySelector(".alignment-second-value");
    const identity = el.querySelector(".alignment-identity");
    const selected = el.querySelector("audio.selected");
    const both = el.querySelector(".alignment-both");
    const playerOne = el.querySelector("audio.one");
    const playerTwo = el.querySelector("audio.two");
    const status = el.querySelector(".alignment-status");
    let boxes = [];

    function flips() { return [...model.get("flips")]; }
    function drawMap() {
      map.replaceChildren();
      boxes = flips().map((flip, index) => {
        const box = document.createElement("button");
        box.className = `alignment-box ${flip ? "flip" : "keep"}`;
        box.textContent = String(index + 1);
        box.title = `Chunk ${index + 1}: ${flip ? "swap online alignment" : "keep online alignment"}`;
        box.addEventListener("click", () => {
          const values = flips(); values[index] = !values[index];
          model.set("flips", values); model.save_changes(); drawMap();
          status.textContent = "Unsaved change.";
        });
        map.appendChild(box);
        return box;
      });
    }
    function chunkAt(second) {
      let index = 0;
      model.get("chunk_starts").forEach((start, candidate) => { if (second >= start) index = candidate; });
      return index;
    }
    function mark(audio) {
      const index = chunkAt(audio.currentTime);
      boxes.forEach((box, candidate) => box.classList.toggle("playing", candidate === index));
      now.textContent = `Playback: ${audio.currentTime.toFixed(1)} s · chunk ${index + 1}`;
    }
    function connect(audio) {
      audio.addEventListener("timeupdate", () => mark(audio));
      audio.addEventListener("play", () => mark(audio));
      audio.addEventListener("ended", () => { boxes.forEach(box => box.classList.remove("playing")); now.textContent = "Playback: stopped"; });
    }
    connect(selected); connect(playerOne); connect(playerTwo);

    slider.max = String(model.get("duration"));
    slider.addEventListener("input", () => { secondValue.value = Number(slider.value).toFixed(1); });
    function message(action) {
      model.send({ action, flips: flips(), second: Number(slider.value), identity: Number(identity.value) });
    }
    el.querySelector(".alignment-play-selected").addEventListener("click", () => message("play_selected"));
    el.querySelector(".alignment-play-both").addEventListener("click", () => message("play_both"));
    el.querySelector(".alignment-save").addEventListener("click", () => message("save"));
    el.querySelector(".alignment-finish").addEventListener("click", () => message("finish"));

    model.on("change:flips", drawMap);
    model.on("change:status", () => { status.textContent = model.get("status"); });
    function setAudioSource(audio, buffer, second) {
      if (audio.alignmentUrl) URL.revokeObjectURL(audio.alignmentUrl);
      audio.alignmentUrl = URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
      audio.onloadedmetadata = () => { audio.currentTime = second; audio.play(); };
      audio.src = audio.alignmentUrl;
    }
    model.on("msg:custom", (content, buffers) => {
      if (content.type !== "audio") return;
      const command = content.command;
      const second = clamp(command.second || 0, 0, model.get("duration"));
      if (command.action === "play_selected") {
        both.hidden = true; selected.hidden = false;
        setAudioSource(selected, buffers[0], second);
      } else if (command.action === "play_both") {
        selected.hidden = true; both.hidden = false;
        setAudioSource(playerOne, buffers[0], second);
        setAudioSource(playerTwo, buffers[1], second);
      }
    });
    drawMap();
    status.textContent = model.get("status");
  },
};
