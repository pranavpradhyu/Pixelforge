(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const stage = $("stage");
  const reveal = $("reveal");
  const handle = $("reveal-handle");
  const beforeEl = $("reveal-before");
  const afterEl = $("reveal-after");

  /* ---------- reveal slider ---------- */
  function setPct(p) {
    p = Math.max(0, Math.min(100, p));
    reveal.style.setProperty("--p", p);
    handle.setAttribute("aria-valuenow", Math.round(p));
  }
  function pctFromEvent(e) {
    const rect = reveal.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    return (x / rect.width) * 100;
  }
  let dragging = false;
  const start = (e) => { dragging = true; setPct(pctFromEvent(e)); e.preventDefault(); };
  const move = (e) => { if (dragging) setPct(pctFromEvent(e)); };
  const end = () => { dragging = false; };
  reveal.addEventListener("mousedown", start);
  reveal.addEventListener("touchstart", start, { passive: false });
  window.addEventListener("mousemove", move);
  window.addEventListener("touchmove", move, { passive: false });
  window.addEventListener("mouseup", end);
  window.addEventListener("touchend", end);
  handle.addEventListener("keydown", (e) => {
    const cur = parseFloat(getComputedStyle(reveal).getPropertyValue("--p")) || 55;
    if (e.key === "ArrowLeft") { setPct(cur - 3); e.preventDefault(); }
    if (e.key === "ArrowRight") { setPct(cur + 3); e.preventDefault(); }
  });

  // one orchestrated intro sweep, then the user is in control
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduce) {
    setPct(8);
    requestAnimationFrame(() => {
      reveal.querySelectorAll(".reveal-before,.reveal-handle")
        .forEach((el) => (el.style.transition = "clip-path .9s ease, left .9s ease"));
      setTimeout(() => { setPct(58);
        setTimeout(() => reveal.querySelectorAll(".reveal-before,.reveal-handle")
          .forEach((el) => (el.style.transition = "")), 950); }, 250);
    });
  }

  /* ---------- state switching ---------- */
  function show(state) {
    stage.dataset.state = state;
    $("dropzone").hidden = state !== "idle";
    $("intents").hidden = state !== "idle";
    $("working").hidden = state !== "working";
    $("result").hidden = state !== "result";
    $("error").hidden = true;
  }
  function fail(msg) {
    const e = $("error");
    e.textContent = msg;
    e.hidden = false;
  }

  /* ---------- file intake ---------- */
  const fileInput = $("file");
  const dz = $("dropzone");
  $("browse").addEventListener("click", () => fileInput.click());
  dz.addEventListener("click", (e) => { if (e.target.id !== "browse") fileInput.click(); });
  ["dragenter", "dragover"].forEach((t) =>
    dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) =>
    dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) startJob(f);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) startJob(fileInput.files[0]);
  });

  function currentIntent() {
    const r = document.querySelector('input[name="intent"]:checked');
    return r ? r.value : "auto";
  }

  /* ---------- job lifecycle ---------- */
  let poll = null;

  async function startJob(file) {
    if (!file.type.startsWith("image/")) return fail("That's not an image file.");
    // show the user's own photo immediately as the 'before'
    const localUrl = URL.createObjectURL(file);
    reveal.dataset.demo = "0";
    beforeEl.style.setProperty("--before", `url("${localUrl}")`);
    beforeEl.style.filter = "none";

    show("working");
    setBar(4, "Uploading…");

    const body = new FormData();
    body.append("image", file);
    body.append("intent", currentIntent());

    let res;
    try {
      res = await fetch("/api/enhance", { method: "POST", body });
    } catch {
      show("idle"); return fail("Network error. Check your connection and retry.");
    }
    if (res.status === 202) {
      const { job_id } = await res.json();
      watch(job_id);
    } else {
      const data = await res.json().catch(() => ({}));
      show("idle"); fail(data.error || "Something went wrong. Try another photo.");
    }
  }

  function setBar(pct, msg) {
    $("bar-fill").style.width = pct + "%";
    if (msg) $("working-msg").textContent = msg;
  }

  function watch(jobId) {
    poll = setInterval(async () => {
      let s;
      try { s = await (await fetch(`/api/status/${jobId}`)).json(); }
      catch { return; }
      setBar(s.progress || 0, s.message);
      if (s.status === "done") {
        clearInterval(poll);
        finish(s);
      } else if (s.status === "error") {
        clearInterval(poll);
        show("idle");
        fail(s.message || "We couldn't process that image.");
      }
    }, 700);
  }

  function finish(s) {
    afterEl.style.setProperty("--after", `url("${s.output_url}")`);
    beforeEl.style.setProperty("--before", `url("${s.input_url}")`);
    const name = s.output_url.split("/").pop();
    $("download").href = `/download/${name}`;

    const tags = (s.analysis && s.analysis.detected) || [];
    $("found").innerHTML = tags.length && tags[0] !== "looks clean"
      ? `We spotted <b>${tags.join("</b>, <b>")}</b> and repaired what we could.`
      : `Your photo already looked clean — we made a light pass to be sure.`;

    show("result");
    if (!reduce) { setPct(4); requestAnimationFrame(() => {
      reveal.querySelectorAll(".reveal-before,.reveal-handle")
        .forEach((el) => (el.style.transition = "clip-path 1s ease, left 1s ease"));
      setTimeout(() => setPct(60), 120);
    }); } else { setPct(55); }
  }

  $("again").addEventListener("click", () => {
    fileInput.value = "";
    reveal.dataset.demo = "1";
    beforeEl.style.removeProperty("--before");
    afterEl.style.removeProperty("--after");
    beforeEl.style.filter = "";
    setPct(55);
    show("idle");
  });
  $("cancel").addEventListener("click", () => {
    if (poll) clearInterval(poll);
    show("idle");
  });

  show("idle");
})();
