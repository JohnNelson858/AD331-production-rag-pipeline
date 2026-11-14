"use strict";
const $ = (id) => document.getElementById(id);
const form = $("query"), input = $("question"), mode = $("mode");
const conversation = $("conversation"), submit = $("submit");
let history = [], examples = {}, liveAvailable = false, accessRequired = false, busy = false;
const status = (msg, error = false) => {
  $("status").textContent = msg;
  $("status").classList.toggle("error", error);
};
const updateTurns = () => {
  $("turns").textContent = conversation.checked ? `${history.length} saved turns` : "";
};
async function configure() {
  try {
    const reply = await fetch("/api/config");
    if (!reply.ok) throw Error();
    const config = await reply.json();
    examples = config.examples;
    liveAvailable = config.live_available;
    accessRequired = config.live_access_required;
    mode.options[1].disabled = !liveAvailable;
    describeMode();
  } catch {
    status("Could not load the workbench settings. Refresh and try again.", true);
  }
}
function describeMode() {
  $("access-box").hidden = !(mode.value === "live" && accessRequired);
  $("mode-help").textContent = mode.value === "live"
    ? "Live requests use the configured providers and may incur API charges."
    : "Offline preview uses sample traces and makes no provider calls."
      + (liveAvailable ? "" : " Live mode needs server keys.");
}
function clearResult() {
  $("answer").textContent = "Run a question to see its answer and sources.";
  $("web").hidden = true;
  $("web-sources").replaceChildren();
  for (const id of ["stage1", "stage2"]) $(id).replaceChildren();
  $("stage1-label").textContent = "Stage 1 · Bi-encoder candidates";
  $("stage2-label").textContent = "Stage 2 · Cross-encoder rankings";
  $("route").textContent = "AWAITING QUERY";
  $("route").className = "badge";
  for (const id of ["transformed", "rewrite", "score"]) $(id).textContent = "—";
  $("confidence").value = 0;
  $("trace-note").textContent = "The reranker score determines the source route.";
}
mode.addEventListener("change", () => {
  history = []; updateTurns(); describeMode(); clearResult(); status("Ready.");
});
conversation.addEventListener("change", updateTurns);
$("clear").addEventListener("click", () => {
  history = []; updateTurns(); status("Conversation history cleared.");
});
document.querySelectorAll("[data-case]").forEach((button) => {
  button.addEventListener("click", () => {
    const key = button.dataset.case;
    if (!examples[key]) return;
    input.value = examples[key];
    if (key === "followup") {
      conversation.checked = true; updateTurns();
      status("Run the internal example first if you have no saved turns.");
    }
    input.focus();
  });
});
function showCitation(source) {
  $("citation-title").textContent = source.title;
  $("citation-loc").textContent = source.location;
  $("citation-body").textContent = source.content;
  const link = $("citation-link");
  link.hidden = true; link.removeAttribute("href");
  if (source.source === "web") {
    try {
      const url = new URL(source.location);
      if (["https:", "http:"].includes(url.protocol) && !url.username && !url.password) {
        link.href = url.href; link.hidden = false;
      }
    } catch { /* omit malformed links */ }
  }
  $("citation").showModal();
}
function row(target, values, source) {
  const tr = document.createElement("tr");
  for (const [index, value] of values.entries()) {
    const cell = document.createElement("td");
    if (index === 1) {
      const button = document.createElement("button");
      button.type = "button"; button.className = "cite"; button.textContent = value;
      button.addEventListener("click", () => showCitation(source));
      cell.append(button);
    } else cell.textContent = value;
    tr.append(cell);
  }
  target.append(tr);
}
function display(payload) {
  const result = payload.result;
  const isWeb = result.route_taken === "WEB_FALLBACK";
  $("route").textContent = result.route_taken;
  $("route").className = "badge " + (isWeb ? "web" : "internal");
  $("transformed").textContent = result.transformed_query;
  $("rewrite").textContent = result.rewrite_status;
  $("score").textContent = result.confidence_score.toFixed(3);
  $("confidence").value = result.confidence_score;
  $("threshold").textContent = `Gate: ${result.threshold.toFixed(2)}`;
  $("trace-note").textContent = payload.mode === "preview"
    ? "Sample trace · scores are illustrative."
    : `${payload.elapsed_ms} ms · score is a relevance signal.`;
  const evidence = [...result.stage2, ...result.web_sources];
  const answer = $("answer");
  answer.replaceChildren();
  for (const line of result.answer.split("\n")) {
    const paragraph = document.createElement("p");
    for (const piece of line.split(/(\[\d+\])/g)) {
      const citation = /^\[(\d+)\]$/.exec(piece);
      const ref = citation && result.citations[Number(citation[1]) - 1];
      const source = ref && evidence.find((s) => s.chunk_id === ref || s.location === ref);
      if (source) {
        const button = document.createElement("button");
        button.type = "button"; button.className = "cite"; button.textContent = piece;
        button.setAttribute("aria-label", `Open citation ${citation[1]}`);
        button.addEventListener("click", () => showCitation(source));
        paragraph.append(button);
      } else paragraph.append(document.createTextNode(piece));
    }
    answer.append(paragraph);
  }
  $("stage1-label").textContent = `Stage 1 · Bi-encoder candidates (${result.stage1.length})`;
  $("stage2-label").textContent = `Stage 2 · Cross-encoder rankings (${result.stage2.length})`;
  for (const item of result.stage1)
    row($("stage1"), [item.dense_rank, item.title, item.distance.toFixed(3)], item);
  for (const item of result.stage2) {
    const shift = item.dense_rank - item.rerank_rank;
    row($("stage2"), [item.rerank_rank, item.title, item.score.toFixed(3),
      item.dense_rank, shift > 0 ? `↑ ${shift}` : shift < 0 ? `↓ ${-shift}` : "—"], item);
  }
  $("web").hidden = !isWeb;
  for (const item of result.web_sources) {
    const p = document.createElement("p"), button = document.createElement("button");
    button.type = "button"; button.className = "cite"; button.textContent = item.title;
    button.addEventListener("click", () => showCitation(item));
    p.append(button, document.createTextNode(" " + item.content)); $("web-sources").append(p);
  }
  if (isWeb && !result.web_sources.length) $("web-sources").textContent = "No snippets returned.";
}
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const question = input.value.trim();
  if (!question) return status("Enter a question.", true);
  const useHistory = conversation.checked, execMode = mode.value;
  const headers = {"Content-Type": "application/json"};
  if (execMode === "live" && accessRequired) {
    const code = $("access-code").value.trim();
    if (!/^[!-~]{1,128}$/.test(code)) return status("Enter the live access code.", true);
    headers["X-Live-Access"] = code;
  }
  busy = true;
  for (const item of [submit, mode, conversation, $("clear")]) item.disabled = true;
  clearResult();
  status(execMode === "live" ? "Running live pipeline. First run may load models." : "Loading sample trace…");
  try {
    const reply = await fetch("/api/query", {
      method: "POST", headers,
      body: JSON.stringify({question, mode: execMode, history: useHistory ? history : []}),
    });
    const data = await reply.json();
    if (!reply.ok) throw Error(typeof data.detail === "string" ? data.detail : "The request could not be completed.");
    display(data);
    if (!useHistory) history = [];
    history = [...history, {question: data.result.transformed_query, answer: data.result.answer.slice(0, 1500)}].slice(-3);
    updateTurns(); status(execMode === "live" ? "Live run complete." : "Offline preview complete.");
  } catch (err) {
    status(err instanceof Error ? err.message : "Could not run the pipeline.", true);
  } finally {
    busy = false;
    for (const item of [submit, mode, conversation, $("clear")]) item.disabled = false;
  }
});
configure();
