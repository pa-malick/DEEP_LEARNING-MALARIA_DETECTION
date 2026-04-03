// ================================================================
// script.js  –  Interface Malaria Detection
// Auteur : Papa Malick NDIAYE
// ================================================================

const fileInput   = document.getElementById("file-input");
const uploadZone  = document.getElementById("upload-zone");
const previewImg  = document.getElementById("preview-img");
const placeholder = document.getElementById("upload-placeholder");
const btnPredict  = document.getElementById("btn-predict");
const resultBox   = document.getElementById("result-box");

let fichierSelectionne = null;

// ── Prévisualisation de l'image ──────────────────────────────────
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  fichierSelectionne = file;

  const reader = new FileReader();
  reader.onload = (ev) => {
    previewImg.src = ev.target.result;
    previewImg.classList.remove("hidden");
    placeholder.classList.add("hidden");
    btnPredict.disabled = false;
    resultBox.className = "result-box hidden";
  };
  reader.readAsDataURL(file);
});

// ── Drag & Drop ──────────────────────────────────────────────────
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) {
    // Simule la sélection de fichier
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    fileInput.dispatchEvent(new Event("change"));
  }
});

// ── Prédiction ───────────────────────────────────────────────────
btnPredict.addEventListener("click", async () => {
  if (!fichierSelectionne) return;

  btnPredict.textContent = "Analyse en cours…";
  btnPredict.disabled    = true;
  resultBox.className    = "result-box hidden";

  const formData = new FormData();
  formData.append("image", fichierSelectionne);

  try {
    const res  = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (data.erreur) {
      resultBox.innerHTML   = `⚠ Erreur : ${data.erreur}`;
      resultBox.className   = "result-box parasitized";
    } else {
      const estInfecte = data.classe_id === 0;
      const emoji      = estInfecte ? "🚨" : "✅";
      resultBox.innerHTML = `
        ${emoji} <strong>${data.label}</strong><br>
        <small style="font-weight:400;opacity:0.85;font-size:0.8rem">
          Confiance : ${data.probabilite} %
        </small>
      `;
      resultBox.className = estInfecte ? "result-box parasitized" : "result-box uninfected";
    }
  } catch {
    resultBox.innerHTML = "⚠ Impossible de joindre l'API.";
    resultBox.className = "result-box parasitized";
  }

  btnPredict.textContent = "Analyser ↗";
  btnPredict.disabled    = false;
});

// ── Chargement des métriques ─────────────────────────────────────
async function chargerMetriques() {
  try {
    const res  = await fetch("/metrics");
    const data = await res.json();
    if (!data.meilleur_modele) return;

    const meilleur = data.meilleur_modele;
    const m        = data.resultats[meilleur];

    document.getElementById("best-model").textContent = meilleur;
    document.getElementById("m-acc").textContent  = (m.accuracy  * 100).toFixed(1) + "%";
    document.getElementById("m-prec").textContent = (m.precision * 100).toFixed(1) + "%";
    document.getElementById("m-rec").textContent  = (m.recall    * 100).toFixed(1) + "%";
    document.getElementById("m-f1").textContent   = (m.f1_score  * 100).toFixed(1) + "%";

    const tbody = document.getElementById("table-body");
    tbody.innerHTML = "";
    for (const [nom, met] of Object.entries(data.resultats)) {
      const tr = document.createElement("tr");
      if (nom === meilleur) tr.className = "best-row";
      tr.innerHTML = `
        <td>${nom}${nom === meilleur ? " 🏆" : ""}</td>
        <td>${(met.accuracy  * 100).toFixed(2)}%</td>
        <td>${(met.precision * 100).toFixed(2)}%</td>
        <td>${(met.recall    * 100).toFixed(2)}%</td>
        <td>${(met.f1_score  * 100).toFixed(2)}%</td>
      `;
      tbody.appendChild(tr);
    }
  } catch {
    document.getElementById("best-model").textContent = "Lancez d'abord : python main.py";
  }
}

// ── Init ─────────────────────────────────────────────────────────
chargerMetriques();
