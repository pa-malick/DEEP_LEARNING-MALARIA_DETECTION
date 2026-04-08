// script.js – Interface Malaria Detection
// Auteur : Papa Malick NDIAYE | Master DSGL – UADB


// --- Fond Three.js ---

function initThreeJS() {
  const canvas   = document.getElementById("bg-canvas");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.z = 30;

  const NB_PARTICULES = 3000;
  const geometry  = new THREE.BufferGeometry();
  const positions = new Float32Array(NB_PARTICULES * 3);

  for (let i = 0; i < NB_PARTICULES * 3; i++) {
    positions[i] = (Math.random() - 0.5) * 100;
  }
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  const material = new THREE.PointsMaterial({
    color      : 0x00ffb4,
    size       : 0.15,
    transparent: true,
    opacity    : 0.6,
    sizeAttenuation: true
  });

  const particles = new THREE.Points(geometry, material);
  scene.add(particles);

  function animate() {
    requestAnimationFrame(animate);
    particles.rotation.x += 0.0002;
    particles.rotation.y += 0.0003;
    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
}

initThreeJS();


// --- Animations GSAP ---

gsap.registerPlugin(ScrollTrigger);

const heroTimeline = gsap.timeline({ delay: 0.3 });
heroTimeline
  .to("#hero-badge",   { opacity: 1, y: 0, duration: 0.6, ease: "power2.out" })
  .to("#hero-title",   { opacity: 1, y: 0, duration: 0.8, ease: "power2.out" }, "-=0.3")
  .to(".hero-desc",    { opacity: 1, y: 0, duration: 0.6, ease: "power2.out" }, "-=0.4")
  .to(".hero-cta",     { opacity: 1, y: 0, duration: 0.5, ease: "power2.out" }, "-=0.3")
  .to(".hero-stats",   { opacity: 1, y: 0, duration: 0.5, ease: "power2.out" }, "-=0.3");

gsap.utils.toArray(".section-header").forEach(el => {
  gsap.fromTo(el,
    { opacity: 0, x: -30 },
    { opacity: 1, x: 0, duration: 0.7, ease: "power2.out",
      scrollTrigger: { trigger: el, start: "top 85%" }
    }
  );
});

gsap.utils.toArray(".arch-card").forEach((card, i) => {
  gsap.fromTo(card,
    { opacity: 0, y: 30 },
    { opacity: 1, y: 0, duration: 0.5, delay: i * 0.1, ease: "power2.out",
      scrollTrigger: { trigger: card, start: "top 90%" }
    }
  );
});


// --- Animations Anime.js ---

anime({ targets: ".brand-text", opacity: [0, 1], duration: 1000, delay: 500, easing: "easeInOutQuad" });

function animateBanner() {
  anime({
    targets : "#best-banner",
    borderLeftColor: ["rgba(0,255,180,0.3)", "rgba(0,255,180,1)", "rgba(0,255,180,0.3)"],
    duration: 3000,
    loop    : true,
    easing  : "easeInOutSine"
  });
}


// --- Navbar ---

window.addEventListener("scroll", () => {
  document.getElementById("navbar").classList.toggle("scrolled", window.scrollY > 40);
});

document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener("click", e => {
    const target = document.querySelector(a.getAttribute("href"));
    if (target) {
      e.preventDefault();
      window.scrollTo({ top: target.getBoundingClientRect().top + window.scrollY - 64, behavior: "smooth" });
    }
  });
});


// --- Upload & Prédiction ---

const fileInput    = document.getElementById("file-input");
const uploadZone   = document.getElementById("upload-zone");
const previewImg   = document.getElementById("preview-img");
const placeholder  = document.getElementById("upload-placeholder");
const btnPredict   = document.getElementById("btn-predict");
const resultIdle   = document.getElementById("result-idle");
const resultOutput = document.getElementById("result-output");

let fichierSelectionne = null;

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
    anime({ targets: "#preview-img", opacity: [0, 1], scale: [0.9, 1], duration: 400, easing: "easeOutQuad" });
  };
  reader.readAsDataURL(file);
});

uploadZone.addEventListener("dragover",  (e) => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
uploadZone.addEventListener("dragleave", ()  => { uploadZone.classList.remove("drag-over"); });
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    fileInput.dispatchEvent(new Event("change"));
  }
});

btnPredict.addEventListener("click", async () => {
  if (!fichierSelectionne) return;

  btnPredict.querySelector(".btn-scan-text").textContent = "ANALYSE EN COURS…";
  btnPredict.disabled = true;

  anime({ targets: ".btn-scan-icon", rotate: "+=360", duration: 800, loop: true, easing: "linear" });

  const formData = new FormData();
  formData.append("image", fichierSelectionne);

  try {
    const res  = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();
    anime.remove(".btn-scan-icon");
    if (data.erreur) alert("Erreur : " + data.erreur);
    else afficherResultat(data);
  } catch (err) {
    anime.remove(".btn-scan-icon");
    alert("Impossible de joindre l'API Flask.");
  }

  btnPredict.querySelector(".btn-scan-text").textContent = "LANCER L'ANALYSE";
  btnPredict.disabled = false;
});

function afficherResultat(data) {
  const estParasite = data.classe_id === 0;
  const cls         = estParasite ? "parasitized" : "uninfected";

  document.getElementById("result-verdict").textContent = estParasite ? "⚠ ALERTE PARASITISME" : "✓ CELLULE SAINE";
  document.getElementById("result-verdict").className   = `result-verdict ${cls}`;
  document.getElementById("result-label").textContent   = data.label;
  document.getElementById("result-label").className     = `result-label ${cls}`;
  document.getElementById("conf-pct").textContent       = data.probabilite + "%";
  document.getElementById("conf-fill").className        = `conf-fill ${cls}`;
  document.getElementById("conf-fill").style.width      = "0%";

  resultIdle.classList.add("hidden");
  resultOutput.classList.remove("hidden");

  gsap.fromTo("#result-output", { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.5 });
  anime({ targets: "#conf-fill", width: data.probabilite + "%", duration: 1200, easing: "easeOutCubic" });
}

document.getElementById("btn-reset").addEventListener("click", () => {
  resultOutput.classList.add("hidden");
  resultIdle.classList.remove("hidden");
  previewImg.classList.add("hidden");
  placeholder.classList.remove("hidden");
  fileInput.value      = "";
  fichierSelectionne   = null;
  btnPredict.disabled  = true;
});


// --- Métriques ---

async function chargerMetriques() {
  try {
    const res  = await fetch("/metrics");
    const data = await res.json();

    if (!data.meilleur_modele) return;

    const meilleur = data.meilleur_modele;
    const m        = data.resultats[meilleur];
    const accStr   = (m.accuracy * 100).toFixed(1) + "%";

    document.getElementById("hero-acc").textContent      = accStr;
    document.getElementById("stat-acc").textContent      = accStr;
    document.getElementById("best-model-name").textContent = meilleur;
    document.getElementById("m-acc").textContent  = (m.accuracy  * 100).toFixed(1) + "%";
    document.getElementById("m-prec").textContent = (m.precision * 100).toFixed(1) + "%";
    document.getElementById("m-rec").textContent  = (m.recall    * 100).toFixed(1) + "%";
    document.getElementById("m-f1").textContent   = (m.f1_score  * 100).toFixed(1) + "%";

    const tbody = document.getElementById("models-tbody");
    tbody.innerHTML = "";
    for (const [nom, met] of Object.entries(data.resultats)) {
      const estMeilleur = nom === meilleur;
      const tr = document.createElement("tr");
      if (estMeilleur) tr.className = "best-row";
      tr.innerHTML = `
        <td>${nom}${estMeilleur ? ' <span class="badge-best">MEILLEUR</span>' : ""}</td>
        <td>${(met.accuracy  * 100).toFixed(2)}%</td>
        <td>${(met.precision * 100).toFixed(2)}%</td>
        <td>${(met.recall    * 100).toFixed(2)}%</td>
        <td>${(met.f1_score  * 100).toFixed(2)}%</td>
        <td>${estMeilleur ? "✓ Déployé" : "—"}</td>
      `;
      tbody.appendChild(tr);
    }

    animateBanner();

  } catch {
    document.getElementById("best-model-name").textContent = "Lancez d'abord : python main.py";
  }
}

chargerMetriques();
