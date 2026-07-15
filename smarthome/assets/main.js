"use strict";

import { loadHigoal } from "./higoal.js";
import { loadElectra, loadMidea } from "./climate.js";
import { loadSchedules } from "./schedules.js";
import "./theme.js";

const LOADERS = { higoal: loadHigoal, electra: loadElectra, midea: loadMidea, schedules: loadSchedules };

document.querySelectorAll("[data-refresh]").forEach(btn=>{
  btn.addEventListener("click", ()=>LOADERS[btn.dataset.refresh](btn));
});

const loaded = new Set(["home"]);
document.querySelectorAll(".tab-btn").forEach(btn=>{
  btn.addEventListener("click", ()=>{
    const tab = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach(b=>b.classList.toggle("active", b===btn));
    document.querySelectorAll(".tabpane").forEach(p=>p.classList.toggle("active", p.id==="tab-"+tab));
    if(!loaded.has(tab)){ loaded.add(tab); LOADERS[tab]?.(); }
  });
});

loadHigoal(); loadElectra(); loadMidea();
