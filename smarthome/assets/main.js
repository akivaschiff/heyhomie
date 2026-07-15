"use strict";

import { loadHigoal } from "./higoal.js";
import { loadElectra, loadMidea } from "./climate.js";
import "./theme.js";

const LOADERS = { higoal: loadHigoal, electra: loadElectra, midea: loadMidea };

document.querySelectorAll("[data-refresh]").forEach(btn=>{
  btn.addEventListener("click", ()=>LOADERS[btn.dataset.refresh](btn));
});

loadHigoal(); loadElectra(); loadMidea();
