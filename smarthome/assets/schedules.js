"use strict";

import { api, el, esc, dirAttr, spinBtn, setCount } from "./core.js";

const RECURS = [
  {v:"daily", label:"Daily"},
  {v:"weekdays", label:"Weekdays (Sun–Thu)"},
  {v:"weekends", label:"Weekend (Fri–Sat)"},
  {v:"sun_fri", label:"Sun–Fri (except Shabbat)"},
  {v:"once", label:"Once"},
];

const named = e => e.name && !/channel\s*\d/i.test(e.name);

// Build the list of schedulable targets from the live device inventory.
async function inventory(){
  const [hig, ele, mid] = await Promise.all([
    api("/api/higoal").catch(()=>[]),
    api("/api/electra").catch(()=>[]),
    api("/api/midea").catch(()=>[]),
  ]);
  const lights=[], blinds=[], acs=[];
  for(const d of (hig||[])){
    for(const e of (d.entities||[])){
      if(!named(e)) continue;
      if(e.type==="shutter") blinds.push({name:e.name, device:d.id, idx:e.idx});
      else lights.push({name:e.name, device:d.id, idx:e.idx});
    }
  }
  for(const u of (ele||[])) if(u.id!=null) acs.push({name:u.name, system:"electra", id:u.id});
  for(const u of (mid||[])) if(u.id!=null) acs.push({name:u.name, system:"midea", id:u.id});

  const targets=[];
  if(lights.length){
    targets.push({label:"All lights", kind:"light", build:a=>lights.map(l=>({system:"higoal",payload:{device:l.device,idx:l.idx,on:a==="on"}}))});
    for(const l of lights) targets.push({label:l.name, kind:"light", build:a=>[{system:"higoal",payload:{device:l.device,idx:l.idx,on:a==="on"}}]});
  }
  if(blinds.length){
    targets.push({label:"All blinds", kind:"blind", build:a=>blinds.map(b=>({system:"higoal",payload:{device:b.device,idx:a==="open"?b.idx:b.idx+1,on:true}}))});
    for(const b of blinds) targets.push({label:b.name, kind:"blind", build:a=>[{system:"higoal",payload:{device:b.device,idx:a==="open"?b.idx:b.idx+1,on:true}}]});
  }
  for(const u of acs) targets.push({label:u.name, kind:"ac", build:(a,x)=>[{system:u.system,payload:{id:u.id,power:a==="on",...(a==="on"&&x.temp?{temp:x.temp}:{})}}]});
  return targets;
}

const ACTIONS = { light:["on","off"], ac:["on","off"], blind:["open","close"] };

let _targets = [];

export async function loadSchedules(btn){
  spinBtn(btn);
  const listWrap = document.getElementById("sched-list");
  let items, master;
  try{
    [items, master] = await Promise.all([
      api("/api/schedules"),
      api("/api/schedules/enabled").catch(()=>({enabled:true})),
    ]);
  }
  catch(e){ listWrap.innerHTML='<div class="empty">Schedules unreachable.</div>'; return; }

  const enabled = master?.enabled !== false;
  listWrap.innerHTML="";
  listWrap.classList.toggle("sched-paused", !enabled);

  const bar = el(`<div class="sched-master${enabled?"":" is-off"}">
    <div class="sm-text">
      <div class="sm-title">All schedules</div>
      <div class="sm-sub">${enabled ? "Active — running normally" : "Paused — nothing fires until you switch back on"}</div>
    </div>
    <button class="pw${enabled?" on":""}" role="switch" aria-checked="${enabled}" aria-label="toggle all schedules"></button>
  </div>`);
  bar.querySelector(".pw").onclick=async()=>{
    bar.classList.add("busy");
    await api("/api/schedules/enabled", {enabled: !enabled}, "POST").catch(()=>{});
    loadSchedules();
  };
  listWrap.appendChild(bar);

  if(!items.length){
    listWrap.appendChild(el('<div class="empty">No schedules yet.</div>'));
  }
  for(const s of items){
    const meta = [RECURS.find(r=>r.v===s.recur)?.label || s.recur, s.date].filter(Boolean).join(" · ");
    const row = el(`<div class="sched">
      <div class="sched-time">${esc(s.time || "—")}</div>
      <div class="sched-main">
        <div class="sched-desc"${dirAttr(s.description)}>${esc(s.description||s.id)}</div>
        <div class="sched-meta">${esc(meta)}</div>
      </div>
      <button class="sched-del" aria-label="delete">✕</button>
    </div>`);
    row.querySelector(".sched-del").onclick=async()=>{
      row.classList.add("busy");
      await api("/api/schedules/"+encodeURIComponent(s.id), null, "DELETE").catch(()=>{});
      loadSchedules();
    };
    listWrap.appendChild(row);
  }
  setCount("sched-count", items.length+"", false);

  const formWrap = document.getElementById("sched-form-wrap");
  if(!_targets.length){
    formWrap.innerHTML='<div class="loading"><span class="spinner"></span>Loading devices…</div>';
    _targets = await inventory();
  }
  renderForm();
}

function renderForm(){
  const wrap = document.getElementById("sched-form-wrap");
  const targetOpts = _targets.map((t,i)=>`<option value="${i}">${esc(t.label)}</option>`).join("");
  const recurOpts = RECURS.map(r=>`<option value="${r.v}">${r.label}</option>`).join("");
  wrap.innerHTML = `<div class="sched-form">
    <div class="sf-title">New schedule</div>
    <div class="sf-grid">
      <label class="sf-field"><span>Target</span><select id="sf-target">${targetOpts}</select></label>
      <label class="sf-field"><span>Action</span><select id="sf-action"></select></label>
      <label class="sf-field"><span>Time</span><input type="time" id="sf-time" value="08:00"></label>
      <label class="sf-field"><span>Repeat</span><select id="sf-recur">${recurOpts}</select></label>
      <label class="sf-field" id="sf-date-field" hidden><span>Date</span><input type="date" id="sf-date"></label>
      <label class="sf-field" id="sf-temp-field" hidden><span>Temp °C</span><input type="number" id="sf-temp" min="16" max="30" value="22"></label>
    </div>
    <div class="sf-foot">
      <span class="sf-msg" id="sf-msg"></span>
      <button class="sf-add" id="sf-add">Add schedule</button>
    </div>
  </div>`;

  const targetSel = wrap.querySelector("#sf-target");
  const actionSel = wrap.querySelector("#sf-action");
  const recurSel = wrap.querySelector("#sf-recur");
  const dateField = wrap.querySelector("#sf-date-field");
  const tempField = wrap.querySelector("#sf-temp-field");

  const syncTarget=()=>{
    const t=_targets[targetSel.value];
    actionSel.innerHTML=(ACTIONS[t.kind]||[]).map(a=>`<option value="${a}">${a}</option>`).join("");
    syncTempVisibility();
  };
  const syncTempVisibility=()=>{
    const t=_targets[targetSel.value];
    tempField.hidden = !(t.kind==="ac" && actionSel.value==="on");
  };
  targetSel.onchange=syncTarget;
  actionSel.onchange=syncTempVisibility;
  recurSel.onchange=()=>{ dateField.hidden = recurSel.value!=="once"; };
  syncTarget();

  wrap.querySelector("#sf-add").onclick=()=>submitForm(wrap);
}

async function submitForm(wrap){
  const msg = wrap.querySelector("#sf-msg");
  const t = _targets[wrap.querySelector("#sf-target").value];
  const action = wrap.querySelector("#sf-action").value;
  const time = wrap.querySelector("#sf-time").value;
  const recur = wrap.querySelector("#sf-recur").value;
  const date = wrap.querySelector("#sf-date").value;
  const temp = Number(wrap.querySelector("#sf-temp").value);

  if(!time){ msg.textContent="Pick a time."; return; }
  if(recur==="once" && !date){ msg.textContent="Pick a date."; return; }

  const commands = t.build(action, {temp});
  const description = `${action} ${t.label}`;

  const btn = wrap.querySelector("#sf-add");
  btn.disabled=true; msg.textContent="";
  const res = await api("/api/schedules", {time, recur, date, description, commands}, "POST").catch(()=>({error:"failed"}));
  btn.disabled=false;
  if(res.error){ msg.textContent=res.error; return; }
  loadSchedules();
}
