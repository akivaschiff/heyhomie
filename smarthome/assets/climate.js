"use strict";

import { api, el, esc, dirAttr, spinBtn, setCount } from "./core.js";

// temperature formatting: keep one decimal only when meaningful
function fmtTemp(v){
  if(v===null||v===undefined||v==="") return null;
  const n = Number(v);
  if(!isFinite(n)) return null;
  return n;
}
function tempHTML(v){
  const n = fmtTemp(v);
  if(n===null) return '<span class="nowval">--<span class="u">°</span></span>';
  const whole = Math.trunc(n);
  const hasDec = Math.abs(n - whole) >= 0.05;
  const dec = hasDec ? '<span class="dec">.'+Math.round(Math.abs(n-whole)*10)+'</span>' : '';
  return '<span class="nowval">'+whole+dec+'<span class="u">°</span></span>';
}
function setLabel(t){
  if(t===null||t===undefined) return '--';
  const n = Number(t);
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}
// which accent a climate unit should wear, by mode
function acAccent(mode, on){
  if(!on) return 'var(--ink-dim)';
  const m = String(mode||"").toUpperCase();
  if(m.includes("HEAT")) return 'var(--heat)';
  if(m.includes("COOL")||m==="AUTO"||m.includes("DRY")) return 'var(--cool)';
  return 'var(--jade)';
}

/* ================= shared climate card (Electra + Midea) ================= */
function acCard(m, endpoint, reload, reconcileMs){
  const offline = !!m.offline;
  const accent = acAccent(m.mode, m.on);
  const modeOpts=(m.modes||[]).filter(x=>x!=="STBY")
     .map(x=>`<option ${x===m.mode?'selected':''}>${esc(x)}</option>`).join("") || '<option>COOL</option>';
  const fanOpts =(m.fans||[]).map(x=>`<option ${x===m.fan?'selected':''}>${esc(x)}</option>`).join("") || '<option>AUTO</option>';

  const statusBadge = m.error ? `<span class="badge err">Error</span>`
                    : offline ? `<span class="badge off-b">Offline</span>` : '';

  const c=el(`<div class="clim ${m.on&&!offline?'active':''} ${!m.on?'off':''}" style="--ac:${accent}">
    <span class="aura"></span>
    <div class="chead">
      <div>
        <div class="cname"${dirAttr(m.name)}>${esc(m.name)}</div>
        <div class="cwhere">${esc(m.where||'Climate')}</div>
      </div>
      ${offline?statusBadge:`<button class="pw ${m.on?'on':''}" aria-label="power"></button>`}
    </div>
    <div class="hero">
      <div class="now">
        <span class="nowlbl">Room</span>
        ${tempHTML(m.current)}
      </div>
      <div class="setblock">
        <span class="setlbl">Set to</span>
        <div class="stepper">
          <button class="step minus">−</button>
          <span class="setval">${setLabel(m.target)}<span class="u">°</span></span>
          <button class="step plus">+</button>
        </div>
      </div>
    </div>
    <div class="cfoot">
      <div class="selwrap"><span class="sl">Mode</span><select class="mode">${modeOpts}</select></div>
      <div class="selwrap"><span class="sl">Fan</span><select class="fan">${fanOpts}</select></div>
    </div>
  </div>`);

  if(offline){ c.classList.add("disabled"); return c; }

  const send=(payload)=>{
    c.classList.add("busy");
    api(endpoint,{id:m.id,...payload})
      .catch(()=>{})
      .finally(()=>{ c.classList.remove("busy"); setTimeout(()=>reload(), reconcileMs); });
  };

  const applyAccent=(on,mode)=>{
    const a=acAccent(mode,on);
    c.style.setProperty("--ac",a);
    c.classList.toggle("active", on);
    c.classList.toggle("off", !on);
  };

  const pw=c.querySelector(".pw");
  pw.onclick=()=>{
    const next=!pw.classList.contains("on");
    pw.classList.toggle("on",next);
    applyAccent(next, c.querySelector(".mode").value);   // optimistic
    send({power:next});
  };

  const setval=c.querySelector(".setval");
  let t = (m.target!==null && m.target!==undefined) ? Number(m.target) : null;
  const stepBtns=c.querySelectorAll(".step");
  stepBtns[0].onclick=()=>{ if(t===null) return; if(t>m.min){ t=Math.round(t-1); setval.innerHTML=setLabel(t)+'<span class="u">°</span>'; send({temp:t}); } };
  stepBtns[1].onclick=()=>{ if(t===null) return; if(t<m.max){ t=Math.round(t+1); setval.innerHTML=setLabel(t)+'<span class="u">°</span>'; send({temp:t}); } };

  const modeSel=c.querySelector(".mode");
  modeSel.onchange=(e)=>{ applyAccent(pw.classList.contains("on"), e.target.value); send({mode:e.target.value}); };
  c.querySelector(".fan").onchange=(e)=>send({fan:e.target.value});

  return c;
}

/* ================= Electra — central (cloud, optimistic) ================= */
export async function loadElectra(btn){
  spinBtn(btn);
  const wrap=document.getElementById("central");
  if(!wrap.children.length || wrap.querySelector(".loading"))
    wrap.innerHTML='<div class="loading"><span class="spinner"></span>Reading units…</div>';
  let devs;
  try{ devs=await api("/api/electra"); }
  catch(e){ wrap.innerHTML='<div class="empty">Electra unreachable.</div>'; return; }
  wrap.innerHTML=""; let coolCount=0;
  for(const d of devs){
    const offline = d.on===null || !!d.error;
    if(!offline && d.on && /COOL|AUTO|DRY/i.test(d.mode||"")) coolCount++;
    wrap.appendChild(acCard({
      id:d.id, name:d.name, where:d.kind||'Central', on:!!d.on,
      current:d.current, target:(d.target!==null&&d.target!==undefined)?Number(d.target):null,
      min:d.min, max:d.max, mode:d.mode, modes:d.modes, fan:d.fan, fans:d.fans,
      offline, error:d.error
    }, "/api/electra/set", loadElectra, 2500));
  }
  if(!devs.length) wrap.innerHTML='<div class="empty">No central units.</div>';
  setCount("central-count", coolCount?coolCount+" COOLING":(devs.length+" UNIT"+(devs.length!==1?"S":"")), false, coolCount>0);
}

/* ================= Midea — splits (local, fast) ================= */
export async function loadMidea(btn){
  spinBtn(btn);
  const wrap=document.getElementById("splits");
  if(!wrap.children.length || wrap.querySelector(".loading"))
    wrap.innerHTML='<div class="loading"><span class="spinner"></span>Reading units…</div>';
  let devs;
  try{ devs=await api("/api/midea"); }
  catch(e){ wrap.innerHTML='<div class="empty">Midea unreachable.</div>'; return; }
  wrap.innerHTML=""; let coolCount=0;
  for(const d of devs){
    const offline=!d.online;
    if(!offline && d.power && /COOL|AUTO|DRY/i.test(d.mode||"")) coolCount++;
    wrap.appendChild(acCard({
      id:d.id, name:d.name, where:'Split · '+(d.ip||''), on:!!d.power,
      current:d.indoor, target:d.target,
      min:d.min, max:d.max, mode:d.mode, modes:d.modes, fan:d.fan, fans:d.fans,
      offline
    }, "/api/midea/set", loadMidea, 1200));
  }
  if(!devs.length) wrap.innerHTML='<div class="empty">No split units.</div>';
  setCount("splits-count", coolCount?coolCount+" COOLING":(devs.length+" ROOM"+(devs.length!==1?"S":"")), false, coolCount>0);
}
