"use strict";

import { api, el, esc, dirAttr, spinBtn, setCount } from "./core.js";

const ICON = {
  bulb:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6"/><path d="M10 21h4"/><path d="M12 3a6 6 0 0 0-4 10.5c.7.7 1 1.4 1 2.5h6c0-1.1.3-1.8 1-2.5A6 6 0 0 0 12 3Z"/></svg>',
  blind:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><rect x="4" y="3" width="16" height="4" rx="1"/><path d="M6 7v11M18 7v11M6 18h12"/><path d="M9 11h6M9 15h6"/></svg>',
  up:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V6M6 12l6-6 6 6"/></svg>',
  down:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v13M18 12l-6 6-6-6"/></svg>'
};

function named(e){ return e.name && !/channel\s*\d/i.test(e.name); }

function blindState(pct){
  if(pct===null||pct===undefined) return null;
  const p = Number(pct);
  if(!isFinite(p)) return null;
  if(p<=0.05) return {label:"Open", cls:"open"};
  if(p>=0.95) return {label:"Closed", cls:"closed"};
  return {label:Math.round((1-p)*100)+"% open", cls:"part"};
}

export async function loadHigoal(btn){
  spinBtn(btn);
  const lightsWrap = document.getElementById("lights");
  const blindsWrap = document.getElementById("blinds");
  let devs;
  try{ devs = await api("/api/higoal"); }
  catch(e){ lightsWrap.innerHTML='<div class="empty">Higoal unreachable.</div>'; return; }

  lightsWrap.innerHTML=""; blindsWrap.innerHTML="";
  let onCount=0, blindCount=0;

  for(const d of devs){
    const panel = d.device && d.device.trim() ? d.device.trim() : d.id;
    for(const e of (d.entities||[])){
      if(!named(e)) continue;

      if(e.type==="shutter"){
        blindCount++;
        const st = blindState(e.percentage);
        const stHTML = st ? `<span class="bstate ${st.cls}">${st.label}</span>` : '';
        const c = el(`<div class="blind">
          <div class="bhead">
            <div class="bico">${ICON.blind}</div>
            <div class="bnamewrap">
              <div class="bname"${dirAttr(e.name)}>${esc(e.name)}</div>
              ${stHTML}
            </div>
          </div>
          <div class="bbtns">
            <button class="bbtn open">${ICON.up}Open</button>
            <button class="bbtn close">${ICON.down}Close</button>
          </div></div>`);
        const [op,cl]=c.querySelectorAll("button");
        op.onclick=()=>{ flash(op,"flash-open"); higoalSet(d.id,e.idx,true); scheduleHigoalReconcile(); };
        cl.onclick=()=>{ flash(cl,"flash-close"); higoalSet(d.id,e.idx+1,true); scheduleHigoalReconcile(); };
        blindsWrap.appendChild(c);
        continue;
      }

      // switch / dimmer -> light tile (whole tile toggles)
      const on = !!e.on;
      if(on) onCount++;
      const c = el(`<button class="light ${on?'on':''}">
        <span class="halo"></span>
        <div class="ltop">
          <div class="bulb">${ICON.bulb}</div>
          <span class="sw"></span>
        </div>
        <div class="lname"${dirAttr(e.name)}>${esc(e.name)}</div>
        <div class="lsub"><span class="st">${on?'On':'Off'}</span>${e.online===false?' · <span>offline</span>':''}</div>
      </button>`);
      c.onclick=()=>{
        const next=!c.classList.contains("on");
        c.classList.toggle("on",next);
        c.querySelector(".lsub .st").textContent = next?'On':'Off';
        higoalSet(d.id,e.idx,next);          // optimistic: fire, don't await UI
        scheduleHigoalReconcile();           // reconcile real state after ~3s
      };
      lightsWrap.appendChild(c);
    }
  }

  if(!lightsWrap.children.length) lightsWrap.innerHTML='<div class="empty">No named lights.</div>';
  if(!blindsWrap.children.length) blindsWrap.innerHTML='<div class="empty">No named blinds.</div>';
  setCount("lights-count", onCount ? onCount+" ON" : "ALL OFF", onCount>0);
  setCount("blinds-count", blindCount+"", false);
}
function flash(btn, cls){ btn.classList.add(cls); setTimeout(()=>btn.classList.remove(cls),450); }
function higoalSet(device,idx,on){ return api("/api/higoal/set",{device,idx,on}); }
let _higoalTimer=null;
function scheduleHigoalReconcile(){ clearTimeout(_higoalTimer); _higoalTimer=setTimeout(()=>loadHigoal(), 3000); }
