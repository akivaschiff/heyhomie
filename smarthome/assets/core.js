"use strict";

const RTL_RE = /[֐-׿؀-ۿ]/;
export const isRTL = s => RTL_RE.test(String(s||""));
export const dirAttr = s => isRTL(s) ? ' dir="rtl"' : '';
export const esc = s => String(s??"").replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

export async function api(path, body){
  const opt = body ? {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)} : {};
  const r = await fetch(path, opt);
  return r.json();
}
export function el(html){const t=document.createElement("template");t.innerHTML=html.trim();return t.content.firstChild;}
export function spinBtn(btn){ if(!btn) return; btn.classList.add("spin"); setTimeout(()=>btn.classList.remove("spin"),700); }

export function setCount(id, text, live, cooling){
  const n=document.getElementById(id); if(!n) return;
  n.textContent=text;
  n.classList.toggle("live", !!live);
  n.classList.toggle("cooling", !!cooling);
}
