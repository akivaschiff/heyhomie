"use strict";

/* ================= theming: time-of-day + Shabbat ================= */
const GEO = { lat:31.78, lng:35.22 };        // Jerusalem
const CANDLE_OFFSET_MIN = 40;                // Friday candle-lighting = sunset - 40m
const NIGHTFALL_OFFSET_MIN = 40;             // Saturday nightfall     = sunset + 40m
const THEME_CYCLE = ["auto","day","night","shabbat"];
const TDOT = { day:"#38b6dc", night:"#4fd6a0", shabbat:"#f0b840" };

// NOAA solar-position algorithm -> local sunrise/sunset Date objects, no network.
function sunTimes(date, lat, lng){
  const rad=Math.PI/180, deg=180/Math.PI;
  const y=date.getFullYear(), mo=date.getMonth()+1, d=date.getDate();
  let yy=y, mm=mo;
  if(mm<=2){ yy-=1; mm+=12; }
  const A=Math.floor(yy/100), B=2-A+Math.floor(A/4);
  const JD=Math.floor(365.25*(yy+4716))+Math.floor(30.6001*(mm+1))+d+B-1524.5;
  const T=(JD-2451545)/36525;
  const L0=((280.46646+T*(36000.76983+T*0.0003032))%360+360)%360;
  const M=357.52911+T*(35999.05029-0.0001537*T);
  const e=0.016708634-T*(0.000042037+0.0000001267*T);
  const Mr=rad*M;
  const C=Math.sin(Mr)*(1.914602-T*(0.004817+0.000014*T))
         +Math.sin(2*Mr)*(0.019993-0.000101*T)
         +Math.sin(3*Mr)*0.000289;
  const trueLong=L0+C;
  const omega=125.04-1934.136*T;
  const lambda=trueLong-0.00569-0.00478*Math.sin(rad*omega);
  const eps0=23+(26+((21.448-T*(46.815+T*(0.00059-T*0.001813))))/60)/60;
  const eps=eps0+0.00256*Math.cos(rad*omega);
  const decl=deg*Math.asin(Math.sin(rad*eps)*Math.sin(rad*lambda));
  const vy=Math.tan(rad*eps/2)**2;
  const eot=4*deg*(vy*Math.sin(2*rad*L0)-2*e*Math.sin(Mr)
        +4*e*vy*Math.sin(Mr)*Math.cos(2*rad*L0)
        -0.5*vy*vy*Math.sin(4*rad*L0)-1.25*e*e*Math.sin(2*Mr));
  const zenith=90.833;
  const haCos=Math.cos(rad*zenith)/(Math.cos(rad*lat)*Math.cos(rad*decl))
            -Math.tan(rad*lat)*Math.tan(rad*decl);
  const clamp=Math.max(-1,Math.min(1,haCos));
  const ha=deg*Math.acos(clamp);          // degrees
  const noonUTCmin=720-4*lng-eot;         // lng positive east
  const mk=min=>new Date(Date.UTC(y,mo-1,d)+Math.round(min*60000));
  return {
    sunrise:mk(noonUTCmin-4*ha),
    sunset :mk(noonUTCmin+4*ha),
    polar:Math.abs(haCos)>1
  };
}

function autoTheme(now){
  const {sunrise,sunset}=sunTimes(now, GEO.lat, GEO.lng);
  const day=now.getDay();                              // 5=Fri, 6=Sat
  if(day===5 && now.getTime()>=sunset.getTime()-CANDLE_OFFSET_MIN*60000) return "shabbat";
  if(day===6 && now.getTime()<=sunset.getTime()+NIGHTFALL_OFFSET_MIN*60000) return "shabbat";
  return (now>=sunrise && now<sunset) ? "day" : "night";
}

let themeMode = "auto";
try{ const s=localStorage.getItem("themeMode"); if(THEME_CYCLE.includes(s)) themeMode=s; }catch(_){}

function effectiveTheme(now){ return themeMode==="auto" ? autoTheme(now) : themeMode; }

function applyTheme(){
  const now=new Date();
  const eff=effectiveTheme(now);
  if(document.documentElement.getAttribute("data-theme")!==eff)
    document.documentElement.setAttribute("data-theme",eff);
  const btn=document.getElementById("themebtn");
  document.getElementById("themebtn-label").textContent=
    themeMode==="auto" ? "AUTO" : themeMode.toUpperCase();
  btn.style.setProperty("--tdot", TDOT[eff]||TDOT.night);
  btn.title = themeMode==="auto"
    ? "Theme: Auto ("+eff+"). Tap to override."
    : "Theme: "+themeMode+" (manual). Tap to cycle.";
}

document.getElementById("themebtn").addEventListener("click",()=>{
  const i=THEME_CYCLE.indexOf(themeMode);
  themeMode=THEME_CYCLE[(i+1)%THEME_CYCLE.length];
  try{ localStorage.setItem("themeMode",themeMode); }catch(_){}
  applyTheme();
});

(function themeSanity(){
  const s=sunTimes(new Date(), GEO.lat, GEO.lng);
  const f=t=>t.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"});
  console.debug("[theme] Jerusalem sun today — sunrise",f(s.sunrise),
    "· sunset",f(s.sunset),
    "· candle-lighting",f(new Date(s.sunset.getTime()-CANDLE_OFFSET_MIN*60000)),
    "· nightfall",f(new Date(s.sunset.getTime()+NIGHTFALL_OFFSET_MIN*60000)),
    "· active",effectiveTheme(new Date()));
})();

/* ================= clock ================= */
let _lastThemeMin=-1;
function tick(){
  const d=new Date();
  const hh=String(d.getHours()).padStart(2,"0");
  const mm=String(d.getMinutes()).padStart(2,"0");
  const ss=String(d.getSeconds()).padStart(2,"0");
  document.getElementById("clock").innerHTML=`${hh}:${mm}<span class="sec">${ss}</span>`;
  document.getElementById("date").textContent=
    d.toLocaleDateString(undefined,{weekday:"long",day:"numeric",month:"long"});
  if(d.getMinutes()!==_lastThemeMin){ _lastThemeMin=d.getMinutes(); applyTheme(); }
}
applyTheme(); tick(); setInterval(tick,1000);
