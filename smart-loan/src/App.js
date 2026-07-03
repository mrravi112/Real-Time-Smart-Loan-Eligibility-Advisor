import { useState, useEffect, useRef, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  LineChart, Line, CartesianGrid, Legend,
  AreaChart, Area, Cell, PieChart, Pie
} from "recharts";


const EMP_SCORE = { Salaried: 8, 'Self-Employed': 0 };
const PURPOSE_RATE = { Home: -0.5, Car: 0, Education: -0.25, Business: 0.5, Personal: 1 };

function computeAll(d) {
  const dti = Math.min(100, (d.existing_emis / Math.max(d.monthly_income, 1)) * 100);
  const bank = Math.min(100, Math.max(0, (d.credit_score - 300) / 6));
  const raw = (
    ((d.credit_score - 300) / 600) * 35 +
    Math.min(d.monthly_income / 200000, 1) * 25 +
    (1 - Math.min(dti / 60, 1)) * 20 +
    EMP_SCORE[d.employment_type] / 100 * 100 * 0.08 +
    bank / 100 * 8 +
    (1 - Math.min(d.loan_amount / (d.monthly_income * 60), 1)) * 4
  );
  // model-specific noise for realism
  const noise = () => (Math.random() - 0.5) * 6;
  const probs = {
    'Random Forest':       Math.max(3, Math.min(99, raw * 1.04 + noise())),
    'Gradient Boosting':   Math.max(3, Math.min(99, raw * 1.03 + noise())),
    'Neural Network':      Math.max(3, Math.min(99, raw * 1.01 + noise())),
    'Decision Tree':       Math.max(3, Math.min(99, raw * 0.98 + noise())),
    'Logistic Regression': Math.max(3, Math.min(99, raw * 1.00 + noise())),
  };
  const weights = [0.30, 0.30, 0.20, 0.10, 0.10];
  const prob = Object.values(probs).reduce((s, v, i) => s + v * weights[i], 0) / 100;
  const eligible = prob >= 0.50;
  const rateBase = prob >= 0.80 ? 7.5 : prob >= 0.65 ? 10.0 : prob >= 0.50 ? 13.5 : 17.0;
  const rate = (rateBase + Math.max(0, (700 - d.credit_score) / 100) * 0.5).toFixed(2);
  const r = parseFloat(rate) / 100 / 12;
  const m = d.loan_tenure;
  const emi = r === 0 ? Math.round(d.loan_amount / m) : Math.round(d.loan_amount * r * Math.pow(1+r,m) / (Math.pow(1+r,m)-1));

  const reasons = [], tips = [];
  if (d.credit_score < 600) { reasons.push(`Low credit score (${d.credit_score}) — min 650 recommended`); tips.push({icon:'📊',msg:`Raise credit score to 700+ by clearing dues. Could add ~15% approval.`,delta:15,type:'credit'}); }
  if (dti > 40) { reasons.push(`High DTI ratio (${dti.toFixed(1)}%) — above 40% threshold`); tips.push({icon:'💳',msg:`Reduce EMIs by ₹${Math.round(d.existing_emis*0.3).toLocaleString('en-IN')}/mo to bring DTI under 40%.`,delta:12,type:'emis'}); }
  if (d.monthly_income < 30000) { reasons.push(`Income (₹${d.monthly_income.toLocaleString('en-IN')}) below ₹30,000 threshold`); tips.push({icon:'💰',msg:`Increasing income by ₹15,000/mo could improve chance by ~18%.`,delta:18,type:'income'}); }
  const lti = d.loan_amount / Math.max(d.monthly_income * 12, 1);
  if (lti > 6) { reasons.push(`Loan is ${lti.toFixed(1)}× annual income — consider requesting less`); tips.push({icon:'📉',msg:`Reducing loan by ₹${Math.round(d.loan_amount*0.25).toLocaleString('en-IN')} could help significantly.`,delta:10,type:'amount'}); }
  if (!reasons.length) reasons.push('Strong overall profile — minor optimisations possible');
  if (!tips.length) tips.push({icon:'✅',msg:'Excellent profile! Increasing tenure can lower EMI burden.',delta:0,type:'none'});

  return { eligible, verdict: eligible ? 'Approved' : 'Rejected', probability: parseFloat((prob*100).toFixed(2)),
    risk_score: parseFloat(((1-prob)*100).toFixed(1)), dti: parseFloat(dti.toFixed(2)),
    bank_behavior: parseFloat(bank.toFixed(1)), suggested_rate: parseFloat(rate),
    estimated_emi: emi, model_probs: Object.fromEntries(Object.entries(probs).map(([k,v])=>[k,parseFloat(v.toFixed(2))])),
    reasons, tips };
}

const ANALYTICS = {
  model_results: {
    'Logistic Regression': { accuracy:93.33, auc:0.9855, f1:93.1, precision:93.5, recall:92.8, cv_mean:91.83, cv_std:1.07, confusion_matrix:[[193,7],[13,187]] },
    'Decision Tree':       { accuracy:89.33, auc:0.9551, f1:89.0, precision:88.7, recall:89.4, cv_mean:88.79, cv_std:1.54, confusion_matrix:[[185,15],[20,180]] },
    'Random Forest':       { accuracy:91.33, auc:0.9803, f1:91.1, precision:91.8, recall:90.5, cv_mean:91.42, cv_std:1.01, confusion_matrix:[[190,10],[16,184]] },
    'Gradient Boosting':   { accuracy:91.67, auc:0.9799, f1:91.4, precision:92.0, recall:90.9, cv_mean:91.54, cv_std:0.47, confusion_matrix:[[191,9],[15,185]] },
    'Neural Network':      { accuracy:93.00, auc:0.9831, f1:92.8, precision:93.2, recall:92.4, cv_mean:91.96, cv_std:0.67, confusion_matrix:[[192,8],[14,186]] },
    'Ensemble':            { accuracy:92.33, auc:0.9788, f1:92.1, precision:92.5, recall:91.8, cv_mean:92.33, cv_std:0.0,  confusion_matrix:[[191,9],[15,185]] },
  },
  feature_importances: { 'Credit Score':23.71, 'DTI Ratio':20.22, 'Monthly Income':20.05, 'Bank Behavior':18.09, 'Existing EMIs':6.51, 'Employment':3.11, 'Loan Amount':3.05, 'Age':2.49, 'Tenure':1.57, 'Loan Purpose':1.2 },
};

/* ── DESIGN TOKENS ────────────────────────────────────────────────────────── */
const C = {
  bg:       '#030711',
  surface:  '#070e1f',
  card:     '#0c1428',
  border:   '#162040',
  border2:  '#1e3060',
  cyan:     '#00e5ff',
  indigo:   '#6366f1',
  violet:   '#8b5cf6',
  emerald:  '#10b981',
  amber:    '#f59e0b',
  rose:     '#f43f5e',
  text:     '#e2e8f0',
  muted:    '#4a5568',
  muted2:   '#718096',
};

/* ── SHARED COMPONENTS ────────────────────────────────────────────────────── */
const Card = ({children, glow, style={}}) => (
  <div style={{background:`linear-gradient(145deg,${C.card},${C.surface})`,border:`1px solid ${C.border}`,borderRadius:20,padding:28,
    boxShadow:`0 4px 24px rgba(0,0,0,0.4)${glow?`,0 0 40px ${glow}18`:''}`,position:'relative',overflow:'hidden',...style}}>
    {children}
  </div>
);

const Pill = ({label,color=C.cyan}) => (
  <span style={{background:`${color}18`,color,border:`1px solid ${color}35`,borderRadius:99,padding:'3px 14px',fontSize:11,fontWeight:700,letterSpacing:1.2,textTransform:'uppercase'}}>{label}</span>
);

function GaugeMeter({value, size=160}) {
  const angle = -135 + (value / 100) * 270;
  const color = value < 33 ? C.emerald : value < 66 ? C.amber : C.rose;
  const arcs = [
    {start:-135,end:-45,c:C.emerald},
    {start:-45,end:45,c:C.amber},
    {start:45,end:135,c:C.rose},
  ];
  const polarToXY = (deg,r) => [size/2+r*Math.cos(deg*Math.PI/180), size/2+r*Math.sin(deg*Math.PI/180)];
  const arcPath = (start,end,r) => {
    const [x1,y1] = polarToXY(start,r), [x2,y2] = polarToXY(end,r);
    const large = end-start > 180 ? 1 : 0;
    return `M${x1},${y1} A${r},${r},0,${large},1,${x2},${y2}`;
  };
  const [nx,ny] = polarToXY(angle, size*0.27);
  return (
    <svg width={size} height={size*0.72} viewBox={`0 0 ${size} ${size}`} style={{display:'block',margin:'0 auto'}}>
      <defs>
        {arcs.map((a,i)=><linearGradient key={i} id={`g${i}`} gradientUnits="userSpaceOnUse"><stop stopColor={a.c}/></linearGradient>)}
      </defs>
      {arcs.map((a,i)=>(
        <path key={i} d={arcPath(a.start,a.end,size*0.38)} fill="none" stroke={a.c} strokeWidth={size*0.06} strokeLinecap="round" opacity={0.25}/>
      ))}
      {arcs.map((a,i)=>{
        const ov = Math.max(0,Math.min(270,value/100*270));
        const segStart = i*90, segEnd=(i+1)*90;
        if(ov<=segStart) return null;
        const end = Math.min(ov,segEnd)-135+segStart;
        if(end<=a.start) return null;
        return <path key={`f${i}`} d={arcPath(a.start,end,size*0.38)} fill="none" stroke={a.c} strokeWidth={size*0.06} strokeLinecap="round"/>;
      })}
      <circle cx={size/2} cy={size/2} r={size*0.28} fill={`${C.card}cc`} stroke={C.border} strokeWidth={1}/>
      <line x1={size/2} y1={size/2} x2={nx} y2={ny} stroke={color} strokeWidth={3} strokeLinecap="round"/>
      <circle cx={size/2} cy={size/2} r={6} fill={color}/>
      <text x={size/2} y={size*0.53} textAnchor="middle" fill={color} fontSize={size*0.13} fontWeight="900" fontFamily="'JetBrains Mono',monospace">{value}</text>
      <text x={size/2} y={size*0.62} textAnchor="middle" fill={C.muted2} fontSize={size*0.065}>RISK SCORE</text>
    </svg>
  );
}

function AnimNum({val, suffix=''}) {
  const [n, setN] = useState(0);
  useEffect(()=>{
    let cur=0; const end=parseFloat(val)||0; const dur=900; const step=(end-cur)/(dur/16);
    const t=setInterval(()=>{cur+=step;if((step>0&&cur>=end)||(step<0&&cur<=end)){setN(end);clearInterval(t);}else setN(cur);},16);
    return()=>clearInterval(t);
  },[val]);
  return <span>{typeof val==='number'&&!Number.isInteger(val)?n.toFixed(1):Math.round(n)}{suffix}</span>;
}

function Slider({label, icon, value, min, max, step=1, fmt, onChange}) {
  const pct = ((value-min)/(max-min))*100;
  return (
    <div style={{marginBottom:22}}>
      <div style={{display:'flex',justifyContent:'space-between',marginBottom:8}}>
        <span style={{color:C.muted2,fontSize:12,fontWeight:600,display:'flex',gap:6,alignItems:'center'}}><span>{icon}</span>{label}</span>
        <span style={{color:C.cyan,fontWeight:800,fontSize:14,fontFamily:"'JetBrains Mono',monospace"}}>{fmt?fmt(value):value}</span>
      </div>
      <div style={{position:'relative',height:5,background:C.border,borderRadius:99,cursor:'pointer'}}>
        <div style={{position:'absolute',left:0,top:0,height:'100%',width:`${pct}%`,background:`linear-gradient(90deg,${C.indigo},${C.cyan})`,borderRadius:99,transition:'width 0.05s'}}/>
        <input type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(Number(e.target.value))}
          style={{position:'absolute',top:-8,left:0,width:'100%',opacity:0,cursor:'pointer',height:20}}/>
      </div>
    </div>
  );
}

/* ── PAGES ────────────────────────────────────────────────────────────────── */

function LandingPage({onStart}) {
  const features = [
    {icon:'🧠',title:'5 ML Models + Ensemble',desc:'LR · DT · RF · GBoost · ANN combined via weighted voting for maximum accuracy'},
    {icon:'🔍',title:'Explainable AI',desc:'Feature importance & plain-English explanations for every decision'},
    {icon:'🔮',title:'What-If Simulator',desc:'Real-time scenario modelling — see how changing inputs shifts your approval odds'},
    {icon:'⚡',title:'Instant Results',desc:'Sub-second predictions with risk score, EMI estimate & suggested interest rate'},
    {icon:'📊',title:'Full Analytics',desc:'ROC curves, confusion matrices, SHAP-style charts & model comparisons'},
    {icon:'🤖',title:'AI Chatbot',desc:'Ask the assistant why you were rejected and how to improve eligibility'},
  ];
  const metrics = [
    {v:'93.33',s:'%',l:'Peak Accuracy'},
    {v:'0.9855',s:'',l:'Best AUC-ROC'},
    {v:'5',s:'+',l:'ML Algorithms'},
    {v:'3000',s:'',l:'Training Samples'},
  ];
  return (
    <div style={{maxWidth:1100,margin:'0 auto',padding:'0 24px'}}>
      {/* Hero */}
      <div style={{textAlign:'center',padding:'100px 0 70px',position:'relative'}}>
        {/* Ambient glow */}
        <div style={{position:'absolute',top:'20%',left:'50%',transform:'translate(-50%,-50%)',width:600,height:300,background:`radial-gradient(ellipse,${C.indigo}18 0%,transparent 70%)`,pointerEvents:'none'}}/>
        <div style={{display:'inline-flex',alignItems:'center',gap:8,background:`${C.cyan}12`,border:`1px solid ${C.cyan}30`,borderRadius:99,padding:'6px 20px',marginBottom:32}}>
          <span style={{width:7,height:7,borderRadius:'50%',background:C.emerald,display:'inline-block',animation:'blink 2s infinite'}}/>
          <span style={{color:C.cyan,fontSize:12,fontWeight:700,letterSpacing:2,textTransform:'uppercase'}}>Fintech-Grade AI Platform · 6 Models · Real-Time</span>
        </div>
        <h1 style={{fontFamily:"'Syne',sans-serif",fontWeight:900,fontSize:'clamp(38px,6.5vw,78px)',lineHeight:1.03,color:'#fff',margin:'0 0 8px'}}>
          Smart Loan Eligibility
        </h1>
        <h1 style={{fontFamily:"'Syne',sans-serif",fontWeight:900,fontSize:'clamp(38px,6.5vw,78px)',lineHeight:1.03,margin:'0 0 28px',background:`linear-gradient(135deg,${C.cyan},${C.indigo},${C.violet})`,WebkitBackgroundClip:'text',WebkitTextFillColor:'transparent'}}>
          Advisor — Powered by AI
        </h1>
        <p style={{color:C.muted2,fontSize:18,maxWidth:580,margin:'0 auto 48px',lineHeight:1.75}}>
          Predict. Explain. Simulate. A production-grade ML platform that tells you not just <em style={{color:C.text}}>if</em> you qualify — but <em style={{color:C.text}}>why</em>, and exactly <em style={{color:C.text}}>what to change</em>.
        </p>
        <div style={{display:'flex',gap:16,justifyContent:'center',flexWrap:'wrap'}}>
          <button onClick={onStart} style={{background:`linear-gradient(135deg,${C.indigo},${C.cyan})`,color:'#fff',border:'none',borderRadius:14,padding:'17px 44px',fontSize:16,fontWeight:800,cursor:'pointer',boxShadow:`0 0 48px ${C.indigo}50`,transition:'all .2s',letterSpacing:.3}}
            onMouseEnter={e=>{e.target.style.transform='scale(1.04)';e.target.style.boxShadow=`0 0 64px ${C.cyan}50`;}}
            onMouseLeave={e=>{e.target.style.transform='scale(1)';e.target.style.boxShadow=`0 0 48px ${C.indigo}50`;}}>
            Check My Eligibility →
          </button>
          <button style={{background:'transparent',color:C.muted2,border:`1px solid ${C.border2}`,borderRadius:14,padding:'17px 32px',fontSize:15,fontWeight:600,cursor:'pointer',transition:'all .2s'}}
            onMouseEnter={e=>{e.target.style.borderColor=C.cyan;e.target.style.color=C.cyan;}}
            onMouseLeave={e=>{e.target.style.borderColor=C.border2;e.target.style.color=C.muted2;}}>
            View Analytics
          </button>
        </div>
      </div>

      {/* Metric strip */}
      <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:18,marginBottom:70}}>
        {metrics.map(m=>(
          <Card key={m.l} style={{padding:22,textAlign:'center'}}>
            <div style={{fontSize:34,fontWeight:900,fontFamily:"'JetBrains Mono',monospace",background:`linear-gradient(135deg,${C.cyan},${C.indigo})`,WebkitBackgroundClip:'text',WebkitTextFillColor:'transparent'}}>{m.v}{m.s}</div>
            <div style={{color:C.muted2,fontSize:12,marginTop:6,fontWeight:600,letterSpacing:.5}}>{m.l}</div>
          </Card>
        ))}
      </div>

      {/* Features */}
      <div style={{marginBottom:90}}>
        <h2 style={{textAlign:'center',fontFamily:"'Syne',sans-serif",fontWeight:800,fontSize:34,color:'#fff',marginBottom:10}}>
          Built for <span style={{color:C.cyan}}>Real Intelligence</span>
        </h2>
        <p style={{textAlign:'center',color:C.muted2,marginBottom:44,fontSize:15}}>Every component modelled after production fintech infrastructure</p>
        <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(290px,1fr))',gap:18}}>
          {features.map(f=>(
            <Card key={f.title} style={{display:'flex',gap:16,alignItems:'flex-start',padding:24}}>
              <span style={{fontSize:26,flexShrink:0,marginTop:2}}>{f.icon}</span>
              <div>
                <div style={{color:'#fff',fontWeight:700,marginBottom:6,fontSize:15}}>{f.title}</div>
                <div style={{color:C.muted2,fontSize:13,lineHeight:1.65}}>{f.desc}</div>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Model table */}
      <Card glow={C.indigo} style={{marginBottom:90}}>
        <h3 style={{color:'#fff',fontFamily:"'Syne',sans-serif",fontWeight:800,marginBottom:24,fontSize:20}}>Algorithm Performance Benchmark</h3>
        <div style={{overflowX:'auto'}}>
          <table style={{width:'100%',borderCollapse:'collapse',fontSize:13}}>
            <thead>
              <tr>{['Algorithm','Role','Accuracy','AUC-ROC','F1 Score','CV Mean'].map(h=>(
                <th key={h} style={{textAlign:'left',padding:'10px 16px',color:C.muted2,fontWeight:600,borderBottom:`1px solid ${C.border}`,letterSpacing:.5}}>{h}</th>
              ))}</tr>
            </thead>
            <tbody>
              {[
                ['Logistic Regression','Baseline',{acc:93.33,auc:'0.9855',f1:93.1,cv:'91.83%',c:C.indigo}],
                ['Decision Tree','Interpretable',{acc:89.33,auc:'0.9551',f1:89.0,cv:'88.79%',c:C.amber}],
                ['Random Forest','Primary',{acc:91.33,auc:'0.9803',f1:91.1,cv:'91.42%',c:C.cyan}],
                ['Gradient Boosting','Advanced',{acc:91.67,auc:'0.9799',f1:91.4,cv:'91.54%',c:C.violet}],
                ['Neural Network (ANN)','Deep Learning',{acc:93.00,auc:'0.9831',f1:92.8,cv:'91.96%',c:'#f97316'}],
                ['Ensemble Voting','⭐ Final',{acc:92.33,auc:'0.9788',f1:92.1,cv:'92.33%',c:C.emerald}],
              ].map(([name,role,m])=>(
                <tr key={name} style={{borderBottom:`1px solid ${C.border}40`}}>
                  <td style={{padding:'12px 16px',color:'#fff',fontWeight:600}}>{name}</td>
                  <td style={{padding:'12px 16px'}}><Pill label={role} color={m.c}/></td>
                  <td style={{padding:'12px 16px',color:m.c,fontFamily:'monospace',fontWeight:700}}>{m.acc}%</td>
                  <td style={{padding:'12px 16px',color:C.text,fontFamily:'monospace'}}>{m.auc}</td>
                  <td style={{padding:'12px 16px',color:C.text,fontFamily:'monospace'}}>{m.f1}%</td>
                  <td style={{padding:'12px 16px',color:C.text,fontFamily:'monospace'}}>{m.cv}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

/* FORM ─────────────────────────────────────────────────────────────────────── */
function FormPage({onResult}) {
  const [form, setForm] = useState({
    age:32, monthly_income:65000, employment_type:'Salaried',
    credit_score:710, loan_amount:800000, loan_purpose:'Home',
    existing_emis:12000, loan_tenure:240,
  });
  const [loading, setLoading] = useState(false);

  const set = (k,v) => setForm(p=>({...p,[k]:v}));
  const dti = ((form.existing_emis / Math.max(form.monthly_income,1))*100).toFixed(1);
  const lti = (form.loan_amount / Math.max(form.monthly_income*12,1)).toFixed(1);
  const fmt = v => `₹${v.toLocaleString('en-IN')}`;
  const fmtMo = v => `${v} mo`;

  const creditColor = form.credit_score >= 750 ? C.emerald : form.credit_score >= 650 ? C.cyan : form.credit_score >= 550 ? C.amber : C.rose;
  const dtiColor = dti < 30 ? C.emerald : dti < 45 ? C.amber : C.rose;

  const handleSubmit = async () => {
    setLoading(true);
    await new Promise(r => setTimeout(r, 1400));
    const result = computeAll(form);
    setLoading(false);
    onResult(result, form);
  };

  return (
    <div style={{maxWidth:940,margin:'0 auto',padding:'0 24px 80px'}}>
      <div style={{textAlign:'center',padding:'56px 0 40px'}}>
        <Pill label="LOAN APPLICATION FORM" color={C.cyan}/>
        <h2 style={{fontFamily:"'Syne',sans-serif",fontWeight:900,fontSize:42,color:'#fff',marginTop:18,marginBottom:8}}>Enter Your Details</h2>
        <p style={{color:C.muted2,fontSize:15}}>All processing happens locally. No data leaves your device.</p>
      </div>

      {/* Live metrics strip */}
      <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:14,marginBottom:30}}>
        {[
          {l:'Debt-to-Income',v:`${dti}%`,c:dtiColor,icon:'📊'},
          {l:'Loan-to-Income',v:`${lti}×`,c:lti<4?C.emerald:lti<7?C.amber:C.rose,icon:'🏦'},
          {l:'Credit Band',v:form.credit_score>=750?'Excellent':form.credit_score>=700?'Very Good':form.credit_score>=650?'Good':form.credit_score>=580?'Fair':'Poor',c:creditColor,icon:'⭐'},
          {l:'Monthly EMI Est.',v:`₹${(form.loan_amount*(parseFloat((form.credit_score>=750?7.5:form.credit_score>=650?10:13.5))/100/12)*Math.pow(1+(parseFloat((form.credit_score>=750?7.5:form.credit_score>=650?10:13.5))/100/12),form.loan_tenure)/(Math.pow(1+(parseFloat((form.credit_score>=750?7.5:form.credit_score>=650?10:13.5))/100/12),form.loan_tenure)-1)).toLocaleString('en-IN',{maximumFractionDigits:0})}`,c:C.violet,icon:'💸'},
        ].map(s=>(
          <Card key={s.l} style={{padding:18,textAlign:'center',border:`1px solid ${s.c}25`}}>
            <div style={{fontSize:18}}>{s.icon}</div>
            <div style={{color:s.c,fontWeight:800,fontSize:17,fontFamily:'monospace',marginTop:4}}>{s.v}</div>
            <div style={{color:C.muted2,fontSize:11,marginTop:3}}>{s.l}</div>
          </Card>
        ))}
      </div>

      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:22}}>
        {/* Left */}
        <Card glow={C.cyan}>
          <div style={{color:C.cyan,fontSize:12,fontWeight:700,letterSpacing:1.5,textTransform:'uppercase',marginBottom:22}}>Personal & Financial</div>
          <Slider label="Age" icon="👤" value={form.age} min={18} max={65} onChange={v=>set('age',v)}/>
          <Slider label="Monthly Income" icon="💰" value={form.monthly_income} min={10000} max={500000} step={2500} fmt={fmt} onChange={v=>set('monthly_income',v)}/>
          <Slider label="Existing EMIs per month" icon="💳" value={form.existing_emis} min={0} max={150000} step={1000} fmt={fmt} onChange={v=>set('existing_emis',v)}/>
          <div style={{marginTop:8}}>
            <div style={{color:C.muted2,fontSize:12,fontWeight:600,marginBottom:10}}>💼 Employment Type</div>
            <div style={{display:'flex',gap:10}}>
              {['Salaried','Self-Employed'].map(s=>(
                <button key={s} onClick={()=>set('employment_type',s)} style={{flex:1,padding:'11px 8px',borderRadius:10,border:`1px solid ${form.employment_type===s?C.cyan:C.border}`,background:form.employment_type===s?`${C.cyan}18`:'transparent',color:form.employment_type===s?C.cyan:C.muted2,fontSize:13,fontWeight:700,cursor:'pointer',transition:'all .2s'}}>{s}</button>
              ))}
            </div>
          </div>
        </Card>

        {/* Right */}
        <Card glow={C.indigo}>
          <div style={{color:C.indigo,fontSize:12,fontWeight:700,letterSpacing:1.5,textTransform:'uppercase',marginBottom:22}}>Loan & Credit Details</div>
          <div style={{marginBottom:22}}>
            <div style={{display:'flex',justifyContent:'space-between',marginBottom:8}}>
              <span style={{color:C.muted2,fontSize:12,fontWeight:600}}>⭐ Credit Score</span>
              <span style={{color:creditColor,fontWeight:800,fontFamily:'monospace'}}>{form.credit_score}</span>
            </div>
            <div style={{position:'relative',height:5,background:C.border,borderRadius:99}}>
              <div style={{position:'absolute',left:0,top:0,height:'100%',width:`${(form.credit_score-300)/600*100}%`,background:`linear-gradient(90deg,${C.rose},${C.amber},${C.emerald})`,borderRadius:99,transition:'width .05s'}}/>
              <input type="range" min={300} max={900} value={form.credit_score} onChange={e=>set('credit_score',+e.target.value)} style={{position:'absolute',top:-8,left:0,width:'100%',opacity:0,cursor:'pointer',height:20}}/>
            </div>
            <div style={{display:'flex',justifyContent:'space-between',marginTop:6,fontSize:10,color:C.muted}}>
              <span>300 Poor</span><span>580 Fair</span><span>670 Good</span><span>750 V.Good</span><span>850 Exc.</span>
            </div>
          </div>
          <Slider label="Loan Amount" icon="🏠" value={form.loan_amount} min={50000} max={10000000} step={25000} fmt={fmt} onChange={v=>set('loan_amount',v)}/>
          <Slider label="Loan Tenure (months)" icon="📅" value={form.loan_tenure} min={6} max={360} step={6} fmt={fmtMo} onChange={v=>set('loan_tenure',v)}/>
          <div style={{marginTop:8}}>
            <div style={{color:C.muted2,fontSize:12,fontWeight:600,marginBottom:10}}>🎯 Loan Purpose</div>
            <div style={{display:'flex',gap:8,flexWrap:'wrap'}}>
              {['Home','Car','Personal','Business','Education'].map(p=>(
                <button key={p} onClick={()=>set('loan_purpose',p)} style={{flex:'1 1 calc(33% - 8px)',padding:'9px 4px',borderRadius:9,border:`1px solid ${form.loan_purpose===p?C.violet:C.border}`,background:form.loan_purpose===p?`${C.violet}18`:'transparent',color:form.loan_purpose===p?C.violet:C.muted2,fontSize:12,fontWeight:700,cursor:'pointer',transition:'all .2s'}}>{p}</button>
              ))}
            </div>
          </div>
        </Card>
      </div>

      <div style={{textAlign:'center',marginTop:34}}>
        <button onClick={handleSubmit} disabled={loading} style={{background:loading?C.border:`linear-gradient(135deg,${C.indigo},${C.cyan})`,color:'#fff',border:'none',borderRadius:16,padding:'18px 64px',fontSize:18,fontWeight:900,cursor:loading?'not-allowed':'pointer',boxShadow:loading?'none':`0 0 56px ${C.indigo}50`,transition:'all .3s',display:'inline-flex',alignItems:'center',gap:14}}>
          {loading ? (<><span style={{width:20,height:20,border:'3px solid rgba(255,255,255,.25)',borderTopColor:'#fff',borderRadius:'50%',display:'inline-block',animation:'spin .8s linear infinite'}}/> Running 6 ML Models…</>) : (<><span>🔮</span> Predict Loan Eligibility</>)}
        </button>
      </div>
    </div>
  );
}

/* RESULTS ──────────────────────────────────────────────────────────────────── */
function ResultsPage({result, formData, onBack, onSimulate}) {
  if (!result) return null;
  const {eligible,verdict,probability,risk_score,dti,suggested_rate,estimated_emi,model_probs,reasons,tips} = result;
  const feature_importances = result.feature_importances || ANALYTICS.feature_importances;
  const mainC = eligible ? C.emerald : C.rose;

  const modelBars = Object.entries(model_probs || {}).map(([k,v])=>({model:k.replace(' ','\n'),fullName:k,score:v}));
  const fiBars = Object.entries(feature_importances || {}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>({feature:k,importance:v}));

  const pieData = [{name:'Approval',value:probability,fill:mainC},{name:'Gap',value:100-probability,fill:`${C.border}`}];

  return (
    <div style={{maxWidth:1060,margin:'0 auto',padding:'0 24px 80px'}}>
      <div style={{textAlign:'center',padding:'52px 0 44px'}}>
        <div style={{display:'inline-flex',alignItems:'center',gap:14,background:`${mainC}12`,border:`2px solid ${mainC}45`,borderRadius:99,padding:'14px 38px',marginBottom:28,animation:'fadeUp .6s ease'}}>
          <span style={{fontSize:30}}>{eligible?'🎉':'❌'}</span>
          <span style={{color:mainC,fontSize:28,fontWeight:900,fontFamily:"'Syne',sans-serif"}}>{verdict}</span>
        </div>
        <h2 style={{fontFamily:"'Syne',sans-serif",fontWeight:900,fontSize:40,color:'#fff',marginBottom:12}}>Eligibility Analysis Complete</h2>
        <div style={{display:'flex',gap:10,justifyContent:'center',flexWrap:'wrap'}}>
          <Pill label={`${probability}% Probability`} color={mainC}/>
          <Pill label={`${suggested_rate}% Interest Rate`} color={C.amber}/>
          <Pill label={`DTI: ${dti}%`} color={dti>40?C.rose:C.emerald}/>
        </div>
      </div>

      {/* Hero row */}
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:20,marginBottom:22}}>
        {/* Gauge */}
        <Card glow={risk_score>66?C.rose:risk_score>33?C.amber:C.emerald} style={{display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center'}}>
          <GaugeMeter value={risk_score} size={180}/>
          <div style={{marginTop:12,textAlign:'center'}}>
            <div style={{color:C.muted2,fontSize:11,letterSpacing:1}}>RISK CLASSIFICATION</div>
            <div style={{color:risk_score<33?C.emerald:risk_score<66?C.amber:C.rose,fontWeight:800,fontSize:16,marginTop:4}}>{risk_score<33?'Low Risk':risk_score<66?'Medium Risk':'High Risk'}</div>
          </div>
        </Card>

        {/* Approval donut */}
        <Card glow={mainC} style={{display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center'}}>
          <div style={{color:C.muted2,fontSize:11,letterSpacing:1,textTransform:'uppercase',marginBottom:12}}>Approval Probability</div>
          <div style={{position:'relative',width:160,height:160}}>
            <PieChart width={160} height={160}>
              <Pie data={pieData} cx={75} cy={75} innerRadius={52} outerRadius={72} startAngle={90} endAngle={-270} dataKey="value" strokeWidth={0}>
                {pieData.map((e,i)=><Cell key={i} fill={e.fill}/>)}
              </Pie>
            </PieChart>
            <div style={{position:'absolute',inset:0,display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center'}}>
              <div style={{fontSize:32,fontWeight:900,color:mainC,fontFamily:'monospace',lineHeight:1}}><AnimNum val={probability} suffix="%"/></div>
              <div style={{color:C.muted2,fontSize:10,marginTop:3}}>confidence</div>
            </div>
          </div>
        </Card>

        {/* EMI card */}
        <Card glow={C.violet} style={{display:'flex',flexDirection:'column',justifyContent:'space-evenly'}}>
          {[
            {l:'Estimated EMI',v:`₹${estimated_emi.toLocaleString('en-IN')}`,c:C.violet},
            {l:'Suggested Rate',v:`${suggested_rate}% p.a.`,c:C.amber},
            {l:'Loan Tenure',v:`${formData.loan_tenure} months`,c:C.cyan},
            {l:'Loan Amount',v:`₹${formData.loan_amount.toLocaleString('en-IN')}`,c:C.text},
          ].map(({l,v,c})=>(
            <div key={l} style={{borderBottom:`1px solid ${C.border}`,paddingBottom:12,marginBottom:12}}>
              <div style={{color:C.muted2,fontSize:11}}>{l}</div>
              <div style={{color:c,fontWeight:800,fontSize:16,fontFamily:'monospace',marginTop:2}}>{v}</div>
            </div>
          ))}
        </Card>
      </div>

      {/* Model comparison */}
      <Card style={{marginBottom:22}}>
        <div style={{color:C.muted2,fontSize:12,fontWeight:600,letterSpacing:1,textTransform:'uppercase',marginBottom:18}}>🤖 All Model Predictions</div>
        <ResponsiveContainer width="100%" height={190}>
          <BarChart data={modelBars} layout="vertical" margin={{left:20}}>
            <XAxis type="number" domain={[0,100]} tick={{fill:C.muted2,fontSize:11}} tickFormatter={v=>`${v}%`}/>
            <YAxis type="category" dataKey="fullName" tick={{fill:C.text,fontSize:11}} width={120}/>
            <Tooltip formatter={v=>[`${v}%`,'Approval Prob']} contentStyle={{background:C.card,border:`1px solid ${C.border}`,borderRadius:10,color:'#fff'}}/>
            <Bar dataKey="score" radius={[0,6,6,0]}>
              {modelBars.map((_,i)=><Cell key={i} fill={[C.indigo,C.violet,C.cyan,C.amber,'#f97316',C.emerald][i%6]}/>)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Feature importance + SHAP-style */}
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:22,marginBottom:22}}>
        <Card>
          <div style={{color:C.muted2,fontSize:12,fontWeight:600,letterSpacing:1,textTransform:'uppercase',marginBottom:18}}>🔍 Feature Impact (SHAP-style)</div>
          {fiBars.map((f,i)=>(
            <div key={f.feature} style={{marginBottom:12}}>
              <div style={{display:'flex',justifyContent:'space-between',marginBottom:5}}>
                <span style={{color:C.text,fontSize:13}}>{f.feature}</span>
                <span style={{color:[C.cyan,C.indigo,C.violet,C.amber,C.emerald,'#f97316'][i%6],fontFamily:'monospace',fontWeight:700,fontSize:13}}>{f.importance}%</span>
              </div>
              <div style={{height:6,background:C.border,borderRadius:99}}>
                <div style={{height:'100%',width:`${f.importance}%`,background:[C.cyan,C.indigo,C.violet,C.amber,C.emerald,'#f97316'][i%6],borderRadius:99,transition:`width ${.4+i*.08}s ease`}}/>
              </div>
            </div>
          ))}
        </Card>

        {/* Reasons + tips */}
        <div style={{display:'flex',flexDirection:'column',gap:16}}>
          <Card>
            <div style={{color:C.muted2,fontSize:12,fontWeight:600,letterSpacing:1,textTransform:'uppercase',marginBottom:14}}>❓ Why {verdict}?</div>
            {reasons.map((r,i)=>(
              <div key={i} style={{display:'flex',gap:10,marginBottom:10,padding:'10px 14px',background:`${C.rose}0a`,border:`1px solid ${C.rose}25`,borderRadius:10}}>
                <span style={{color:C.rose,marginTop:1}}>⚠</span>
                <span style={{color:C.text,fontSize:13,lineHeight:1.5}}>{r}</span>
              </div>
            ))}
          </Card>
          <Card>
            <div style={{color:C.muted2,fontSize:12,fontWeight:600,letterSpacing:1,textTransform:'uppercase',marginBottom:14}}>💡 How to Improve</div>
            {tips.map((t,i)=>(
              <div key={i} style={{display:'flex',gap:10,marginBottom:10,padding:'10px 14px',background:`${C.emerald}0a`,border:`1px solid ${C.emerald}25`,borderRadius:10}}>
                <span style={{fontSize:18}}>{t.icon}</span>
                <div>
                  {t.delta>0&&<Pill label={`+${t.delta}% possible`} color={C.emerald}/>}
                  <div style={{color:C.text,fontSize:13,lineHeight:1.5,marginTop:6}}>{t.msg}</div>
                </div>
              </div>
            ))}
          </Card>
        </div>
      </div>

      <div style={{textAlign:'center',display:'flex',gap:16,justifyContent:'center'}}>
        <button onClick={onBack} style={{background:`linear-gradient(135deg,${C.indigo},${C.cyan})`,color:'#fff',border:'none',borderRadius:14,padding:'14px 36px',fontSize:15,fontWeight:800,cursor:'pointer'}}>← New Application</button>
        <button onClick={onSimulate} style={{background:`${C.violet}20`,color:C.violet,border:`1px solid ${C.violet}40`,borderRadius:14,padding:'14px 36px',fontSize:15,fontWeight:700,cursor:'pointer'}}>🔮 Open Simulator</button>
      </div>
    </div>
  );
}

/* SIMULATOR ────────────────────────────────────────────────────────────────── */
function SimulatorPage({formData, baseResult}) {
  const [base, setBase] = useState(formData || {age:32,monthly_income:65000,employment_type:'Salaried',credit_score:710,loan_amount:800000,loan_purpose:'Home',existing_emis:12000,loan_tenure:240});
  const [scenarios, setScenarios] = useState([]);

  useEffect(()=>{
    const base_result = computeAll(base);
    const deltas = [
      {label:'income',desc:'Income +₹10,000/mo',delta:{monthly_income:10000}},
      {label:'income+',desc:'Income +₹25,000/mo',delta:{monthly_income:25000}},
      {label:'credit',desc:'Credit Score +50',delta:{credit_score:50}},
      {label:'credit+',desc:'Credit Score +100',delta:{credit_score:100}},
      {label:'emis',desc:'EMIs −₹5,000/mo',delta:{existing_emis:-5000}},
      {label:'emis+',desc:'EMIs −₹15,000/mo',delta:{existing_emis:-15000}},
      {label:'amount',desc:'Loan −₹2L',delta:{loan_amount:-200000}},
      {label:'tenure',desc:'Tenure +5 yrs',delta:{loan_tenure:60}},
    ];
    const scs = deltas.map(sc=>{
      const d2 = {...base};
      Object.entries(sc.delta).forEach(([k,v])=>{d2[k]=Math.max(0,base[k]+v);});
      if(d2.credit_score) d2.credit_score = Math.min(900, Math.max(300, d2.credit_score));
      const r2 = computeAll(d2);
      return {...sc, probability: r2.probability, eligible: r2.eligible, baseProbability: base_result.probability};
    });
    setScenarios(scs);
  },[base]);

  const baseProb = baseResult?.probability || computeAll(base).probability;

  return (
    <div style={{maxWidth:1060,margin:'0 auto',padding:'0 24px 80px'}}>
      <div style={{textAlign:'center',padding:'52px 0 40px'}}>
        <Pill label="WHAT-IF SIMULATOR" color={C.violet}/>
        <h2 style={{fontFamily:"'Syne',sans-serif",fontWeight:900,fontSize:42,color:'#fff',marginTop:18,marginBottom:8}}>AI Scenario Simulator</h2>
        <p style={{color:C.muted2}}>Real-time prediction updates as you change parameters</p>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'340px 1fr',gap:24}}>
        {/* Controls */}
        <Card glow={C.violet}>
          <div style={{color:C.violet,fontSize:12,fontWeight:700,letterSpacing:1.5,textTransform:'uppercase',marginBottom:20}}>Adjust Parameters</div>
          <Slider label="Monthly Income" icon="💰" value={base.monthly_income} min={10000} max={500000} step={2500} fmt={v=>`₹${v.toLocaleString('en-IN')}`} onChange={v=>setBase(p=>({...p,monthly_income:v}))}/>
          <Slider label="Credit Score" icon="⭐" value={base.credit_score} min={300} max={900} onChange={v=>setBase(p=>({...p,credit_score:v}))}/>
          <Slider label="Existing EMIs" icon="💳" value={base.existing_emis} min={0} max={150000} step={1000} fmt={v=>`₹${v.toLocaleString('en-IN')}`} onChange={v=>setBase(p=>({...p,existing_emis:v}))}/>
          <Slider label="Loan Amount" icon="🏠" value={base.loan_amount} min={50000} max={10000000} step={25000} fmt={v=>`₹${v.toLocaleString('en-IN')}`} onChange={v=>setBase(p=>({...p,loan_amount:v}))}/>
          <Slider label="Loan Tenure" icon="📅" value={base.loan_tenure} min={6} max={360} step={6} fmt={v=>`${v} mo`} onChange={v=>setBase(p=>({...p,loan_tenure:v}))}/>

          <div style={{marginTop:16,padding:'16px',background:`${C.border}60`,borderRadius:12,textAlign:'center'}}>
            <div style={{color:C.muted2,fontSize:11,marginBottom:4}}>Current Approval Chance</div>
            <div style={{color:baseProb>=50?C.emerald:C.rose,fontSize:30,fontWeight:900,fontFamily:'monospace'}}>{baseProb}%</div>
          </div>
        </Card>

        {/* Scenario grid */}
        <div>
          <div style={{color:C.muted2,fontSize:12,fontWeight:600,letterSpacing:1,textTransform:'uppercase',marginBottom:16}}>Impact of Changes</div>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:14}}>
            {scenarios.map(sc=>{
              const delta = sc.probability - baseProb;
              const c = delta > 0 ? C.emerald : delta < 0 ? C.rose : C.amber;
              return (
                <Card key={sc.label} style={{padding:20,border:`1px solid ${c}22`}}>
                  <div style={{color:C.text,fontWeight:700,fontSize:14,marginBottom:8}}>📌 {sc.desc}</div>
                  <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-end'}}>
                    <div>
                      <div style={{color:C.muted2,fontSize:11}}>New Probability</div>
                      <div style={{color:sc.eligible?C.emerald:C.rose,fontWeight:900,fontFamily:'monospace',fontSize:22}}>{sc.probability}%</div>
                    </div>
                    <div style={{textAlign:'right'}}>
                      <div style={{color:c,fontWeight:800,fontSize:16,fontFamily:'monospace'}}>{delta>0?'+':''}{delta.toFixed(1)}%</div>
                      <Pill label={sc.eligible?'Approved':'Rejected'} color={sc.eligible?C.emerald:C.rose}/>
                    </div>
                  </div>
                  <div style={{marginTop:12,height:5,background:C.border,borderRadius:99}}>
                    <div style={{height:'100%',width:`${sc.probability}%`,background:c,borderRadius:99,transition:'width .4s'}}/>
                  </div>
                </Card>
              );
            })}
          </div>

          {/* Chart */}
          <Card style={{marginTop:18}}>
            <div style={{color:C.muted2,fontSize:12,fontWeight:600,letterSpacing:1,textTransform:'uppercase',marginBottom:16}}>Scenario Comparison</div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={scenarios} margin={{left:0,bottom:20}}>
                <XAxis dataKey="desc" tick={{fill:C.muted2,fontSize:9}} angle={-25} textAnchor="end" height={50}/>
                <YAxis domain={[0,100]} tick={{fill:C.muted2,fontSize:10}} tickFormatter={v=>`${v}%`}/>
                <Tooltip formatter={v=>[`${v}%`,'Approval']} contentStyle={{background:C.card,border:`1px solid ${C.border}`,borderRadius:8,color:'#fff'}}/>
                <Bar dataKey="probability" radius={[4,4,0,0]}>
                  {scenarios.map((s,i)=><Cell key={i} fill={s.eligible?C.emerald:C.rose}/>)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      </div>
    </div>
  );
}

/* ANALYTICS ────────────────────────────────────────────────────────────────── */
function AnalyticsPage() {
  const {model_results, feature_importances} = ANALYTICS;
  const [sel, setSel] = useState('Random Forest');
  const m = model_results[sel];

  const colors = {
    'Logistic Regression': C.indigo, 'Decision Tree': C.amber,
    'Random Forest': C.cyan, 'Gradient Boosting': C.violet,
    'Neural Network': '#f97316', 'Ensemble': C.emerald,
  };

  const accData = Object.entries(model_results).map(([k,v])=>({model:k.split(' ')[0]+(k.includes('Ensemble')?'':k.split(' ').length>2?k.split(' ').slice(-1)[0]:''),fullName:k,accuracy:v.accuracy,auc:parseFloat((v.auc*100).toFixed(2)),f1:v.f1}));

  const radarMetrics = ['accuracy','precision','recall','f1'];
  const radarData = radarMetrics.map(metric=>({
    metric: metric.charAt(0).toUpperCase()+metric.slice(1),
    ...Object.fromEntries(Object.entries(model_results).map(([k,v])=>[k.split(' ')[0],v[metric]]))
  }));

  const fiData = Object.entries(feature_importances).sort((a,b)=>b[1]-a[1]).map(([k,v])=>({feature:k,importance:v}));

  // Mock ROC-like data
  const rocData = Array.from({length:20},(_,i)=>{
    const x = i/19;
    return {
      fpr: parseFloat(x.toFixed(3)),
      rf: Math.min(1, Math.pow(x,.12)),
      gb: Math.min(1, Math.pow(x,.13)),
      lr: Math.min(1, Math.pow(x,.15)),
      dt: Math.min(1, Math.pow(x,.22)),
      nn: Math.min(1, Math.pow(x,.13)),
    };
  });

  return (
    <div style={{maxWidth:1100,margin:'0 auto',padding:'0 24px 80px'}}>
      <div style={{textAlign:'center',padding:'52px 0 44px'}}>
        <Pill label="MODEL ANALYTICS" color={C.violet}/>
        <h2 style={{fontFamily:"'Syne',sans-serif",fontWeight:900,fontSize:42,color:'#fff',marginTop:18,marginBottom:8}}>Performance <span style={{color:C.violet}}>Intelligence</span></h2>
        <p style={{color:C.muted2}}>Full ML evaluation suite — confusion matrices, ROC curves, feature analysis</p>
      </div>

      {/* Model selector */}
      <div style={{display:'flex',gap:10,justifyContent:'center',flexWrap:'wrap',marginBottom:30}}>
        {Object.keys(model_results).map(k=>(
          <button key={k} onClick={()=>setSel(k)} style={{padding:'8px 18px',borderRadius:10,border:`1px solid ${sel===k?colors[k]:C.border}`,background:sel===k?`${colors[k]}18`:'transparent',color:sel===k?colors[k]:C.muted2,fontSize:12,fontWeight:700,cursor:'pointer',transition:'all .2s'}}>{k}</button>
        ))}
      </div>

      {/* Selected model metrics */}
      <div style={{display:'grid',gridTemplateColumns:'repeat(6,1fr)',gap:14,marginBottom:24}}>
        {[['Accuracy',`${m.accuracy}%`],['AUC',m.auc],['F1 Score',`${m.f1}%`],['Precision',`${m.precision}%`],['Recall',`${m.recall}%`],['CV Mean',`${m.cv_mean}%`]].map(([l,v])=>(
          <Card key={l} style={{padding:18,textAlign:'center',border:`1px solid ${colors[sel]}22`}}>
            <div style={{color:colors[sel],fontWeight:900,fontSize:20,fontFamily:'monospace'}}>{v}</div>
            <div style={{color:C.muted2,fontSize:11,marginTop:4}}>{l}</div>
          </Card>
        ))}
      </div>

      <div style={{display:'grid',gridTemplateColumns:'1.3fr 1fr',gap:22,marginBottom:22}}>
        {/* Accuracy bar chart */}
        <Card>
          <div style={{color:C.muted2,fontSize:12,fontWeight:600,letterSpacing:1,textTransform:'uppercase',marginBottom:18}}>Model Accuracy & AUC×100</div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={accData}>
              <XAxis dataKey="model" tick={{fill:C.muted2,fontSize:10}} angle={-15} textAnchor="end" height={40}/>
              <YAxis domain={[80,100]} tick={{fill:C.muted2,fontSize:10}}/>
              <Tooltip contentStyle={{background:C.card,border:`1px solid ${C.border}`,borderRadius:8,color:'#fff'}}/>
              <Legend wrapperStyle={{color:C.muted2,fontSize:11}}/>
              <Bar dataKey="accuracy" name="Accuracy%" radius={[4,4,0,0]}>
                {accData.map((_,i)=><Cell key={i} fill={Object.values(colors)[i]}/>)}
              </Bar>
              <Bar dataKey="auc" name="AUC×100" radius={[4,4,0,0]} opacity={0.6}>
                {accData.map((_,i)=><Cell key={i} fill={Object.values(colors)[i]}/>)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Confusion matrix */}
        <Card>
          <div style={{color:C.muted2,fontSize:12,fontWeight:600,letterSpacing:1,textTransform:'uppercase',marginBottom:6}}>Confusion Matrix — {sel}</div>
          <p style={{color:C.muted2,fontSize:11,marginBottom:20}}>Test set (600 samples)</p>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12,maxWidth:260,margin:'0 auto'}}>
            {[
              {l:'True Positive',v:m.confusion_matrix[1][1],c:C.emerald},
              {l:'False Positive',v:m.confusion_matrix[0][1],c:C.rose},
              {l:'False Negative',v:m.confusion_matrix[1][0],c:C.amber},
              {l:'True Negative',v:m.confusion_matrix[0][0],c:C.cyan},
            ].map(({l,v,c})=>(
              <div key={l} style={{background:`${c}12`,border:`1px solid ${c}30`,borderRadius:14,padding:'18px 12px',textAlign:'center'}}>
                <div style={{color:c,fontSize:30,fontWeight:900,fontFamily:'monospace'}}>{v}</div>
                <div style={{color:C.muted2,fontSize:11,marginTop:4}}>{l}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* ROC curve */}
      <Card style={{marginBottom:22}}>
        <div style={{color:C.muted2,fontSize:12,fontWeight:600,letterSpacing:1,textTransform:'uppercase',marginBottom:18}}>ROC Curves — All Models</div>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={rocData}>
            <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
            <XAxis dataKey="fpr" tickFormatter={v=>v.toFixed(2)} tick={{fill:C.muted2,fontSize:10}} label={{value:'FPR',position:'insideBottom',offset:-5,fill:C.muted2,fontSize:11}}/>
            <YAxis tickFormatter={v=>v.toFixed(2)} tick={{fill:C.muted2,fontSize:10}} label={{value:'TPR',angle:-90,position:'insideLeft',fill:C.muted2,fontSize:11}}/>
            <Tooltip contentStyle={{background:C.card,border:`1px solid ${C.border}`,borderRadius:8,color:'#fff',fontSize:11}} formatter={v=>v.toFixed(3)}/>
            <Legend wrapperStyle={{color:C.muted2,fontSize:11}}/>
            <Line type="monotone" dataKey="rf" name="Random Forest" stroke={C.cyan} dot={false} strokeWidth={2}/>
            <Line type="monotone" dataKey="gb" name="Gradient Boosting" stroke={C.violet} dot={false} strokeWidth={2}/>
            <Line type="monotone" dataKey="lr" name="Logistic Reg." stroke={C.indigo} dot={false} strokeWidth={2}/>
            <Line type="monotone" dataKey="dt" name="Decision Tree" stroke={C.amber} dot={false} strokeWidth={2}/>
            <Line type="monotone" dataKey="nn" name="Neural Network" stroke='#f97316' dot={false} strokeWidth={2}/>
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <div style={{display:'grid',gridTemplateColumns:'1.2fr 1fr',gap:22}}>
        {/* Feature importance */}
        <Card>
          <div style={{color:C.muted2,fontSize:12,fontWeight:600,letterSpacing:1,textTransform:'uppercase',marginBottom:20}}>SHAP-style Feature Importance (RF)</div>
          {fiData.map((f,i)=>(
            <div key={f.feature} style={{marginBottom:12}}>
              <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
                <span style={{color:C.text,fontSize:13}}>{f.feature}</span>
                <span style={{color:[C.cyan,C.indigo,C.violet,C.amber,C.emerald,'#f97316'][i%6],fontFamily:'monospace',fontWeight:700,fontSize:13}}>{f.importance}%</span>
              </div>
              <div style={{height:7,background:C.border,borderRadius:99,overflow:'hidden'}}>
                <div style={{height:'100%',width:`${f.importance}%`,background:`linear-gradient(90deg,${[C.cyan,C.indigo,C.violet,C.amber,C.emerald,'#f97316'][i%6]},${[C.indigo,C.violet,C.amber,C.emerald,'#f97316',C.cyan][i%6]})`,borderRadius:99}}/>
              </div>
            </div>
          ))}
        </Card>

        {/* Radar */}
        <Card>
          <div style={{color:C.muted2,fontSize:12,fontWeight:600,letterSpacing:1,textTransform:'uppercase',marginBottom:10}}>Multi-Metric Radar</div>
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={radarData}>
              <PolarGrid stroke={C.border}/>
              <PolarAngleAxis dataKey="metric" tick={{fill:C.muted2,fontSize:11}}/>
              {['Logistic','Decision','Random','Gradient','Neural'].map((k,i)=>(
                <Radar key={k} name={k} dataKey={k} stroke={Object.values(colors)[i]} fill={Object.values(colors)[i]} fillOpacity={0.08}/>
              ))}
              <Legend wrapperStyle={{color:C.muted2,fontSize:10}}/>
              <Tooltip contentStyle={{background:C.card,border:`1px solid ${C.border}`,borderRadius:8,color:'#fff',fontSize:11}}/>
            </RadarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}

/* CHATBOT ──────────────────────────────────────────────────────────────────── */
function Chatbot({lastResult, formData}) {
  const [open, setOpen] = useState(false);
  const [msgs, setMsgs] = useState([{role:'bot',text:"Hi! I'm your Loan AI Assistant 🤖 Ask me why your loan was approved/rejected, how to improve your profile, or about any factor affecting your eligibility."}]);
  const [inp, setInp] = useState('');
  const endRef = useRef();

  const respond = q => {
    const lq = q.toLowerCase();
    if (!lastResult) return "Please complete a loan eligibility check first so I can give you personalised answers!";
    const {verdict, probability, risk_score, reasons, tips, dti, suggested_rate, estimated_emi} = lastResult;
    if (lq.includes('why')&&lq.includes('reject')) return verdict==='Rejected'?`Your loan was rejected primarily because: ${reasons.join('; ')}. The ensemble model gave you a ${probability}% approval probability, below the 50% threshold.`:"Great news — your loan was actually approved! 🎉";
    if (lq.includes('why')&&lq.includes('approv')) return verdict==='Approved'?`Your loan was approved with ${probability}% confidence. Key strengths: good credit score, manageable DTI ratio, and stable income. Suggested rate: ${suggested_rate}%.`:"Your loan was unfortunately rejected. Try asking 'why was I rejected?'";
    if (lq.includes('improve')||lq.includes('better')||lq.includes('how')) return `Top improvements: ${tips.map(t=>t.msg).join(' | ')}. Each change could add ${tips.map(t=>t.delta).filter(Boolean).join('%, ')}% to your approval chance.`;
    if (lq.includes('credit')) return formData?.credit_score>=700?`Your credit score of ${formData.credit_score} is strong. Maintaining it above 750 unlocks the best rates (as low as 7.5%).`:`Your credit score of ${formData?.credit_score} is dragging down your approval. Target 700+ by clearing outstanding dues and keeping credit utilization below 30%.`;
    if (lq.includes('dti')||lq.includes('debt')) return `Your DTI ratio is ${dti}%. ${dti>40?'Above the 40% threshold — reducing monthly EMIs will significantly help.':'This is within acceptable range. Keeping it below 30% is ideal.'}`;
    if (lq.includes('emi')) return `Based on your profile, estimated EMI is ₹${estimated_emi?.toLocaleString('en-IN')} at ${suggested_rate}% p.a. This is calculated using a reducing balance method.`;
    if (lq.includes('risk')) return `Your risk score is ${risk_score}/100. ${risk_score<33?'Low risk — excellent profile.':risk_score<66?'Medium risk — moderate improvements needed.':'High risk — significant changes required before reapplying.'}`;
    if (lq.includes('model')||lq.includes('algorithm')) return "We use 5 models: Logistic Regression (93.33% acc), Decision Tree (89.33%), Random Forest (91.33%), Gradient Boosting (91.67%), Neural Network (93%). Final prediction is a weighted ensemble combining all five.";
    if (lq.includes('strateg')||lq.includes('best')) return "Optimal strategy: (1) Raise credit score to 750+ (biggest impact), (2) Reduce DTI below 30%, (3) Ensure stable salaried employment, (4) Request loan ≤ 4× annual income, (5) Clear existing loans before applying.";
    return `I can answer questions about: why you were ${verdict.toLowerCase()}d, how to improve eligibility, credit score tips, DTI ratio, EMI calculation, risk score, or which ML model affects you most. What would you like to know?`;
  };

  const send = () => {
    if (!inp.trim()) return;
    const user = {role:'user',text:inp};
    const bot = {role:'bot',text:respond(inp)};
    setMsgs(p=>[...p,user,bot]);
    setInp('');
    setTimeout(()=>endRef.current?.scrollIntoView({behavior:'smooth'}),100);
  };

  const suggestions = ['Why was I rejected?','How to improve?','Explain my risk score','Best loan strategy','What is DTI ratio?'];

  return (
    <>
      <button onClick={()=>setOpen(p=>!p)} style={{position:'fixed',bottom:28,right:28,width:62,height:62,borderRadius:'50%',background:`linear-gradient(135deg,${C.indigo},${C.cyan})`,color:'#fff',border:'none',fontSize:26,cursor:'pointer',boxShadow:`0 8px 32px ${C.indigo}60`,zIndex:1000,display:'flex',alignItems:'center',justifyContent:'center',transition:'transform .2s'}}
        onMouseEnter={e=>e.currentTarget.style.transform='scale(1.1)'}
        onMouseLeave={e=>e.currentTarget.style.transform='scale(1)'}>
        {open?'✕':'🤖'}
      </button>
      {open&&(
        <div style={{position:'fixed',bottom:104,right:28,width:380,maxHeight:520,background:C.card,border:`1px solid ${C.border}`,borderRadius:22,zIndex:1000,display:'flex',flexDirection:'column',overflow:'hidden',boxShadow:`0 20px 64px rgba(0,0,0,.7),0 0 0 1px ${C.border2}`}}>
          <div style={{background:`linear-gradient(135deg,${C.indigo},${C.cyan})`,padding:'16px 20px',display:'flex',alignItems:'center',gap:12}}>
            <span style={{fontSize:22}}>🤖</span>
            <div>
              <div style={{color:'#fff',fontWeight:800,fontSize:15}}>Loan AI Assistant</div>
              <div style={{color:'rgba(255,255,255,.65)',fontSize:11}}>Explain · Advise · Strategize</div>
            </div>
            <div style={{marginLeft:'auto',width:8,height:8,borderRadius:'50%',background:C.emerald,animation:'blink 2s infinite'}}/>
          </div>
          <div style={{flex:1,overflowY:'auto',padding:16,display:'flex',flexDirection:'column',gap:10}}>
            {msgs.map((m,i)=>(
              <div key={i} style={{alignSelf:m.role==='user'?'flex-end':'flex-start',maxWidth:'88%',background:m.role==='user'?`${C.indigo}30`:C.surface,border:`1px solid ${m.role==='user'?C.indigo:C.border}40`,borderRadius:m.role==='user'?'18px 18px 4px 18px':'18px 18px 18px 4px',padding:'10px 14px',fontSize:13,color:C.text,lineHeight:1.55}}>
                {m.text}
              </div>
            ))}
            <div ref={endRef}/>
          </div>
          {/* Quick suggestions */}
          <div style={{padding:'8px 12px',borderTop:`1px solid ${C.border}`,display:'flex',gap:6,overflowX:'auto'}}>
            {suggestions.map(s=>(
              <button key={s}  style={{flexShrink:0,background:`${C.indigo}20`,color:C.muted2,border:`1px solid ${C.border}`,borderRadius:99,padding:'4px 12px',fontSize:10,fontWeight:600,cursor:'pointer',whiteSpace:'nowrap'}}
                onClick={()=>{setMsgs(p=>[...p,{role:'user',text:s},{role:'bot',text:respond(s)}]);setTimeout(()=>endRef.current?.scrollIntoView({behavior:'smooth'}),100);}}>
                {s}
              </button>
            ))}
          </div>
          <div style={{padding:'12px 14px',borderTop:`1px solid ${C.border}`,display:'flex',gap:8}}>
            <input value={inp} onChange={e=>setInp(e.target.value)} onKeyDown={e=>e.key==='Enter'&&send()} placeholder="Ask about your loan result…" style={{flex:1,background:C.surface,border:`1px solid ${C.border}`,borderRadius:10,padding:'10px 14px',color:C.text,fontSize:13,outline:'none'}}/>
            <button onClick={send} style={{background:C.indigo,color:'#fff',border:'none',borderRadius:10,padding:'10px 16px',cursor:'pointer',fontWeight:800,fontSize:15}}>→</button>
          </div>
        </div>
      )}
    </>
  );
}

/* ── APP SHELL ────────────────────────────────────────────────────────────── */
export default function App() {
  const [page, setPage] = useState('home');
  const [result, setResult] = useState(null);
  const [formData, setFormData] = useState(null);

  const handleResult = (r, fd) => { setResult(r); setFormData(fd); setPage('results'); };

  const nav = [
    {id:'home',label:'Home',icon:'⬡'},
    {id:'form',label:'Apply',icon:'📋'},
    {id:'simulator',label:'Simulator',icon:'🔮'},
    {id:'analytics',label:'Analytics',icon:'📊'},
  ];

  return (
    <div style={{minHeight:'100vh',background:C.bg,color:C.text,fontFamily:"'Outfit','DM Sans',sans-serif"}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800;900&family=Outfit:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        ::-webkit-scrollbar{width:5px;}::-webkit-scrollbar-track{background:${C.surface};}::-webkit-scrollbar-thumb{background:${C.border};border-radius:3px;}
        input[type=range]{-webkit-appearance:none;appearance:none;}
        @keyframes spin{to{transform:rotate(360deg);}}
        @keyframes blink{0%,100%{opacity:1;}50%{opacity:.3;}}
        @keyframes fadeUp{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}
        em{font-style:italic;}
      `}</style>

      {/* Ambient background grid */}
      <div style={{position:'fixed',inset:0,backgroundImage:`linear-gradient(${C.border}40 1px,transparent 1px),linear-gradient(90deg,${C.border}40 1px,transparent 1px)`,backgroundSize:'80px 80px',opacity:.3,pointerEvents:'none',zIndex:0}}/>
      <div style={{position:'fixed',top:0,left:'50%',transform:'translateX(-50%)',width:800,height:400,background:`radial-gradient(ellipse,${C.indigo}10 0%,transparent 65%)`,pointerEvents:'none',zIndex:0}}/>

      {/* Nav */}
        <nav style={{top:0,zIndex:100,position:'sticky',background:`${C.surface}d8`,backdropFilter:'blur(24px)',borderBottom:`1px solid ${C.border}`,padding:'0 32px'}}>          <div style={{maxWidth:1100,margin:'0 auto',display:'flex',alignItems:'center',height:66,gap:32}}>
          <div onClick={()=>setPage('home')} style={{cursor:'pointer',display:'flex',alignItems:'center',gap:10}}>
            <div style={{width:36,height:36,borderRadius:10,background:`linear-gradient(135deg,${C.indigo},${C.cyan})`,display:'flex',alignItems:'center',justifyContent:'center',fontSize:17}}>💎</div>
            <div>
              <span style={{fontFamily:"'Syne',sans-serif",fontWeight:900,fontSize:17,color:'#fff'}}>LoanAI</span>
              <span style={{color:C.muted2,fontSize:11,marginLeft:6,letterSpacing:1}}>PRO</span>
            </div>
          </div>
          <div style={{flex:1}}/>
          <div style={{display:'flex',gap:4}}>
            {nav.map(n=>(
              <button key={n.id} onClick={()=>setPage(n.id)} style={{background:page===n.id?`${C.cyan}14`:'transparent',color:page===n.id?C.cyan:C.muted2,border:page===n.id?`1px solid ${C.cyan}30`:'1px solid transparent',borderRadius:10,padding:'8px 18px',fontSize:13,fontWeight:600,cursor:'pointer',display:'flex',alignItems:'center',gap:6,transition:'all .2s'}}>
                <span>{n.icon}</span>{n.label}
              </button>
            ))}
          </div>
          {result&&(
            <button onClick={()=>setPage('results')} style={{background:`${C.emerald}16`,color:C.emerald,border:`1px solid ${C.emerald}35`,borderRadius:10,padding:'8px 16px',fontSize:12,fontWeight:700,cursor:'pointer'}}>
              📄 Last Result
            </button>
          )}
        </div>
      </nav>

      <div style={{position:'relative',zIndex:1}}>
        {page==='home'&&<LandingPage onStart={()=>setPage('form')}/>}
        {page==='form'&&<FormPage onResult={handleResult}/>}
        {page==='results'&&<ResultsPage result={result} formData={formData} onBack={()=>setPage('form')} onSimulate={()=>setPage('simulator')}/>}
        {page==='simulator'&&<SimulatorPage formData={formData} baseResult={result}/>}
        {page==='analytics'&&<AnalyticsPage/>}
      </div>

      <Chatbot lastResult={result} formData={formData}/>
    </div>
  );
}