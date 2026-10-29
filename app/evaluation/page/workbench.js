'use strict';
const $ = id => document.getElementById(id);
const state = {datasets: [], queries: [], reports: [], detail: null, facts: [], evidence: [], patches: new Map(), dirty: false};
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = value => typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : '—';
const apiBase = () => `/api/datasets/${encodeURIComponent($('dataset').value)}`;
let noticeTimer;
function notify(message, error = false) { clearTimeout(noticeTimer); $('notice').textContent = message; $('notice').className = error ? 'error' : ''; $('notice').hidden = false; noticeTimer = setTimeout(() => $('notice').hidden = true, 9000); }
async function api(url, options = {}) { const response = await fetch(url, {headers: {'Content-Type':'application/json'}, ...options}); const data = await response.json(); if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)); return data; }
async function action(button, fn) { button.disabled = true; try { await fn(); } catch (error) { notify(error.message, true); } finally { button.disabled = false; } }
function guard() { return !state.dirty || window.confirm('有尚未保存的审核修改，确定切换吗？'); }
function options(element, rows, valueKey, label, selected) { element.innerHTML = rows.map(row => `<option value="${escape(row[valueKey])}">${escape(label(row))}</option>`).join(''); if (selected && rows.some(r => r[valueKey] === selected)) element.value = selected; }
async function loadDatasets(selected) {
  state.datasets = (await api('/api/datasets')).filter(d => !d.error);
  const preferred = state.datasets.some(d => d.id === 'productdoc-diverse-v1') ? 'productdoc-diverse-v1' : state.datasets[0]?.id;
  for (const id of ['dataset','runDataset','sourceDataset']) options($(id), state.datasets, 'id', d => `${d.id} · ${d.approved}/${d.total}`, id === 'sourceDataset' ? 'productdoc-v1' : selected || $('dataset').value || preferred);
  await loadQueries();
}
async function loadQueries() {
  const dataset = state.datasets.find(d => d.id === $('dataset').value);
  $('datasetInfo').textContent = dataset ? `批准 ${dataset.approved} / ${dataset.total} 题 · 人工核验 ${dataset.human_verified} 题` : '';
  $('provenance').textContent = dataset?.human_verified ? `已确认人工核验 ${dataset.human_verified} 题。其他样本仍按原来源标识。` : '当前没有人工核验样本。AI来源审核与人工审核分别统计。';
  const params = new URLSearchParams({search:$('search').value, status:$('status').value, cases_only:$('casesOnly').checked});
  state.queries = await api(`${apiBase()}/queries?${params}`);
  $('queryCount').textContent = state.queries.length;
  $('queries').innerHTML = state.queries.length ? state.queries.map(q => `<button class="query-item ${state.detail?.query.query_id === q.query_id ? 'selected' : ''}" data-qid="${escape(q.query_id)}"><small>${escape(q.query_id)} · ${escape(q.category)}</small><strong>${escape(q.query)}</strong><div class="tags">${q.case_count ? `<span class="tag bad">${q.case_count} 条失败观察</span>` : '<span class="tag">常规样本</span>'}${q.draft_revision ? `<span class="tag good">${q.published_to ? '已发布' : `审核 v${q.draft_revision}`}</span>` : ''}</div></button>`).join('') : '<p class="hint" style="padding:16px">没有符合条件的问题。</p>';
  document.querySelectorAll('[data-qid]').forEach(button => button.onclick = () => { if (guard()) action(button, () => loadDetail(button.dataset.qid)); });
}
function renderFacts() {
  $('facts').innerHTML = state.facts.map((f,i) => `<div class="fact-card"><div class="fact-head"><span>必要事实 ${i + 1}</span><button class="text-button" data-remove-fact="${i}">移除</button></div><textarea data-fact="${i}" rows="2" aria-label="必要事实${i+1}">${escape(f.text)}</textarea><details><summary>支持证据 · ${f.evidence_ids.length} 条${f.evidence_sets ? '，已配置替代路径' : '，默认联合支持'}</summary><p class="hint">修改勾选后按任一条证据足够处理；需要联合证明时，可在“编辑证据组合”中设置。</p>${state.evidence.map(e => `<label class="check"><input type="checkbox" data-fact-evidence="${i}" value="${escape(e.evidence_id)}" ${f.evidence_ids.includes(e.evidence_id) ? 'checked' : ''}>${escape(e.quote.slice(0,120))}</label>`).join('')}</details></div>`).join('');
  document.querySelectorAll('[data-fact]').forEach(el => el.oninput = () => { state.facts[Number(el.dataset.fact)].text = el.value; state.dirty = true; syncFactsJson(); });
  document.querySelectorAll('[data-remove-fact]').forEach(el => el.onclick = () => { state.facts.splice(Number(el.dataset.removeFact),1); state.dirty = true; renderFacts(); });
  document.querySelectorAll('[data-fact-evidence]').forEach(el => el.onchange = () => { const i=Number(el.dataset.factEvidence); const ids=[...document.querySelectorAll(`[data-fact-evidence="${i}"]:checked`)].map(x=>x.value); Object.assign(state.facts[i], {evidence_ids:ids, evidence_sets:ids.map(id=>[id])}); state.dirty=true; syncFactsJson(); });
  syncFactsJson();
}
function syncFactsJson() { $('factsJson').value = JSON.stringify(state.facts, null, 2); }
function renderEvidence() {
  $('evidenceCount').textContent = state.evidence.length;
  $('evidence').innerHTML = state.evidence.length ? state.evidence.map(e => {const pages=e.locator.pdf_pages || []; return `<div class="evidence-card"><small>${escape(e.evidence_id)}</small><pre>${escape(e.quote)}</pre><small>${escape(e.locator.match || '来源定位')} · ${escape(e.locator.extraction_method || '')}</small>${pages.length ? `<br><a href="${apiBase()}/documents/${encodeURIComponent(e.doc_id)}/pdf#page=${pages[0]}" target="_blank" rel="noopener">查看原始 PDF · 物理第 ${escape(pages.join('、'))} 页 ↗</a>` : ''}</div>`;}).join('') : '<p class="hint">当前无正向答案证据。无答案结论表示冻结手册没有支持，不能推断设备绝对不具备功能。</p>';
}
function renderChunks() {
  $('chunks').innerHTML = state.detail.chunks.map(c => `<details class="chunk"><summary><span>${escape(c.canonical_name || c.file_title)}<small>${escape(c.chunk_id)} · ${c.content.length} 字符</small></span><select data-grade="${escape(c.chunk_id)}" aria-label="${escape(c.chunk_id)}相关性"><option value="">未判断</option>${[0,1,2].map(g=>`<option value="${g}" ${c.relevance===g ? 'selected' : ''}>${g} · ${['无关','有帮助','直接证据'][g]}</option>`).join('')}</select></summary><pre>${escape(c.content)}</pre></details>`).join('');
  document.querySelectorAll('[data-grade]').forEach(el => { el.onclick = e=>e.stopPropagation(); el.onchange = ()=>{const c=state.detail.chunks.find(c=>c.chunk_id===el.dataset.grade); if(el.value===''){state.patches.delete(c.chunk_id);el.value=c.relevance??'';return;} const grade=Number(el.value);state.patches.set(c.chunk_id,{chunk_id:c.chunk_id,relevance:grade,evidence_ids:grade ? c.evidence_ids : []});state.dirty=true;}; });
  options($('anchorChunk'), state.detail.chunks, 'chunk_id', c=>`${c.canonical_name || c.file_title} · ${c.chunk_id}`);
}
async function loadDetail(qid) {
  const d = await api(`${apiBase()}/queries/${encodeURIComponent(qid)}`); d.datasetId=$('dataset').value; state.detail=d; state.patches.clear();
  const draft=d.review?.payload; const update=draft?.decision.query || {}; state.facts=structuredClone(update.reference_facts ?? d.query.reference_facts); state.evidence=structuredClone([...d.evidence, ...(draft?.decision.evidence || [])]);
  if(draft) for(const patch of draft.decision.qrel_patches || []) state.patches.set(patch.chunk_id,patch);
  $('empty').hidden=true;$('detail').hidden=false;$('qid').textContent=qid;$('question').textContent=d.query.query;$('split').textContent=d.query.split;
  $('answerable').value=String(update.answerable ?? d.query.answerable ?? true);$('groupId').value=update.group_id ?? d.query.group_id ?? '';$('retrievalQuery').value=update.retrieval_query ?? d.query.retrieval_query ?? '';
  $('reviewer').value=draft?.decision.reviewer || localStorage.getItem('productdocReviewer') || '';$('notes').value=draft?.decision.notes || '';$('human').checked=draft?.decision.human_verified || false;$('revision').textContent=`审核修订 ${d.review?.revision || 0}${d.review?.published_to ? ' · 已发布到 '+d.review.published_to : ''}`;
  $('saveReview').disabled=d.query.split==='test';
  options($('sourceCase'), [{case_id:'',variant:'不关联案例',diagnosis:[]},...d.cases], 'case_id', c=>c.case_id ? `${c.variant} · ${c.diagnosis.map(x=>x.type).join(' / ')}` : '不关联案例', draft?.decision.source_case_id);
  renderFacts();renderEvidence();renderChunks();
  for(const [cid,patch] of state.patches) {const select=[...document.querySelectorAll('[data-grade]')].find(e=>e.dataset.grade===cid);if(select)select.value=patch.relevance;}
  $('cases').innerHTML=d.cases.map(c=>{const o=c.observation;const detail=o.answer_record ? {answer:o.answer_record.answer,context_source:o.answer_record.context_source,assessment:o.assessment} : {metrics:o.metrics,stages:Object.fromEntries(Object.entries(o.stages || {}).filter(([k])=>k!=='eligible').map(([k,v])=>[k,v.map((x,i)=>({rank:i+1,chunk_id:x.chunk_id,score:x.score}))]))};return `<details class="case"><summary>${escape(c.variant)} · ${escape(c.diagnosis.map(x=>x.type).join(' / '))}</summary><small class="muted">${escape(c.case_id)}</small><pre>${escape(JSON.stringify(detail,null,2))}</pre></details>`;}).join('');
  state.dirty=false;await loadQueries();
}
async function saveReview() {
  if(!state.detail)return;
  if($('answerable').value==='false' && state.facts.length)throw new Error('无答案题不能保留参考事实，请先移除事实并修改相关正向标注。');
  const facts=state.facts.map((f,i)=>({...f,fact_id:f.fact_id || `${state.detail.query.query_id}:review:${i+1}`}));
  const evidenceIds=[...new Set(facts.flatMap(f=>f.evidence_ids))];
  const originalIds=new Set(state.detail.evidence.map(e=>e.evidence_id));
  const patches=new Map(state.patches);
  for(const e of state.evidence.filter(e=>!originalIds.has(e.evidence_id))){ for(const c of state.detail.chunks.filter(c=>c.doc_id===e.doc_id && c.content.includes(e.quote))){const p=patches.get(c.chunk_id)||{chunk_id:c.chunk_id,relevance:2,evidence_ids:[...c.evidence_ids]};if(p.relevance===0)throw new Error('新增证据所属片段被标为无关，请修正相关性');p.evidence_ids=[...new Set([...p.evidence_ids,e.evidence_id])];patches.set(c.chunk_id,p);} }
  const query={answerable:$('answerable').value==='true',reference_facts:facts,evidence_ids:evidenceIds,group_id:$('groupId').value.trim()};
  if($('retrievalQuery').value.trim())query.retrieval_query=$('retrievalQuery').value.trim();
  const body={expected_revision:state.detail.review?.revision||0,reviewer:$('reviewer').value,notes:$('notes').value,human_verified:$('human').checked,source_case_id:$('sourceCase').value||null,query,evidence:state.evidence.filter(e=>!originalIds.has(e.evidence_id)),qrel_patches:[...patches.values()]};
  const result=await api(`${apiBase()}/queries/${encodeURIComponent(state.detail.query.query_id)}/reviews`,{method:'POST',body:JSON.stringify(body)});
  localStorage.setItem('productdocReviewer',body.reviewer);state.dirty=false;await loadDetail(state.detail.query.query_id);notify(`已保存审核修订 ${result.revision} · ${result.human_verified?'人工核验':'来源审核'}`);
}
async function loadReports() {
  state.reports=await api('/api/reports');
  const old=$('compareReport').value;
  const selectedPredictions=[...$('predictionReports').selectedOptions].map(o=>o.value);
  options($('predictionReports'),state.reports.filter(r=>r.schema==='productdoc.experiment.v1'),'id',r=>r.name);
  for(const option of $('predictionReports').options)option.selected=selectedPredictions.includes(option.value);
  options($('compareReport'),state.reports,'id',r=>r.name,old || state.reports.find(r=>r.name==='logs/source_diagnostic_v2/retrieval.json')?.id);
  options($('compareWith'),[{id:'',name:'不添加对照报告'},...state.reports],'id',r=>r.name,$('compareWith').value);
  await showComparison();
}
function reportTable(report) {
  const summary=report.summary;
  let heads,rows;
  if(report.schema==='productdoc.context-answer-experiment.v1') {heads=['上下文策略','完成 / 总数','事实覆盖率','答案正确率','事实支持率','误拒答率','平均输入token'];rows=Object.entries(summary).map(([policy,s])=>[policy,`${s.completed}/${s.total}`,pct(s.context_metrics.fact_recall?.mean),pct(s.answer_metrics.correct?.mean),pct(s.answer_metrics.faithfulness?.mean),pct(s.false_abstention_rate),s.mean_prompt_tokens?.toFixed(0)??'—']);}
  else if(Array.isArray(summary)) {heads=['方案','Top-K','字符预算','事实覆盖率','完整覆盖率','平均字符'];rows=summary.map(s=>[s.variant,s.context_k,s.budget,pct(s.metrics.fact_recall?.mean),pct(s.metrics.all_facts?.mean),Math.round(s.mean_context_chars)]);}
  else if(report.schema==='productdoc.answer-eval.v1') {heads=['指标','结果','样本数'];rows=Object.entries(summary.metrics).filter(([k])=>['correct','faithfulness','citation_precision','citation_completeness'].includes(k)).map(([k,v])=>[k,pct(v.mean),v.n]);rows.push(['误拒答率',pct(summary.false_abstention_rate),summary.answerable_n],['无答案误回答率',pct(summary.false_answer_rate),summary.unanswerable_n]);}
  else {heads=['方案','完成 / 总数','Recall@50','nDCG@5','事实覆盖率','完整覆盖率','判定覆盖@5'];rows=Object.entries(summary).map(([k,s])=>{const m=s.metrics||{};return[k,`${s.completed}/${s.total}`,pct(m.candidates?.['recall@50']?.mean),pct(m.rerank_order?.['ndcg@5']?.mean),pct(m.context_evidence?.fact_recall?.mean),pct(m.context_evidence?.all_facts?.mean),pct(m.rerank_order?.['judged_fraction@5']?.mean)];});}
  return `<h3 style="margin-top:22px">${escape(report.name)}</h3><p class="hint">数据指纹 ${escape(report.dataset_sha256?.slice(0,20))}… · ${escape(JSON.stringify(report.config || {}))}</p><div class="table-wrap"><table><thead><tr>${heads.map(h=>`<th>${escape(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(v=>`<td>${escape(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}
async function showComparison() {
  const report=state.reports.find(r=>r.id===$('compareReport').value);if(!report)return;
  const second=state.reports.find(r=>r.id===$('compareWith').value);
  if(second && (second.dataset_sha256!==report.dataset_sha256 || second.schema!==report.schema)) {
    $('comparison').innerHTML='<p class="hint">所选报告的数据版本或评测类型不同，不能作为同口径对照。请重新选择。</p>'+reportTable(report);return;
  }
  $('comparison').innerHTML=reportTable(report)+(second?reportTable(second):'');
}
async function loadJobs() {
  const jobs=await api('/api/runs');
  $('jobs').innerHTML=jobs.length?jobs.map(j=>`<div class="job"><strong>${escape(j.config.kind)} · ${escape(j.config.dataset)}</strong> <span class="tag ${j.status==='completed'?'good':j.status==='failed'?'bad':''}">${escape(j.status)}</span><p>${escape(j.id)} · ${escape(new Date(j.updated_at).toLocaleString())}</p>${j.error?`<pre>${escape(j.error)}</pre>`:''}${['failed','interrupted'].includes(j.status)?`<button class="text-button" data-retry="${j.id}">重试任务</button>`:''}${j.status==='completed'?'<small class="muted">结果已加入报告列表</small>':''}</div>`).join(''):'<p class="hint">暂无任务。已有历史实验可在下方查看。</p>';
  document.querySelectorAll('[data-retry]').forEach(b=>b.onclick=()=>action(b,async()=>{await api(`/api/runs/${b.dataset.retry}/retry`,{method:'POST'});await loadJobs();}));
  return jobs;
}
function numberList(value) {const values=value.split(',').map(v=>Number(v.trim()));if(values.some(v=>!Number.isInteger(v)||v<1))throw new Error('请输入逗号分隔的正整数');return [...new Set(values)];}
$('saveReview').onclick=()=>action($('saveReview'),saveReview);
$('addFact').onclick=()=>{state.facts.push({text:'',evidence_ids:[]});state.dirty=true;renderFacts();};
$('applyFacts').onclick=()=>action($('applyFacts'),async()=>{const f=JSON.parse($('factsJson').value);if(!Array.isArray(f)||f.some(x=>typeof x.text!=='string'||!Array.isArray(x.evidence_ids)))throw new Error('参考事实格式不正确');state.facts=f;state.dirty=true;renderFacts();notify('已应用，请继续保存审核决定');});
$('addAnchor').onclick=()=>action($('addAnchor'),async()=>{const c=state.detail?.chunks.find(c=>c.chunk_id===$('anchorChunk').value);const quote=$('anchorQuote').value;const page=Number($('anchorPage').value);if(!c||!quote.trim()||!c.content.includes(quote)||!Number.isInteger(page)||page<1)throw new Error('连续引文必须逐字来自所选片段，并填写物理页码');const doc=state.detail.documents.find(d=>d.doc_id===c.doc_id);const e={evidence_id:'ev_review_'+crypto.randomUUID().replaceAll('-','').slice(0,20),doc_id:c.doc_id,source_revision:doc.source_revision,quote,locator:{pdf_pages:[page],match:'explicit_workbench_source_review'}};state.evidence.push(e);state.dirty=true;renderEvidence();renderFacts();$('anchorQuote').value='';notify('证据已添加，请在对应参考事实中勾选支持证据');});
for(const id of ['reviewer','notes','human','answerable','groupId','retrievalQuery','sourceCase'])$(id).addEventListener('input',()=>state.dirty=true);
$('dataset').onchange=()=>{if(!guard()){if(state.detail)$('dataset').value=state.detail.datasetId;return;}state.detail=null;state.dirty=false;$('detail').hidden=true;$('empty').hidden=false;action($('refresh'),loadQueries);};
let searchTimer;$('search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>action($('refresh'),loadQueries),250);};for(const id of ['status','casesOnly'])$(id).onchange=()=>action($('refresh'),loadQueries);
$('refresh').onclick=()=>action($('refresh'),async()=>{await loadQueries();await loadReports();await loadJobs();});
$('publish').onclick=()=>action($('publish'),async()=>{await loadQueries();const rows=state.queries.filter(q=>q.review_id&&!q.published_to);if(!rows.length)throw new Error('当前列表没有尚未发布的审核决定');$('publishList').innerHTML=rows.map(q=>`<label class="check"><input type="checkbox" value="${escape(q.review_id)}" checked>${escape(q.query_id)} · ${escape(q.query)}</label>`).join('');$('publishDialog').showModal();});
$('confirmPublish').onclick=()=>action($('confirmPublish'),async()=>{const ids=[...$('publishList').querySelectorAll('input:checked')].map(e=>e.value);const r=await api(`${apiBase()}/publish`,{method:'POST',body:JSON.stringify({output_name:$('outputName').value,version:$('version').value,review_ids:ids})});$('publishDialog').close();state.dirty=false;state.detail=null;$('detail').hidden=true;$('empty').hidden=false;await loadDatasets(r.dataset_id);notify(`已发布 ${r.dataset_id}，包含 ${r.reviewed} 条审核决定`);});
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('active',x===b));$('reviewTab').hidden=b.dataset.tab!=='review';$('experimentsTab').hidden=b.dataset.tab!=='experiments';$('pageTitle').textContent=b.dataset.tab==='review'?'让每个判断，都有证据。':'从一次实验，走向下一次改进。';$('pageSub').textContent=b.dataset.tab==='review'?'核对原文、修正标注，再发布可追溯的数据版本。':'冻结输入、对比策略，保留每次运行的结果。';if(b.dataset.tab==='experiments')action(b,async()=>{await loadReports();await loadJobs();});});
$('runKind').onchange=()=>{$('sourceLabel').hidden=$('runKind').value!=='retrieval_replay';$('sweepFields').hidden=$('runKind').value!=='context_sweep';};
$('createRun').onclick=()=>action($('createRun'),async()=>{const body={kind:$('runKind').value,dataset:$('runDataset').value,source_dataset:$('sourceDataset').value,prediction_ids:[...$('predictionReports').selectedOptions].map(o=>o.value),context_ks:numberList($('contextKs').value),budgets:numberList($('budgets').value)};const r=await api('/api/runs',{method:'POST',body:JSON.stringify(body)});notify(`已创建任务 ${r.id}`);await loadJobs();});
$('refreshJobs').onclick=()=>action($('refreshJobs'),async()=>{await loadJobs();await loadReports();});$('compareReport').onchange=()=>showComparison();$('compareWith').onchange=()=>showComparison();
window.addEventListener('beforeunload',e=>{if(state.dirty){e.preventDefault();e.returnValue='';}});
setInterval(async()=>{if(!$('experimentsTab').hidden){try{const jobs=await loadJobs();if(jobs.some(j=>j.status==='completed'))await loadReports();}catch(error){notify(error.message,true);}}},8000);
(async()=>{try{await loadDatasets('productdoc-source-v2');await loadReports();await loadJobs();}catch(error){notify(error.message,true);}})();
