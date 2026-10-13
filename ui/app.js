const $ = selector => document.querySelector(selector);
const state = {contest:null, filter:'all', query:'', page:1, pageSize:25, selected:null, list:null, request:0, detailRequest:0};
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const number = value => Number(value).toLocaleString('en-US');
const clock = seconds => `${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;
const language = name => name.replace('GNU ','').replace(' (GCC ',' · ').replace(')','').replace('PyPy 3-64','PyPy 3');
const statusNames = {unchecked:'Not checked', review:'Flagged for review', low_priority:'Low priority'};
const statusHtml = status => `<span class="status ${esc(status)}"><span class="status-mark">${status === 'review' ? '↗' : status === 'low_priority' ? '·' : '—'}</span>${esc(statusNames[status])}</span>`;
async function api(path, options={}) { const response=await fetch(path,options); const value=await response.json(); if(!response.ok) throw new Error(value.error || 'The request could not be completed.'); return value; }
function notice(message, success=false) { const el=$('#notice'); el.textContent=message; el.classList.toggle('success',success); el.hidden=!message; }
function paintOverview() {
  const c=state.contest; const date=new Intl.DateTimeFormat('en-US',{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'}).format(new Date(c.start_time*1000));
  document.body.classList.toggle('reviewed',c.reviewed);
  $('#contest-meta').innerHTML=`Div. 2 <span class="meta-divider">/</span> ${esc(date)} <span class="meta-divider">/</span> ${c.duration/3600}h contest <span class="meta-divider">/</span> Finished`;
  $('#participant-count').textContent=number(c.participants); $('#submission-count').textContent=number(c.submissions); $('#all-tab-count').textContent=number(c.participants);
  $('#run-button').disabled=c.reviewed; $('#run-button').classList.toggle('complete',c.reviewed);
  $('#run-label').textContent=c.reviewed?'Review complete':'Run review'; $('#run-icon').textContent=c.reviewed?'✓':'↗';
  $('#model-status').innerHTML=`<span class="local-dot"></span> ${c.reviewed?'Policy pass complete':'Model ready'}`;
  document.querySelector('.summary-result').classList.toggle('reviewed',c.reviewed); $('#review-stamp').hidden=!c.reviewed;
  if(c.reviewed) {
    $('#flag-count').textContent=number(c.flagged); $('#flag-label').innerHTML=`Flagged for review <span class="summary-note">${(100*c.flagged/c.participants).toFixed(1)}% of reviewed sessions</span>`;
    $('#review-tab-count').textContent=number(c.flagged); $('#low-tab-count').textContent=number(c.participants-c.flagged);
    document.querySelectorAll('.tab').forEach(tab=>tab.disabled=false);
  }
}
async function loadList(keepSelection=false) {
  const request=++state.request; const params=new URLSearchParams({q:state.query,status:state.filter,page:state.page,page_size:state.pageSize});
  try {
    const result=await api(`/api/participants?${params}`); if(request!==state.request)return; state.list=result;
    const offset=(state.page-1)*state.pageSize;
    $('#participant-rows').innerHTML=result.rows.length ? result.rows.map((row,index)=>`<tr data-id="${esc(row.id)}" class="${row.status==='review'?'flagged':''} ${row.id===state.selected?'selected':''}"><td class="number">${String(offset+index+1).padStart(2,'0')}</td><td><span class="participant-dot"></span><button class="participant-button" data-id="${esc(row.id)}">${esc(row.handle)}</button></td><td class="numeric">${row.problem_count}</td><td class="numeric">${row.attempts}</td><td class="time">+${clock(row.last_seconds)}</td><td>${statusHtml(row.status)}</td></tr>`).join('') : '<tr><td colspan="6" class="empty">No sessions match this view. Try another handle or queue.</td></tr>';
    $('#page-label').textContent=result.total?`${number(offset+1)}–${number(Math.min(offset+state.pageSize,result.total))} of ${number(result.total)} sessions`:'0 sessions';
    $('#prev-button').disabled=state.page===1; $('#next-button').disabled=offset+state.pageSize>=result.total;
    $('#queue-title').textContent=state.filter==='review'?'FLAGGED REVIEW QUEUE':state.filter==='low_priority'?'LOW PRIORITY SESSIONS':'PARTICIPANT DIRECTORY';
    $('#queue-note').textContent=state.contest.reviewed?'Policy recommendations · session-specific':'Select a session to inspect';
    if(!keepSelection && result.rows.length) await selectParticipant(result.rows[0].id);
  } catch(error) { notice(error.message); }
}
async function selectParticipant(id) {
  state.selected=id; const request=++state.detailRequest;
  document.querySelectorAll('#participant-rows tr').forEach(row=>row.classList.toggle('selected',row.dataset.id===id));
  try {
    const detail=await api(`/api/participants/${encodeURIComponent(id)}`); if(request!==state.detailRequest)return;
    const markers=detail.submissions.map(s=>`<span class="timeline-marker ${s.verdict==='SKIPPED'?'skipped':''}" data-position="${Math.min(99.5,Math.max(0,s.seconds/detail.duration*100))}" title="${esc(s.problem)} · +${clock(s.seconds)}"></span>`).join('');
    const trace=detail.actions.map((action,i)=>`<span class="trace-step ${action==='review'||action==='low_priority'?'terminal':''} ${action==='review'?'review':''}">${i+1}. ${esc(({inspect_pacing:'Pacing',inspect_trajectory:'Trajectory',inspect_language:'Language',low_priority:'Low priority',review:'Review'})[action])}</span>`).join('');
    const decision=detail.status==='unchecked'?'<p class="policy-pending">This session has not been checked.<br>Run review to see the policy decision.</p>':`<p class="policy-decision ${detail.status}">${detail.status==='review'?'↗ Flagged for review':'· Low priority'}</p><div class="trace">${trace}</div><p class="trace-note">${detail.inspections} additional evidence ${detail.inspections===1?'group':'groups'} inspected</p>`;
    const verdictNames={OK:'Accepted',SKIPPED:'Skipped',WRONG_ANSWER:'Wrong answer',TIME_LIMIT_EXCEEDED:'Time limit',COMPILATION_ERROR:'Compile error',RUNTIME_ERROR:'Runtime error',MEMORY_LIMIT_EXCEEDED:'Memory limit'};
    const submissions=detail.submissions.map(s=>`<div class="submission"><span class="problem-tag" title="${esc(s.problem_name)}">${esc(s.problem)}</span><div><a class="submission-id" href="${esc(s.url)}" target="_blank" rel="noreferrer">#${s.id} ↗</a><span class="submission-language" title="${esc(s.language)}">${esc(language(s.language))}</span></div><div class="submission-right">+${clock(s.seconds)}<span class="verdict ${s.verdict==='OK'?'ok':s.verdict==='SKIPPED'?'skipped':'error'}">${esc(verdictNames[s.verdict] || s.verdict.replaceAll('_',' ').toLowerCase())}</span></div></div>`).join('');
    $('#inspector').innerHTML=`<div class="detail-header"><span class="detail-eyebrow">SESSION INSPECTOR</span><div class="detail-title-row"><h2>${esc(detail.handle)}</h2><a class="external-link" href="https://codeforces.com/profile/${encodeURIComponent(detail.handle)}" aria-label="Open profile on Codeforces" target="_blank" rel="noreferrer">↗</a></div><div class="detail-basics"><span><b>${detail.problem_count}</b> problems</span><span><b>${detail.attempts}</b> attempts</span><span><b>+${clock(detail.last_seconds)}</b> last</span></div></div><div class="detail-section"><div class="section-label">SUBMISSION WINDOW<span class="duration-label">${detail.duration/60} min</span></div><div class="timeline">${markers}</div><div class="timeline-axis"><span>00:00</span><span>${clock(detail.duration/2)}</span><span>${clock(detail.duration)}</span></div></div><div class="detail-section"><div class="section-label">POLICY DECISION</div>${decision}</div><div class="detail-section"><div class="section-label">SUBMISSIONS<span class="duration-label">${detail.attempts} recorded</span></div><div class="submission-list">${submissions}</div><p class="verdict-note">Recorded verdicts are shown here for review. They are not model inputs.</p></div>`;
    // Position markers with CSS properties; executable scripts remain external.
    $('#inspector').querySelectorAll('.timeline-marker').forEach(marker=>marker.style.left=`${marker.dataset.position}%`);
  } catch(error) { notice(error.message); }
}
async function runReview() {
  const button=$('#run-button'); button.disabled=true; button.classList.add('running'); $('#run-label').textContent='Reviewing…'; $('#run-icon').textContent=''; notice('');
  try {
    const result=await api('/api/review',{method:'POST'}); state.contest=result; state.page=1; state.filter='review'; state.query=''; $('#search-input').value='';
    document.querySelectorAll('.tab').forEach(tab=>{const active=tab.dataset.filter===state.filter;tab.classList.toggle('active',active);tab.setAttribute('aria-selected',String(active));});
    paintOverview(); notice(`${number(result.flagged)} sessions moved to the review queue. The saved policy checked ${number(result.participants)} sessions in ${result.elapsed_seconds.toFixed(2)}s.`,true);
    await loadList();
  } catch(error) { notice(error.message); button.disabled=false; $('#run-label').textContent='Try review again'; $('#run-icon').textContent='↗'; }
  finally {button.classList.remove('running');}
}
$('#run-button').addEventListener('click',runReview);
$('#participant-rows').addEventListener('click',event=>{const row=event.target.closest('tr[data-id]');if(row)selectParticipant(row.dataset.id);});
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>{state.filter=tab.dataset.filter;state.page=1;document.querySelectorAll('.tab').forEach(other=>{const active=other===tab;other.classList.toggle('active',active);other.setAttribute('aria-selected',String(active));});loadList();}));
let searchTimer; $('#search-input').addEventListener('input',event=>{clearTimeout(searchTimer); searchTimer=setTimeout(()=>{state.query=event.target.value;state.page=1;loadList();},180);});
$('#prev-button').addEventListener('click',()=>{state.page--;loadList();}); $('#next-button').addEventListener('click',()=>{state.page++;loadList();});
$('#about-button').addEventListener('click',()=>$('#about-dialog').showModal()); $('#close-dialog').addEventListener('click',()=>$('#about-dialog').close());
$('#about-dialog').addEventListener('click',event=>{if(event.target===$('#about-dialog')){const r=event.target.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)event.target.close();}});
(async()=>{try{state.contest=await api('/api/contest');paintOverview();await loadList();}catch(error){notice(`Workspace unavailable: ${error.message}`);$('#model-status').textContent='Data could not load';}})();
