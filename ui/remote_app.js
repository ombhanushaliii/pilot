const $ = selector => document.querySelector(selector);
const state = {contest:null, rows:[], review:null, snapshotId:null, loading:false, filter:'all', query:'', page:1, pageSize:25, selected:null};
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const number = value => Number(value).toLocaleString('en-US');
const clock = seconds => `${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;
const language = name => name.replace('GNU ','').replace(' (GCC ',' · ').replace(')','').replace('PyPy 3-64','PyPy 3');
const statusNames = {unchecked:'Not checked', review:'Flagged for review', low_priority:'Low priority', insufficient:'Outside model scope'};
const decisionFor = row => state.review?.decisions?.[row.id] || {status:row.eligible?'unchecked':'insufficient', inspections:0, actions:[]};
async function api(path, options={}) {
  const headers = new Headers(options.headers || {});
  if(options.body) headers.set('Content-Type','application/json');
  const response = await fetch(path,{...options,headers});
  if(!response.headers.get('Content-Type')?.includes('application/json'))throw new Error('The review service is temporarily unavailable. Try loading the contest again.');
  const value = await response.json();
  if(!response.ok) throw new Error(value.error || 'The request could not be completed.');
  return value;
}
function notice(message, success=false) { const el=$('#notice'); el.textContent=message; el.classList.toggle('success',success); el.hidden=!message; }
function paintOverview() {
  const c=state.contest, date=new Intl.DateTimeFormat('en-US',{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'}).format(new Date(c.start_time*1000));
  document.body.classList.toggle('reviewed',Boolean(state.review));
  $('#contest-title').innerHTML=`${esc(c.name)}<span class="heading-dot">.</span>`;
  $('#contest-meta').innerHTML=`${esc(date)} <span class="meta-divider">/</span> ${c.duration/3600}h contest <span class="meta-divider">/</span> Finished`;
  $('#participant-count').textContent=number(c.participants); $('#submission-count').textContent=number(c.submissions); $('#all-tab-count').textContent=number(c.participants);
  $('#run-button').disabled=Boolean(state.review); $('#run-button').classList.toggle('complete',Boolean(state.review));
  $('#run-label').textContent=state.review?'Review complete':'Run review'; $('#run-icon').textContent=state.review?'✓':'↗';
  $('#model-status').innerHTML=`<span class="local-dot"></span> ${state.review?'Policy pass complete':'Model ready'}`;
  document.querySelector('.summary-result').classList.toggle('reviewed',Boolean(state.review)); $('#review-stamp').hidden=!state.review;
  if(state.review) {
    $('#flag-count').textContent=number(state.review.flagged); $('#flag-label').innerHTML=`Flagged for review <span class="summary-note">${(100*state.review.flagged/state.review.participants).toFixed(1)}% of scored sessions</span>`;
    $('#review-tab-count').textContent=number(state.review.flagged); $('#low-tab-count').textContent=number(state.review.participants-state.review.flagged);
    document.querySelectorAll('.tab').forEach(tab=>tab.disabled=false);
  } else {
    $('#flag-count').textContent='—'; $('#flag-label').innerHTML='Awaiting review <span class="summary-note">one pass, saved model</span>';
    $('#review-tab-count').textContent='—'; $('#low-tab-count').textContent='—';
    document.querySelectorAll('.tab').forEach(tab=>tab.disabled=tab.dataset.filter!=='all');
  }
  $('#run-button').disabled=Boolean(state.review)||!c.eligible;
  $('#contest-link').href=`https://codeforces.com/contest/${c.contest_id}`;
}
function loadList(keepSelection=false) {
  const matching=state.rows.filter(row=>row.handle.toLowerCase().includes(state.query.toLowerCase()) && (state.filter==='all'||decisionFor(row).status===state.filter));
  matching.sort((a,b)=>(decisionFor(a).status==='review'?-1:0)-(decisionFor(b).status==='review'?-1:0)||a.handle.toLowerCase().localeCompare(b.handle.toLowerCase()));
  const offset=(state.page-1)*state.pageSize, shown=matching.slice(offset,offset+state.pageSize);
  $('#participant-rows').innerHTML=shown.length ? shown.map((row,index)=>{const decision=decisionFor(row);return `<tr data-id="${esc(row.id)}" class="${decision.status==='review'?'flagged':''} ${row.id===state.selected?'selected':''}"><td class="number">${String(offset+index+1).padStart(2,'0')}</td><td><span class="participant-dot"></span><button class="participant-button" data-id="${esc(row.id)}">${esc(row.handle)}</button></td><td class="numeric">${row.problem_count}</td><td class="numeric">${row.attempts}</td><td class="time">+${clock(row.last_seconds)}</td><td>${statusHtml(decision.status)}</td></tr>`;}).join('') : '<tr><td colspan="6" class="empty">No sessions match this view. Try another handle or queue.</td></tr>';
  $('#page-label').textContent=matching.length?`${number(offset+1)}–${number(Math.min(offset+state.pageSize,matching.length))} of ${number(matching.length)} sessions`:'0 sessions';
  $('#prev-button').disabled=state.page===1; $('#next-button').disabled=offset+state.pageSize>=matching.length;
  $('#queue-title').textContent=state.filter==='review'?'FLAGGED REVIEW QUEUE':state.filter==='low_priority'?'LOW PRIORITY SESSIONS':'PARTICIPANT DIRECTORY';
  $('#queue-note').textContent=state.review?'Policy recommendations · session-specific':'Select a session to inspect';
  if(!keepSelection && shown.length) selectParticipant(shown[0].id);
  if(!shown.length){state.selected=null;$('#inspector').innerHTML='<p class="policy-pending">No sessions in this view.</p>';}
}
function statusHtml(status) { return `<span class="status ${esc(status)}"><span class="status-mark">${status==='review'?'↗':status==='low_priority'?'·':'—'}</span>${esc(statusNames[status])}</span>`; }
async function selectParticipant(id) {
  state.selected=String(id);
  document.querySelectorAll('#participant-rows tr').forEach(row=>row.classList.toggle('selected',row.dataset.id===state.selected));
  const detail=state.rows.find(row=>row.id===state.selected); if(!detail)return;
  if(!detail.submissions) {
    const snapshotId=state.snapshotId;
    $('#inspector').innerHTML='<p class="policy-pending">Loading submission sequence…</p>';
    try {
      const loaded=await api(`/api/detail?contestId=${state.contest.contest_id}&sessionId=${encodeURIComponent(id)}&snapshotId=${snapshotId}`);
      if(state.snapshotId!==snapshotId)return;
      detail.submissions=loaded.submissions;
      if(state.selected!==String(id))return;
    } catch(error) {if(state.selected===String(id))$('#inspector').innerHTML=`<p class="policy-pending">${esc(error.message)}</p>`;return;}
  }
  const decision=decisionFor(detail), duration=state.contest.duration;
  const markers=detail.submissions.map(s=>`<span class="timeline-marker ${s.verdict==='SKIPPED'?'skipped':''}" data-position="${Math.min(99.5,Math.max(0,s.seconds/duration*100))}" title="${esc(s.problem)} · +${clock(s.seconds)}"></span>`).join('');
  const trace=decision.actions.map((action,i)=>`<span class="trace-step ${action==='review'||action==='low_priority'?'terminal':''} ${action==='review'?'review':''}">${i+1}. ${esc(({inspect_pacing:'Pacing',inspect_trajectory:'Trajectory',inspect_language:'Language',low_priority:'Low priority',review:'Review'})[action])}</span>`).join('');
  const decisionHtml=decision.status==='insufficient'?'<p class="policy-pending">Only one problem attempted.<br>This session is outside the trained model’s scope.</p>':decision.status==='unchecked'?'<p class="policy-pending">This session has not been checked.<br>Run review to see the policy decision.</p>':`<p class="policy-decision ${decision.status}">${decision.status==='review'?'↗ Flagged for review':'· Low priority'}</p><div class="trace">${trace}</div><p class="trace-note">${decision.inspections} additional evidence ${decision.inspections===1?'group':'groups'} inspected</p>`;
  const verdictNames={OK:'Accepted',SKIPPED:'Skipped',WRONG_ANSWER:'Wrong answer',TIME_LIMIT_EXCEEDED:'Time limit',COMPILATION_ERROR:'Compile error',RUNTIME_ERROR:'Runtime error',MEMORY_LIMIT_EXCEEDED:'Memory limit'};
  const submissions=detail.submissions.map(s=>`<div class="submission"><span class="problem-tag" title="${esc(s.problem_name)}">${esc(s.problem)}</span><div><a class="submission-id" href="https://codeforces.com/contest/${state.contest.contest_id}/submission/${s.id}" target="_blank" rel="noreferrer">#${s.id} ↗</a><span class="submission-language" title="${esc(s.language)}">${esc(language(s.language))}</span></div><div class="submission-right">+${clock(s.seconds)}<span class="verdict ${s.verdict==='OK'?'ok':s.verdict==='SKIPPED'?'skipped':'error'}">${esc(verdictNames[s.verdict]||s.verdict.replaceAll('_',' ').toLowerCase())}</span></div></div>`).join('');
  $('#inspector').innerHTML=`<div class="detail-header"><span class="detail-eyebrow">SESSION INSPECTOR</span><div class="detail-title-row"><h2>${esc(detail.handle)}</h2><a class="external-link" href="https://codeforces.com/profile/${encodeURIComponent(detail.handle)}" aria-label="Open profile on Codeforces" target="_blank" rel="noreferrer">↗</a></div><div class="detail-basics"><span><b>${detail.problem_count}</b> problems</span><span><b>${detail.attempts}</b> attempts</span><span><b>+${clock(detail.last_seconds)}</b> last</span></div></div><div class="detail-section"><div class="section-label">SUBMISSION WINDOW<span class="duration-label">${duration/60} min</span></div><div class="timeline">${markers}</div><div class="timeline-axis"><span>00:00</span><span>${clock(duration/2)}</span><span>${clock(duration)}</span></div></div><div class="detail-section"><div class="section-label">POLICY DECISION</div>${decisionHtml}</div><div class="detail-section"><div class="section-label">SUBMISSIONS<span class="duration-label">${detail.attempts} recorded</span></div><div class="submission-list">${submissions}</div><p class="verdict-note">Recorded verdicts are shown here for review. They are not model inputs.</p></div>`;
  $('#inspector').querySelectorAll('.timeline-marker').forEach(marker=>marker.style.left=`${marker.dataset.position}%`);
}
async function runReview() {
  if(!state.contest||state.loading)return;
  $('#load-button').disabled=true;
  const button=$('#run-button'); button.disabled=true; button.classList.add('running'); $('#run-label').textContent='Reviewing…'; $('#run-icon').textContent=''; notice('');
  try {
    const result=await api(`/api/review?contestId=${state.contest.contest_id}`,{method:'POST',body:JSON.stringify({snapshot_id:state.snapshotId})});
    state.review=result; state.page=1; state.filter='review'; state.query=''; $('#search-input').value='';
    document.querySelectorAll('.tab').forEach(tab=>{const active=tab.dataset.filter===state.filter;tab.classList.toggle('active',active);tab.setAttribute('aria-selected',String(active));});
    paintOverview(); notice(`${number(result.flagged)} sessions moved to the review queue. The saved policy checked ${number(result.participants)} sessions in ${result.elapsed_seconds.toFixed(2)}s.`,true); loadList();
  } catch(error) { notice(error.message); button.disabled=false; $('#run-label').textContent='Try review again'; $('#run-icon').textContent='↗'; }
  finally {button.classList.remove('running');$('#load-button').disabled=false;}
}
function parseContest(value) {
  const text=value.trim();
  const match=text.match(/^https?:\/\/(?:www\.)?codeforces\.com\/contest\/(\d+)(?:\/|[?#]|$)/i);
  const id=match?match[1]:text;
  if(!/^\d+$/.test(id)||Number(id)<1||Number(id)>100000)throw new Error('Enter a Codeforces contest URL or its contest ID. Round 1127 uses contest ID 2274.');
  return Number(id);
}
async function openContest(value) {
  if(state.loading)return;
  let id;
  try{id=parseContest(value);}catch(error){notice(error.message);return;}
  state.loading=true; $('#load-button').disabled=true; $('#run-button').disabled=true;
  $('#load-status').textContent='Fetching complete submission pages...'; notice('');
  state.snapshotId=null; state.selected=null; state.contest=null; state.rows=[]; state.review=null;
  document.body.classList.remove('reviewed');
  $('#run-button').classList.remove('complete');$('#run-label').textContent='Run review';$('#run-icon').textContent='↗';
  $('#review-stamp').hidden=true;document.querySelector('.summary-result').classList.remove('reviewed');
  $('#flag-label').innerHTML='Awaiting review <span class="summary-note">one pass, saved model</span>';
  $('#review-tab-count').textContent='—';$('#low-tab-count').textContent='—';$('#all-tab-count').textContent='0';
  $('#page-label').textContent='No contest loaded';
  $('#contest-title').textContent='Loading contest...'; $('#contest-meta').textContent='Codeforces API: fetching all submission pages';
  $('#participant-rows').innerHTML='<tr><td colspan="6" class="empty">Loading contest submissions...</td></tr>';
  $('#inspector').innerHTML='';
  ['participant-count','submission-count','flag-count'].forEach(key=>$('#'+key).textContent='...');
  document.querySelectorAll('.tab').forEach(tab=>{tab.disabled=true;tab.classList.toggle('active',tab.dataset.filter==='all');tab.setAttribute('aria-selected',String(tab.dataset.filter==='all'));});
  try {
    const snapshot=await api(`/api/snapshot?contestId=${id}`);
    state.contest={contest_id:snapshot.contest.id,name:snapshot.contest.name,start_time:snapshot.contest.startTimeSeconds,duration:snapshot.contest.durationSeconds,participants:snapshot.participants,submissions:snapshot.submissions,eligible:snapshot.eligible_participants,model:snapshot.model};
    state.rows=snapshot.rows; state.snapshotId=snapshot.snapshot_id; state.page=1; state.filter='all';state.query='';$('#search-input').value='';
    $('#contest-input').value=`https://codeforces.com/contest/${id}`;
    const url=new URL(location.href);url.searchParams.set('contest',id);history.replaceState(null,'',url);
    $('#load-status').textContent=`${number(snapshot.eligible_participants)} sessions within model scope. Fetched ${new Date(snapshot.snapshot_time).toLocaleTimeString()}`;
    paintOverview();loadList();
    if(!snapshot.eligible_participants)notice('No sessions with at least two attempted problems. Submission details are available, but no model decisions can be made.');
  } catch(error) {
    $('#load-status').textContent='Collection failed'; $('#contest-title').textContent='Could not load contest.';
    $('#contest-meta').textContent='Try another completed contest or retry shortly.';notice(error.message);
    $('#participant-rows').innerHTML='<tr><td colspan="6" class="empty">No contest loaded.</td></tr>';
  } finally {state.loading=false;$('#load-button').disabled=false;}
}
$('#contest-form').addEventListener('submit',event=>{event.preventDefault();openContest($('#contest-input').value);});
$('#run-button').addEventListener('click',runReview);
$('#participant-rows').addEventListener('click',event=>{const row=event.target.closest('tr[data-id]');if(row)selectParticipant(row.dataset.id);});
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>{state.filter=tab.dataset.filter;state.page=1;document.querySelectorAll('.tab').forEach(other=>{const active=other===tab;other.classList.toggle('active',active);other.setAttribute('aria-selected',String(active));});loadList();}));
let searchTimer; $('#search-input').addEventListener('input',event=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{state.query=event.target.value;state.page=1;loadList();},180);});
$('#prev-button').addEventListener('click',()=>{state.page--;loadList();}); $('#next-button').addEventListener('click',()=>{state.page++;loadList();});
$('#about-button').addEventListener('click',()=>$('#about-dialog').showModal()); $('#close-dialog').addEventListener('click',()=>$('#about-dialog').close());
$('#about-dialog').addEventListener('click',event=>{if(event.target===$('#about-dialog')){const r=event.target.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)event.target.close();}});
const requested=new URLSearchParams(location.search).get('contest');
if(requested)openContest(requested);
