/* Test the exact inline template renderers and routes without opening a browser.
 * Run with: node --test tests/test_power_ui.cjs [or set POWER_UI_DATA to a snapshot].
 */
const {readFileSync} = require('node:fs');
const {resolve} = require('node:path');
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {runInNewContext} = require('node:vm');

const root = resolve(__dirname, '..');
const html = readFileSync(resolve(root, 'ui/template.html'), 'utf8');
const data = JSON.parse(readFileSync(process.env.POWER_UI_DATA || resolve(root, 'output/app_data.json'), 'utf8'));
const script = html.slice(html.indexOf('/* ---------------- model top 25 ---------------- */'),
                         html.indexOf('/* ---------------- matchup lab ---------------- */'));
const helper = html.match(/const escapeText = [^\n]+/)[0];

function simulate(data, initialTab='bets'){
  class Element {
    constructor(value=''){this.value=value; this.handlers={}; this.hidden=true;}
    addEventListener(name, fn){this.handlers[name]=fn;}
    closest(){return this;}
    setAttribute(k,v){this[k]=v;}
    getAttribute(k){return this[k];}
    querySelector(){return {scrollIntoView(){}};}
  }
  const tabList = [...html.matchAll(/<button[^>]*data-tab="([^"]+)"/g)].map((m)=>{
    const e = new Element(); e.dataset={tab:m[1]}; return e;
  });
  const panels = [...html.matchAll(/class="panel" id="([^"]+)"/g)].map(m=>{
    const e = new Element();e.id=m[1];return e;
  });
  const selectors = Object.fromEntries(panels.map(e=>['#'+e.id,e]));
  const $ = key => selectors[key] ||= new Element(key==='#t25Depth'?'25':'change_since_last_week');
  const location={href:'https://example.test/sec-ranking-model/?tab='+initialTab};
  const context = {D:data,$,URL,location,document:{
    querySelectorAll(selector){return selector==='nav button[data-tab]'?tabList:panels;},
    getElementById(id){return $('#'+id);},
  },history:{replaceState(_state,_title,url){location.href=String(url);}},
  navigator:{clipboard:{async writeText(text){context.copied=text;}}}};
  runInNewContext(`
    const ACTIVE = D.inseason?.calibration || D.calibration;
    const currentRating = t => D.inseason ? t.current_rating : t.projected_rating;
    const sign = v => (v>0?'+':'') + v.toFixed(1);
    ${helper}
    ${script}
    ${html.slice(html.indexOf('/* ---------------- tabs ---------------- */'),html.indexOf("$('#themeBtn').addEventListener"))}
    activateTab(new URL(location.href).searchParams.get('tab'), false);
  `, context);
  return {$,location,context,tabList};
}
function powerRows($){
  const decode = s => s.replace(/&(amp|lt|gt|quot|#39);/g, (_,e) => ({amp:'&',lt:'<',gt:'>',quot:'"', '#39':"'"}[e]));
  return [...$('#powerTable tbody').innerHTML.matchAll(/<tr><td>(\d+)<\/td><th scope="row">([^<]+)<\/th><td class="note">([^<]+)<\/td><td class="num">([^<]+)<\/td><td class="num">([^<]+)<\/td><\/tr>/g)].map(
    ([_,rank,team,conference,points,games])=>({rank:+rank,team:decode(team),conference:decode(conference),points:+points,games}));
}

test('all rated teams sorted, centered and scaled to Matchup Lab neutral points',()=>{
  const {$}=simulate(data,'power');
  const rows=powerRows($);
  const active=data.inseason?.calibration || data.calibration;
  const key=data.inseason?'current_rating':'projected_rating';
  const ordered=data.all_ratings.filter(t=>Number.isFinite(t[key])).sort((a,b)=>b[key]-a[key]||a.team.localeCompare(b.team));
  assert.equal(rows.length,ordered.length);
  assert.ok(rows.length>100);
  assert.deepEqual(rows.map(r=>r.team),ordered.map(r=>r.team));
  assert.equal($('#powerResults').hidden,false);
  const mean = rows.reduce((sum,r)=>sum+r.points,0)/rows.length;
  assert.ok(Math.abs(mean)<=.05);
  const expected=active.rating_diff_coef*(ordered[0][key]-ordered.at(-1)[key]);
  assert.ok(Math.abs(rows[0].points-rows.at(-1).points-expected)<=.101);
  assert.equal(rows[0].games,String(data.inseason?ordered[0].games_rated:0));
});

test('distinct panels route, switch, and share through existing query parameter',async()=>{
  const {$,location,context,tabList}=simulate(data,'power');
  assert.equal($('#p-power').hidden,false);
  assert.equal($('#p-top25').hidden,true);
  const find=id=>tabList.find(t=>t.dataset.tab===id);
  assert.equal(find('power').getAttribute('aria-selected'),'true');
  await $('#sharePage').handlers.click();
  assert.equal(context.copied,'https://example.test/sec-ranking-model/?tab=power');
  assert.equal(simulate(data,new URL(context.copied).searchParams.get('tab')).$('#p-power').hidden,false);
  find('top25').handlers.click();
  assert.equal($('#p-top25').hidden,false);
  assert.equal($('#p-power').hidden,true);
  assert.equal(new URL(location.href).searchParams.get('tab'),'top25');
  let canceled=false;
  $('#t25PowerLink').handlers.click({button:0,preventDefault:()=>{canceled=true;}});
  assert.equal(canceled,true);
  assert.equal($('#p-power').hidden,false);
  assert.equal(simulate(data,'not-a-tab').$('#p-bets').hidden,false);
});

test('strict and old snapshots stay distinct, and missing predictive data is not mislabeled',()=>{
  const fallback=simulate({...data,top25_basis:'current_rating'});
  assert.match(fallback.$('#t25Note').textContent,/does not contain results-only rankings/);
  const strict=simulate({...data,top25_basis:'strict_results',national_ranking:[]});
  assert.match(strict.$('#t25Note').textContent,/No completed games/);
  const preseason=simulate({...data,inseason:null});
  assert.match(preseason.$('#powerNote').textContent,/preseason only/);
  assert.equal(powerRows(preseason.$)[0].games,'0');
  const broken=simulate({...data,inseason:{calibration:{rating_diff_coef:null}},all_ratings:data.all_ratings});
  assert.match(broken.$('#powerNote').textContent,/unavailable/);
  assert.equal(broken.$('#powerResults').hidden,true);
});

if(data.top25_basis==='strict_results' && data.national_ranking.length){
  test('live earned rows keep their exact ordering and are not power ratings',()=>{
    const {$}=simulate(data,'top25');
    const rows=[...$('#t25Table tbody').innerHTML.matchAll(/<tr><td>\d+<\/td><td>.*?<\/td><td>([^<]+)<\/td>/g)].map(m=>m[1]);
    assert.deepEqual(rows,data.national_ranking.slice(0,25).map(r=>r.team));
    assert.notDeepEqual(rows,powerRows($).slice(0,25).map(r=>r.team));
  });
}
