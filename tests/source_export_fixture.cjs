// Controlled DOM fixture. This does not claim a successful live Helium export.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const script = fs.readFileSync(process.argv[2], 'utf8');

function run(options = {}) {
  const body = '  // λ\nint main() {\n\treturn 0;\n}\n\n';
  const link = href => ({getAttribute: key => key === 'href' ? href : null});
  const cells = [
    {tagName:'TD', querySelectorAll: () => [link('/profile/' + (options.author || 'Example'))]},
    {tagName:'TD', textContent:'GNU C++17'}
  ];
  const headers = [{textContent:'Who'},{textContent:'Lang'}];
  if (options.plainId) {
    cells.unshift({tagName:'TD',textContent:options.idText || '100'});
    headers.unshift({textContent:'#'});
  }
  const row = {children:cells, querySelectorAll: () => [
    ...(options.plainId ? [] : [link('/contest/2211/submission/100')]),
    link(options.problemHref || '/contest/2211/problem/F')]};
  const table = {querySelectorAll: selector => selector === 'tr th' ? headers : [row]};
  let artifact = null, filename = null, clicks = 0;
  const alerts = [];
  class LocalURL extends URL {
    static createObjectURL(blob) { artifact = blob.parts.join(''); return 'blob:fixture'; }
    static revokeObjectURL() {}
  }
  const document = {
    title:'Submission - Codeforces',
    querySelector: () => options.challenge ? {} : null,
    querySelectorAll: selector => selector === '#program-source-text' ? (options.missing ? [] : [{textContent:body,
      querySelectorAll: () => options.rendered ? [{textContent:'int main() {'},{textContent:'\u00a0'},{textContent:'\treturn 0;'}, {textContent:'}'}] : []}]) : [table],
    createElement: () => ({set download(value) {filename = value;}, click() {clicks++;}, remove() {}}),
    body: {appendChild() {}}
  };
  vm.runInNewContext(script, {
    document, location:{protocol:'https:',host:options.host || 'codeforces.com',pathname:options.path || '/contest/2211/submission/100'},
    URL:LocalURL, Blob:class {constructor(parts) {this.parts=parts;}},
    alert: text => alerts.push(text), setTimeout: callback => callback(), Date
  }, {timeout:1000});
  return {artifact:artifact && JSON.parse(artifact), filename, clicks, alerts, body};
}
const good = run();
assert.equal(good.clicks, 1);
assert.equal(good.filename, 'cf-source-2211-100.json');
assert.equal(good.artifact.snapshot.source, good.body);
assert.equal(good.artifact.snapshot.rows[0].authors[0], 'Example');
assert.equal(good.artifact.method, 'user_click_loaded_dom');
const plain = run({plainId:true});
assert.equal(plain.clicks, 1);
assert.equal(plain.artifact.snapshot.rows[0].submission_href, '/contest/2211/submission/100');
const rendered = run({plainId:true,rendered:true});
assert.equal(rendered.artifact.snapshot.source, 'int main() {\n\n\treturn 0;\n}');
assert.equal(rendered.artifact.snapshot.source_text_mode, 'rendered_lines_reconstructed');
for (const options of [{host:'example.com'},{path:'/contest/2211/submission/999'},{author:'Wrong'},{challenge:true},{missing:true}]) {
  const bad = run(options);
  assert.equal(bad.clicks, 0);
  assert.equal(bad.artifact, null);
  assert.equal(bad.alerts.length, 1);
}
for (const options of [{plainId:true,idText:'101'},{plainId:true,idText:'100 extra'}, {plainId:true,problemHref:'https://elsewhere.example/contest/2211/problem/F'}]) {
  const bad = run(options);
  assert.equal(bad.clicks, 0);
  assert.equal(bad.alerts.length, 1);
}
console.log('Exporter controlled DOM fixture passed.');
